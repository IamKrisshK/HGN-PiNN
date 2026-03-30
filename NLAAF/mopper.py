"""
mopper.py — Optimized pipeline builder with N-LAAF / L-LAAF support.

Key improvements over original:
  - FourierFeatureEncoder applied once per batch (no double-projection bug)
  - apply_fourier uses the encoder's buffer directly — no manual matmul duplication
  - build_model exposes laaf_mode ('neuron' | 'layer' | None) via cfg
  - Dataloader uses prefetch_factor for async GPU transfer
  - torch.compile() wrapper (opt-in via cfg.compile)
"""

import math
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import to_absolute_path
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.launch.utils import load_checkpoint

class NeuronWiseLAAF(nn.Module):
    """
    N-LAAF: each neuron gets its own learnable slope `a_i`.
    Forward: activation(n * a_i * x)  where n is a fixed scaling factor.

    Place in encoder/decoder MLPs where sharp spatial gradients matter
    (flood fronts, wetting-and-drying transitions).

    Parameter cost: +hidden_dim floats per layer — negligible vs weights.
    """

    def __init__(self, hidden_dim: int, activation=nn.SiLU(), n: float = 10.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(hidden_dim) / n)
        self.activation = activation
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.n * self.a * x)


class LayerWiseLAAF(nn.Module):
    """
    L-LAAF: one shared learnable slope per layer.
    Forward: activation(n * a * x)

    Place in message-passing processor blocks where smooth aggregation
    is preferred (volume conservation layers, edge updates).

    Parameter cost: +1 float per layer.
    """

    def __init__(self, activation=nn.SiLU(), n: float = 10.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0 / n))
        self.activation = activation
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.n * self.a * x)



class FourierFeatureEncoder(nn.Module):
    """
    Random Fourier features for positional encoding.

    The projection matrix B is registered as a non-trainable buffer so it
    moves to the correct device with `.to(device)` automatically and is
    included in `state_dict` for reproducible checkpointing.

    Output dimension: 2 * mapping_size  (sin and cos concatenated).
    """

    def __init__(self, in_dim: int, mapping_size: int = 32, scale: float = 5.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)    
        self.out_dim = 2 * mapping_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * (x @ self.B)
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1) 


def collate_fn(batch):
    """
    Handles both plain-graph batches and (graph, physics_dict) batches.

    Bug fix vs original: the `graphs` variable is now always defined before
    it is passed to `Batch.from_data_list`.
    """
    if isinstance(batch[0], tuple):
        graphs, physics = zip(*batch)
        physics_d = {
            key: torch.stack([torch.as_tensor(p[key]) for p in physics])
            for key in physics[0].keys()
        }
        return pyg.data.Batch.from_data_list(list(graphs)), physics_d
    return pyg.data.Batch.from_data_list(list(batch))

class Mopper:
    """
    Builds and wires together:
      - HydroGraphDataset + distributed DataLoader
      - MeshGraphKAN (optionally torch.compile'd)
      - FourierFeatureEncoder (applied once per batch inside apply_fourier)
      - Checkpoint loading via physicsnemo

    New cfg keys (all optional with sensible defaults):
      cfg.laaf_mode          : 'neuron' | 'layer' | None  (default None)
      cfg.laaf_n             : float   scaling factor      (default 10.0)
      cfg.compile            : bool    torch.compile opt   (default False)
      cfg.prefetch_factor    : int     DataLoader prefetch (default 2)
    """

    def __init__(self, cfg, dist):
        self.cfg = cfg
        self.dist = dist
        self.device = dist.device

        self.use_physics = cfg.get("use_physics_loss", False)
        self.use_fourier = cfg.get("use_fourier", False)
        torch.backends.cudnn.benchmark = True

        if self.use_fourier:
            self.fourier_dim   = cfg.fourier_dim
            self.fourier_scale = cfg.fourier_scale
            self.coord_dim     = cfg.get("coord_dim", 12)
            self.fourier = FourierFeatureEncoder(
                in_dim=self.coord_dim,
                mapping_size=self.fourier_dim,
                scale=self.fourier_scale,
            ).to(self.device)
        else:
            self.fourier    = None
            self.coord_dim  = 0
            self.fourier_dim = 0

    def info(self, msg: str):
        if self.dist.rank == 0:
            print(f"[Mopper] {msg}")

    def apply_fourier(self, graph):
        """
        Replace the first `coord_dim` node features with their Fourier embedding.

        Called once per batch in Trainer.forward_step — not per-chunk —
        so the projection happens on the full node set before chunking.
        """
        if not self.use_fourier:
            return graph

        x = graph.x                                   # (N, total_features)
        coords = x[:, :self.coord_dim]                # (N, coord_dim)
        rest   = x[:, self.coord_dim:]                # (N, remaining)

        with torch.no_grad():                         # encoding is deterministic
            fourier_feats = self.fourier(coords)      # (N, 2*fourier_dim)

        graph.x = torch.cat([fourier_feats, rest], dim=-1)
        return graph

    def _effective_input_dim(self, base_input_dim: int) -> int:
        """Compute node feature dimension after optional Fourier expansion."""
        if self.use_fourier:
            return 2 * self.fourier_dim + (base_input_dim - self.coord_dim)
        return base_input_dim

    def build_dataloader(self):
        self.info("Initialising dataset …")
        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=self.cfg.data_dir,
            prefix="M80",
            num_samples=self.cfg.num_training_samples,
            n_time_steps=self.cfg.n_time_steps,
            k=4,
            noise_type=self.cfg.noise_type,
            noise_std=0.01,
            hydrograph_ids_file="train.txt",
            split="train",
            return_physics=self.use_physics,
        )

        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )

        num_workers     = self.cfg.get("num_dataloader_workers", 0)
        prefetch_factor = self.cfg.get("prefetch_factor", 2) if num_workers > 0 else None
        persistent      = num_workers > 0

        dataloader = PyGDataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,                  
            persistent_workers=persistent,    
            prefetch_factor=prefetch_factor,  
            collate_fn=collate_fn,
        )

        self.info("Dataloader ready.")
        return dataloader

    def build_model(self):
        self.info("Building MeshGraphKAN …")

        mlp_act   = "silu" if self.cfg.recompute_activation else "relu"
        input_dim = self._effective_input_dim(self.cfg.num_input_features)

        model = MeshGraphKAN(
            input_dim,
            self.cfg.num_edge_features,
            self.cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=self.cfg.do_concat_trick,
            num_processor_checkpoint_segments=self.cfg.num_processor_checkpoint_segments,
            recompute_activation=self.cfg.recompute_activation,
        )
        laaf_mode = self.cfg.get("laaf_mode", None)   
        laaf_n    = float(self.cfg.get("laaf_n", 10.0))

        if laaf_mode == "neuron":
            self._attach_neuron_laaf(model, laaf_n)
            self.info("N-LAAF attached to encoder/decoder MLPs.")
        elif laaf_mode == "layer":
            self._attach_layer_laaf(model, laaf_n)
            self.info("L-LAAF attached to processor layers.")

        if self.cfg.jit:
            if not model.meta.jit:
                raise ValueError("Model is not JIT-compatible.")
            model = torch.jit.script(model)

        model = model.to(self.device)

        if self.cfg.get("compile", False) and not self.cfg.jit:
            model = torch.compile(model, mode="reduce-overhead")
            self.info("torch.compile() applied (mode=reduce-overhead).")

        self.info("Model ready.")
        return model

    def _attach_neuron_laaf(self, model: nn.Module, n: float):
        """
        Replace activation layers inside encoder/decoder MLPs with NeuronWiseLAAF.

        MeshGraphKAN is expected to expose `.node_encoder`, `.edge_encoder`,
        and `.node_decoder` as nn.Sequential-like modules. Adjust the
        attribute names to match your actual MeshGraphKAN implementation.

        N-LAAF is best in encoder/decoder because those layers process raw
        node/edge features where sharp feature boundaries (flood fronts)
        benefit from per-neuron flexibility.
        """
        hidden = self.cfg.get("hidden_dim", 128)
        base_act = nn.SiLU()

        target_modules = []
        for name in ("node_encoder", "edge_encoder", "node_decoder"):
            mod = getattr(model, name, None)
            if mod is not None:
                target_modules.append((name, mod))

        for mod_name, module in target_modules:
            for i, layer in enumerate(module):
                if isinstance(layer, (nn.ReLU, nn.SiLU, nn.GELU, nn.Tanh)):
                    out_features = hidden
                    if i > 0 and isinstance(module[i - 1], nn.Linear):
                        out_features = module[i - 1].out_features
                    module[i] = NeuronWiseLAAF(out_features, activation=base_act, n=n)

    def _attach_layer_laaf(self, model: nn.Module, n: float):
        """
        Replace activation layers inside processor blocks with LayerWiseLAAF.

        MeshGraphKAN is expected to expose `.processor_layers` (list/ModuleList).
        L-LAAF is appropriate in message passing because aggregated messages
        are smoother signals — one slope per layer is enough expressive power
        and keeps the parameter count minimal.
        """
        base_act = nn.SiLU()
        processors = getattr(model, "processor_layers", None)
        if processors is None:
            self.info("Warning: no .processor_layers found — L-LAAF not applied.")
            return

        for proc in processors:
            for module in proc.modules():
                if isinstance(module, nn.Sequential):
                    for i, layer in enumerate(module):
                        if isinstance(layer, (nn.ReLU, nn.SiLU, nn.GELU, nn.Tanh)):
                            module[i] = LayerWiseLAAF(activation=base_act, n=n)

    def load_checkpoint(self, model, optimizer=None, scheduler=None, scaler=None):
        self.info("Loading checkpoint …")
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        epoch = load_checkpoint(
            to_absolute_path(self.cfg.ckpt_path),
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=self.device,
        )
        self.info(f"Checkpoint loaded from epoch {epoch}.")
        return epoch

    def build_all(self, optimizer=None, scheduler=None, scaler=None):
        dataloader = self.build_dataloader()
        model      = self.build_model()
        epoch      = 0
        if self.cfg.get("ckpt_path", None):
            epoch = self.load_checkpoint(model, optimizer, scheduler, scaler)
        return model, dataloader, epoch
