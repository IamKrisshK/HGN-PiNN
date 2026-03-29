import torch
import torch.nn as nn
import math
import torch_geometric as pyg
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import to_absolute_path
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.launch.utils import load_checkpoint

class FourierFeatureEncoder(nn.Module):
    def __init__(self, in_dim, mapping_size=32, scale=5.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)
    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, physics = zip(*batch)
        physics_d = {
            key: torch.stack([torch.as_tensor(p[key]) for p in physics])
            for key in physics[0].keys()
        }
        return pyg.data.Batch.from_data_list(list(graphs)), physics_d
    return pyg.data.Batch.from_data_list(list(graphs))

class Mopper:
    def __init__(self, cfg, dist):
        self.cfg = cfg
        self.dist = dist
        self.device = dist.device
        self.use_physics = cfg.get("use_physics_loss", False)
        self.use_fourier = cfg.get("use_fourier", False)
        
        if self.use_fourier:
            self.fourier_dim = cfg.fourier_dim
            self.fourier_scale = cfg.fourier_scale
            #2 dims = spatial coords
            self.coord_dim = cfg.get("coord_dim", 12)
            self.fourier = FourierFeatureEncoder(
                in_dim=self.coord_dim,
                mapping_size=self.fourier_dim,
                scale=self.fourier_scale,
            ).to(self.device)
    def info(self, msg):
        print(msg)
    def apply_fourier(self, graph):
        if not self.use_fourier:
            return graph
        x = graph.x
        x_proj = (2 * math.pi * x[:, :self.coord_dim]) @ self.fourier.B
        graph.x = torch.cat([x_proj.sin(), x_proj.cos(), x[:, self.coord_dim:]], dim=-1)
        return graph
    def build_dataloader(self):
        self.info("Initializing dataset...")
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
        if self.cfg.num_dataloader_workers:
            dataloader = PyGDataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=sampler,
                num_workers=self.cfg.num_dataloader_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            dataloader = PyGDataLoader(
                dataset,
                batch_size=self.cfg.batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
                collate_fn=collate_fn,
            )

        self.info("Dataloader ready.")
        return dataloader
    def build_model(self):
        self.info("Building MeshGraphKAN model...")

        mlp_act = "silu" if self.cfg.recompute_activation else "relu"
        input_dim = self.cfg.num_input_features
        if self.use_fourier:
            input_dim = 2 * self.fourier_dim + (input_dim - self.coord_dim)
        model = MeshGraphKAN(
            input_dim,
            self.cfg.num_edge_features,
            self.cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=self.cfg.do_concat_trick,
            num_processor_checkpoint_segments=self.cfg.num_processor_checkpoint_segments,
            recompute_activation=self.cfg.recompute_activation,
        )

        if self.cfg.jit:
            if not model.meta.jit:
                raise ValueError("Model not JIT compatible.")
            model = torch.jit.script(model)

        model = model.to(self.device)

        self.info("Model ready.")
        return model
    def load_checkpoint(self, model, optimizer=None, scheduler=None, scaler=None):
        self.info("Loading checkpoint...")
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

        self.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    def build_all(self, optimizer=None, scheduler=None, scaler=None):
        dataloader = self.build_dataloader()
        model = self.build_model()
        epoch = 0
        if self.cfg.get("ckpt_path", None):
            epoch = self.load_checkpoint(model, optimizer, scheduler, scaler)
        return model, dataloader, epoch
