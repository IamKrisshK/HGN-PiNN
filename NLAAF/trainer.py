import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, cfg, model, dataloader, dist, wrapper):
        self.cfg        = cfg
        self.model      = model
        self.dataloader = dataloader
        self.dist       = dist
        self.wrapper    = wrapper
        self.device     = dist.device
        self.amp        = cfg.amp

        self.use_physics    = cfg.get("use_physics_loss", True)
        self.physics_weight = cfg.get("physics_loss_weight", 1.0)
        self.delta_t        = cfg.get("delta_t", 1200.0)
        self.noise_type     = cfg.noise_type

        self.criterion = nn.MSELoss()
        laaf_lr_scale = float(cfg.get("laaf_lr_scale", 1.0))
        laaf_params   = [p for n, p in model.named_parameters() if n.endswith(".a")]
        base_params   = [p for n, p in model.named_parameters() if not n.endswith(".a")]

        param_groups = [{"params": base_params, "lr": cfg.lr}]
        if laaf_params:
            param_groups.append({"params": laaf_params, "lr": cfg.lr * laaf_lr_scale})
        use_fused = (self.device.type == "cuda")
        self.optimizer = torch.optim.Adam(
            param_groups,
            lr=cfg.lr,
            fused=use_fused,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda e: cfg.lr_decay_rate ** e,
        )
        self.scaler = GradScaler(device=self.device.type, enabled=self.amp)

        if self.dist.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[self.dist.local_rank],
                output_device=self.device,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        self.model.to(self.device)
        self.model.train()
    def info(self, msg: str):
        if self.dist.rank == 0:
            print(f"[Trainer] {msg}")
    def _raw_model(self):
        """Return the underlying model, unwrapping DDP if necessary."""
        return self.model.module if isinstance(self.model, DDP) else self.model
    def chunk_graph(self, graph, chunk_size: int = 5000):
        """Yield node-induced subgraphs of at most `chunk_size` nodes."""
        N = graph.num_nodes
        for i in range(0, N, chunk_size):
            mask = torch.arange(i, min(i + chunk_size, N), device=graph.x.device)
            yield graph.subgraph(mask)

    def forward_step(self, batch):
        if self.use_physics:
            graph, physics = batch
            physics = {k: v.to(self.device, non_blocking=True) for k, v in physics.items()}
        else:
            graph   = batch
            physics = None

        graph = graph.to(self.device, non_blocking=True)
        graph = self.wrapper.apply_fourier(graph)

        chunk_size  = self.cfg.get("chunk_size", 5000)
        total_loss  = torch.tensor(0.0, device=self.device)
        loss_dict   = {}
        num_chunks  = 0

        for subgraph in self.chunk_graph(graph, chunk_size=chunk_size):
            with autocast(device_type=self.device.type, enabled=self.amp):
                pred = self.model(
                    subgraph.x,
                    getattr(subgraph, "edge_attr", None),
                    subgraph,
                )
                loss = self.criterion(pred, subgraph.y)

                if self.cfg.get("use_pushforward", False):
                    stab_loss = self.pushforward_pass(subgraph)
                    loss = loss + self.cfg.get("pushforward_weight", 0.1) * stab_loss

                if self.use_physics and physics is not None:
                    phy_loss = self.compute_physics(pred, physics, subgraph)
                    loss     = loss + self.physics_weight * phy_loss
                    loss_dict[f"physics_loss_chunk_{num_chunks}"] = phy_loss.detach()

            total_loss = total_loss + loss
            num_chunks += 1

        total_loss = total_loss / max(num_chunks, 1)
        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict

    def pushforward_pass(self, graph) -> torch.Tensor:
        """
        Temporal stability regularisation (pushforward trick).

        Optimised: only ONE differentiable forward pass (pred_stab2) contributes
        gradients. The two intermediate predictions use torch.no_grad() to avoid
        storing activations for three backward passes.

        Steps:
          1. no_grad: predict from t=0..T-1 features  → pred_stab  (detached)
          2. no_grad: roll the state forward one step  → X_stab2    (detached)
          3. grad:    predict from X_stab2             → pred_stab2 (differentiable)
          4. loss: MSE(pred_stab2, graph.y)
        """
        X         = graph.x
        n_static  = self.cfg.get("n_static_features", 12)
        if self.wrapper.use_fourier:
            n_static = 2 * self.wrapper.fourier_dim + (n_static - self.wrapper.coord_dim)

        n_time  = (X.shape[1] - n_static) // 2
        static  = X[:, :n_static]
        depth   = X[:, n_static : n_static + n_time]
        volume  = X[:, n_static + n_time :]
        with torch.no_grad():
            X_stab    = torch.cat([static, depth[:, :-1], volume[:, :-1]], dim=1)
            pred_stab = self.model(X_stab, graph.edge_attr, graph)
            depth_new = depth[:, 1:2] + pred_stab[:, 0:1]
            vol_new   = volume[:, 1:2] + pred_stab[:, 1:2]

        X_stab2   = torch.cat([static, depth_new, vol_new], dim=1)
        pred_stab2 = self.model(X_stab2, graph.edge_attr, graph)

        return self.criterion(pred_stab2, graph.y)

    def compute_physics(self, pred, physics, graph) -> torch.Tensor:
        from hgn.utils import compute_physics_loss
        return compute_physics_loss(pred, physics, graph, delta_t=self.delta_t)

    def backward(self, loss: torch.Tensor):
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

  def step(self, batch, step_idx: int):
        """
        Single training step with corrected gradient accumulation.

        Fix vs original: zero_grad, backward, and optimizer.step are all
        gated on the SAME modulo condition so the effective batch size is
        exactly batch_size × grad_accum_steps regardless of step index.
        """
        accum = self.cfg.get("grad_accum_steps", 1)
        if step_idx % accum == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss, loss_dict = self.forward_step(batch)
        scaled_loss     = loss / accum
        self.backward(scaled_loss)
        if (step_idx + 1) % accum == 0:
            if self.amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()

        return loss.detach(), loss_dict

    def save_checkpoint(self, epoch: int, ckpt_path: str = None):
        """
        Save model + optimizer + scaler + scheduler.

        Unwraps DDP so the checkpoint contains a plain state dict compatible
        with single-GPU inference and resuming on a different world_size.
        """
        if ckpt_path is None:
            ckpt_path = getattr(self.cfg, "ckpt_path", "checkpoint.pt")
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        state = {
            "epoch":             epoch,
            "model_state_dict":  self._raw_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
        }
        torch.save(state, ckpt_path)
        if self.dist.rank == 0:
            self.info(f"Checkpoint saved → {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str = None) -> int:
        """Load model + optimizer + scaler + scheduler. Returns next epoch."""
        if ckpt_path is None:
            ckpt_path = getattr(self.cfg, "ckpt_path", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            self.info("No checkpoint found — starting from scratch.")
            return 0

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self._raw_model().load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])

        self.info(f"Checkpoint loaded from epoch {state['epoch']}.")
        return state["epoch"] + 1

    def train(self, start_epoch: int = 0, end_epoch: int = None):
        if end_epoch is None:
            end_epoch = self.cfg.epochs

        for epoch in range(start_epoch, end_epoch):
            self.dataloader.sampler.set_epoch(epoch)
            total_loss = 0.0
            count      = 0

            for step_idx, batch in enumerate(self.dataloader):
                loss, loss_dict = self.step(batch, step_idx)
                total_loss     += loss.item()
                count          += 1

            avg = total_loss / max(count, 1)
            if self.dist.rank == 0:
                self.info(f"Epoch {epoch:04d} | avg loss: {avg:.4e}")
                self.save_checkpoint(epoch)
