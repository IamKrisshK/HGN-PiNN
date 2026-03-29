import torch
import os
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from mopper import Mopper
class Trainer:
    def __init__(self, cfg, model, dataloader, dist, wrapper):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.dist = dist
        self.wrapper = wrapper
        self.device = dist.device
        self.amp = cfg.amp

        self.use_physics = cfg.get("use_physics_loss", True)
        self.physics_weight = cfg.get("physics_loss_weight", 1.0)
        self.delta_t = cfg.get("delta_t", 1200.0)

        self.noise_type = cfg.noise_type

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: cfg.lr_decay_rate ** e
        )

        self.scaler = GradScaler(enabled=self.amp)

        if self.dist.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.device,
                find_unused_parameters=False,
            )

        self.model.to(self.device)
        self.model.train()
    def info(self, msg):
        print(msg)
    def chunk_graph(self, graph, chunk_size=5000):
        N = graph.num_nodes
    
        for i in range(0, N, chunk_size):
            mask = torch.arange(i, min(i + chunk_size, N), device=graph.x.device)
            subgraph = graph.subgraph(mask)
            yield subgraph
            
    def forward_step(self, batch):
        if self.use_physics:
            graph, physics = batch
            physics = {k: v.to(self.device) for k, v in physics.items()}
        else:
            graph = batch
            physics = None
    
        graph = graph.to(self.device, non_blocking=True)
        graph = self.wrapper.apply_fourier(graph)
    
        total_loss = 0.0
        loss_dict = {}
        num_chunks = 0
        for subgraph in self.chunk_graph(graph, chunk_size=self.cfg.get("chunk_size", 5000)):
            with autocast(device_type=self.device.type, enabled=self.amp):
                pred = self.model(subgraph.x, getattr(subgraph, "edge_attr", None), subgraph)
                loss = self.criterion(pred, subgraph.y)
                if self.cfg.get("use_pushforward", False):
                    _, stab_loss = self.pushforward_pass(subgraph)
                    loss = loss + self.cfg.get("pushforward_weight", 0.1) * stab_loss
                if self.use_physics and physics is not None:
                    phy_loss = self.compute_physics(pred, physics, subgraph)
                    loss += self.physics_weight * phy_loss
                    loss_dict[f"physics_loss_chunk_{num_chunks}"] = phy_loss.detach()
                total_loss += loss
            num_chunks += 1
        total_loss /= max(num_chunks, 1)
        loss_dict["total_loss"] = total_loss.detach()
        return total_loss, loss_dict
        
    def pushforward_pass(self, graph):
        X = graph.x
        n_static = self.cfg.get("n_static_features", 12)
        if self.wrapper.use_fourier:
            n_static = 2 * self.wrapper.fourier_dim + (n_static - self.wrapper.coord_dim)
        n_time = (X.shape[1] - n_static) // 2
        static = X[:, :n_static]
        depth = X[:, n_static : n_static + n_time]
        volume = X[:, n_static + n_time :]
        X_one = torch.cat([static, depth[:, 1:], volume[:, 1:]], dim=1)
        pred_one = self.model(X_one, graph.edge_attr, graph)
        with torch.no_grad():
            X_stab = torch.cat([static, depth[:, :-1], volume[:, :-1]], dim=1)
            pred_stab = self.model(X_stab, graph.edge_attr, graph)

            depth_new = depth[:, 1:2] + pred_stab[:, 0:1]
            vol_new = volume[:, 1:2] + pred_stab[:, 1:2]

        X_stab2 = torch.cat([static, depth_new, vol_new], dim=1)
        pred_stab2 = self.model(X_stab2, graph.edge_attr, graph)

        stability_loss = self.criterion(pred_stab2, graph.y)

        return pred_one, stability_loss
    def compute_physics(self, pred, physics, graph):
        from hgn.utils import compute_physics_loss

        return compute_physics_loss(
            pred, physics, graph, delta_t=self.delta_t
            )
    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, batch, step_idx):
        if step_idx % self.cfg.get("grad_accum_steps", 1) == 0:
            self.optimizer.zero_grad(set_to_none=True)
    
        loss, loss_dict = self.forward_step(batch)
        loss = loss / self.cfg.get("grad_accum_steps", 1)
        self.backward(loss)
    
        if (step_idx + 1) % self.cfg.get("grad_accum_steps", 1) == 0:
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
    
        return loss.detach(), loss_dict
    def save_checkpoint(self, epoch, ckpt_path=None):
        """Save model + optimizer + scaler + scheduler states."""
        if ckpt_path is None:
            ckpt_path = getattr(self.cfg, "ckpt_path", "checkpoint.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        torch.save(state, ckpt_path)
        if self.dist.rank == 0:
            self.info(f"Checkpoint saved at {ckpt_path}")
    def load_checkpoint(self, ckpt_path=None):
        """Load model + optimizer + scaler + scheduler states."""
        if ckpt_path is None:
            ckpt_path = getattr(self.cfg, "ckpt_path", None)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            self.info("No checkpoint found, starting from scratch.")
            return 0

        state = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])

        self.info(f"Checkpoint loaded from epoch {state['epoch']}")
        return state["epoch"] + 1
    def train(self, start_epoch=0, end_epoch=None):
        if end_epoch is None:
            end_epoch = self.cfg.epochs
        for epoch in range(start_epoch, end_epoch):
            self.dataloader.sampler.set_epoch(epoch)
            total = 0.0
            count = 0
            for step_idx, batch in enumerate(self.dataloader):
                loss, loss_dict = self.step(batch, step_idx)
                total += loss.item()
                count += 1
            avg = total / max(count, 1)
            if self.dist.rank == 0:
                self.info(f"Epoch {epoch} | Loss: {avg:.4e}")
                self.save_checkpoint(epoch)
