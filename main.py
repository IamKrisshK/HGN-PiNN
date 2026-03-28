import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class trainer:
    def __init__(self, cfg, model, dataloader, dist, logger):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.dist = dist
        self.logger = logger

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

    def step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        loss, loss_dict = self.forward_step(batch)

        self.backward(loss)

        return loss.detach(), loss_dict

    def forward_step(self, batch):
        if self.use_physics:
            graph, physics = batch
            physics = {k: v.to(self.device) for k, v in physics.items()}
        else:
            graph = batch
            physics = None

        graph = graph.to(self.device)

        with autocast(device_type=self.device.type, enabled=self.amp):
            if self.noise_type == "pushforward":
                pred, aux_loss = self.pushforward_pass(graph)
            else:
                pred = self.model(graph.x, graph.edge_attr, graph)
                aux_loss = 0.0

            mse_loss = self.criterion(pred, graph.y)
            total_loss = mse_loss + aux_loss

            loss_dict = {
                "mse_loss": mse_loss.detach(),
                "aux_loss": torch.tensor(aux_loss).detach()
                if not isinstance(aux_loss, torch.Tensor)
                else aux_loss.detach(),
            }
            if self.use_physics and physics is not None:
                phy_loss = self.compute_physics(pred, physics, graph)
                total_loss += self.physics_weight * phy_loss
                loss_dict["physics_loss"] = phy_loss.detach()

            loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict
    def pushforward_pass(self, graph):
        X = graph.x

        n_static = 12
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
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
    def train_epochs(self, epochs):
        for epoch in range(epochs):
            total = 0.0
            count = 0

            for batch in self.dataloader:
                loss, _ = self.step(batch)
                total += loss.item()
                count += 1

            avg = total / max(count, 1)

            if self.dist.rank == 0:
                self.logger.info(f"Epoch {epoch} | Loss: {avg:.4e}")

            self.scheduler.step()
