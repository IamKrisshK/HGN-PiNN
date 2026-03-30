import os
import math
import logging
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["font.size"]   = 20
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch_geometric.utils import to_networkx
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN



class Inferencer:
    def __init__(self, cfg: dict):
        self.cfg          = cfg
        self.device       = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        self.rollout_len  = cfg["num_test_time_steps"]
        self.n_time_steps = cfg["n_time_steps"]
        self.use_fourier  = cfg.get("use_fourier", False)

        os.makedirs(cfg["animation_output_dir"], exist_ok=True)

        if self.use_fourier:
            torch.manual_seed(cfg["fourier_seed"])
            self.B = (
                torch.randn(cfg["coord_dim"], cfg["fourier_dim"]) * cfg["fourier_scale"]
            ).to(self.device)
            self._model_input_dim = (
                2 * cfg["fourier_dim"] + (cfg["num_input_features"] - cfg["coord_dim"])
            )
        else:
            self._model_input_dim = cfg["num_input_features"]

        self.dataset = self._build_dataset()
        self.model   = self._build_model()

    def _build_dataset(self):
        return HydroGraphDataset(
            data_dir=self.cfg["test_dir"],
            prefix=self.cfg["prefix"],
            n_time_steps=self.cfg["n_time_steps"],
            hydrograph_ids_file=self.cfg["test_ids_file"],
            split="test",
            rollout_length=self.rollout_len,
            return_physics=False,
        )

    def _build_model(self):
        model = MeshGraphKAN(
            self._model_input_dim,
            self.cfg["num_edge_features"],
            self.cfg["num_output_features"],
        ).to(self.device)

        ckpt_path = self.cfg["ckpt_path"]
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state["model_state_dict"])
            print(f"Checkpoint loaded from epoch {state['epoch']}")
        else:
            print("No checkpoint found, using random weights.")

        model.eval()
        return model

    def _apply_fourier(self, x: torch.Tensor) -> torch.Tensor:
        coords  = x[:, :self.cfg["coord_dim"]]
        rest    = x[:, self.cfg["coord_dim"]:]
        x_proj  = (2 * math.pi * coords) @ self.B
        return torch.cat([x_proj.sin(), x_proj.cos(), rest], dim=-1)

    def _rollout(self, g, rollout_data):
        edge_features = g.edge_attr
        X_iter        = g.x
        num_nodes     = X_iter.size(0)
        n             = self.n_time_steps

        inflow_seq = rollout_data["inflow"].to(self.device)
        precip_seq = rollout_data["precipitation"].to(self.device)
        wd_gt_seq  = rollout_data["water_depth_gt"].to(self.device)

        preds, gts, rmses = [], [], []

        for t in range(self.rollout_len):
            static  = X_iter[:, :12]
            wd_win  = X_iter[:, 12 : 12 + n]
            vol_win = X_iter[:, 12 + n : 12 + 2 * n]

            X_input = torch.cat([static, wd_win, vol_win], dim=1)
            if self.use_fourier:
                X_input = self._apply_fourier(X_input)

            pred    = self.model(X_input, edge_features, g)

            new_wd  = wd_win[:,  -1:] + pred[:, 0:1]
            new_vol = vol_win[:, -1:] + pred[:, 1:2]

            static_upd = static.clone()
            static_upd[:, 10:12] = torch.cat([
                inflow_seq[t].unsqueeze(0).expand(num_nodes, 1),
                precip_seq[t].unsqueeze(0).expand(num_nodes, 1),
            ], dim=1)

            X_iter = torch.cat([
                static_upd,
                torch.cat([wd_win[:,  1:], new_wd],  dim=1),
                torch.cat([vol_win[:, 1:], new_vol], dim=1),
            ], dim=1)

            preds.append(new_wd.squeeze(1).cpu())
            gts.append(wd_gt_seq[t].cpu())
            rmses.append(
                torch.sqrt(torch.mean((new_wd.squeeze(1) - wd_gt_seq[t]) ** 2)).item()
            )

        return preds, gts, rmses

    def _animate(self, preds, gts, g_snapshot, rmses, path):
        fig, axes = plt.subplots(2, 2, figsize=(30, 30))
        cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
        cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
        cax3 = fig.add_axes([0.05, 0.10, 0.02, 0.35])

        feats      = g_snapshot.x
        pos        = {i: (feats[i, 0].item(), feats[i, 1].item()) for i in range(feats.size(0))}
        all_vals   = torch.cat(preds + gts)
        vmin, vmax = all_vals.min().item(), all_vals.max().item()
        tps        = self.cfg["time_per_step"]
        gr         = to_networkx(g_snapshot).to_undirected()

        def _panel(ax, cax, vals, title):
            nd = nx.draw_networkx_nodes(
                gr, pos, node_color=vals, node_size=250,
                cmap=plt.cm.viridis, ax=ax, vmin=vmin, vmax=vmax, node_shape="s",
            )
            nx.draw_networkx_edges(gr, pos, alpha=0.5, ax=ax)
            ax.set_title(title, fontsize=24)
            fig.colorbar(nd, cax=cax)

        def update(frame):
            for ax in axes.flat:
                ax.clear()
            t_label = (frame + 1) * tps
            _panel(axes[0, 0], cax1, preds[frame].numpy(),
                   f"Time {t_label:.2f}h — Prediction")
            _panel(axes[0, 1], cax2, gts[frame].numpy(),
                   f"Time {t_label:.2f}h — Ground Truth")
            _panel(axes[1, 0], cax3,
                   torch.abs(preds[frame] - gts[frame]).numpy(),
                   f"Time {t_label:.2f}h — Absolute Error")
            times = [(i + 1) * tps for i in range(frame + 1)]
            axes[1, 1].plot(times, rmses[:frame + 1], color="b", linewidth=3,
                            label="Water Depth RMSE")
            axes[1, 1].set_title("RMSE Over Time", fontsize=24)
            axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
            axes[1, 1].set_ylabel("RMSE", fontsize=24)
            axes[1, 1].legend(fontsize=20)
            axes[1, 1].grid(True)

        animation.FuncAnimation(fig, update, frames=len(preds), repeat=False).save(
            path, writer="pillow", fps=2
        )
        plt.close(fig)

    def run(self):
        all_rmses = []

        with torch.no_grad():
            for idx in range(len(self.dataset)):
                g, rollout_data = self.dataset[idx]
                g_snapshot      = g.clone()
                g               = g.to(self.device)

                preds, gts, rmses = self._rollout(g, rollout_data)
                all_rmses.append(rmses)

                sample_id = self.dataset.dynamic_data[idx].get("hydro_id", idx)
                print(f"Hydrograph {sample_id}  |  Mean RMSE = {sum(rmses)/len(rmses):.4f}")

                self._animate(
                    preds, gts, g_snapshot, rmses,
                    os.path.join(self.cfg["animation_output_dir"], f"animation_{sample_id}.gif"),
                )

        self._summary(all_rmses)

    def _summary(self, all_rmses):
        tensor    = torch.tensor(all_rmses)
        mean      = torch.mean(tensor, dim=0)
        std       = torch.std(tensor,  dim=0)
        timesteps = [(i + 1) * self.cfg["time_per_step"] for i in range(self.rollout_len)]

        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, mean.numpy(), label="Mean RMSE", linewidth=3)
        plt.fill_between(
            timesteps,
            (mean - std).numpy(),
            (mean + std).numpy(),
            alpha=0.3, label="± Std",
        )
        plt.xlabel("Time (Hours)", fontsize=20)
        plt.ylabel("RMSE (Water Depth)", fontsize=20)
        plt.title("Overall RMSE Over Rollout", fontsize=24)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cfg["animation_output_dir"], "overall_rmse.png"), dpi=150
        )
        plt.show()
