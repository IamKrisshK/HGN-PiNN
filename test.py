import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch_geometric.utils import to_networkx
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN


class Inferencer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
        self.rollout_length = cfg["num_test_time_steps"]
        self.n_time_steps = cfg["n_time_steps"]
        os.makedirs(cfg["animation_output_dir"], exist_ok=True)
        self.model = self._load_model()
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        return HydroGraphDataset(
            data_dir=self.cfg["test_dir"],
            prefix=self.cfg["prefix"],
            n_time_steps=self.cfg["n_time_steps"],
            hydrograph_ids_file=self.cfg["test_ids_file"],
            split="test",
            rollout_length=self.rollout_length,
            return_physics=False,
        )

    def _load_model(self):
        model = MeshGraphKAN(
            self.cfg["num_input_features"],
            self.cfg["num_edge_features"],
            self.cfg["num_output_features"],
        ).to(self.device)
        load_checkpoint(
            self.cfg["ckpt_path"],
            models=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=self.device,
        )
        model.eval()
        return model

    def _rollout_sample(self, g, rollout_data):
        edge_features = g.edge_attr.to(self.device)
        X_iter = g.x.to(self.device)
        num_nodes = X_iter.size(0)

        inflow_seq = rollout_data["inflow"].to(self.device)
        precip_seq = rollout_data["precipitation"].to(self.device)
        wd_gt_seq = rollout_data["water_depth_gt"].to(self.device)
        rollout_preds, ground_truth_list, rmse_list = [], [], []

        for t in range(self.rollout_length):
            static_part = X_iter[:, :12]
            water_depth_window = X_iter[:, 12: 12 + self.n_time_steps]
            volume_window = X_iter[:, 12 + self.n_time_steps: 12 + 2 * self.n_time_steps]
            X_input = torch.cat([static_part, water_depth_window, volume_window], dim=1)
            pred = self.model(X_input, edge_features, g)
            new_wd = water_depth_window[:, -1:] + pred[:, 0:1]
            new_vol = volume_window[:, -1:] + pred[:, 1:2]
            water_depth_updated = torch.cat([water_depth_window[:, 1:], new_wd], dim=1)
            volume_updated = torch.cat([volume_window[:, 1:], new_vol], dim=1)
            static_upd = static_part.clone()
            static_upd[:, 10:12] = torch.cat([
                inflow_seq[t].unsqueeze(0).expand(num_nodes, 1),
                precip_seq[t].unsqueeze(0).expand(num_nodes, 1),
            ], dim=1)
            X_iter = torch.cat([static_upd, water_depth_updated, volume_updated], dim=1)
            rollout_preds.append(new_wd.squeeze(1).cpu())
            ground_truth_list.append(wd_gt_seq[t].cpu())
            rmse_list.append(
                torch.sqrt(torch.mean((new_wd.squeeze(1) - wd_gt_seq[t].cpu()) ** 2)).item()
            )

        return rollout_preds, ground_truth_list, rmse_list

    def _create_animation(self, rollout_preds, ground_truth_list, g, rmse_list, output_path):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        fig, axes = plt.subplots(2, 2, figsize=(30, 30))
        cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
        cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
        cax3 = fig.add_axes([0.05, 0.10, 0.02, 0.35])
        pos = {
            i: (g.x[i, 0].item(), g.x[i, 1].item())
            for i in range(g.x.shape[0])
        }
        all_vals = torch.cat(rollout_preds + ground_truth_list)
        vmin, vmax = all_vals.min().item(), all_vals.max().item()
        tps = self.cfg["time_per_step"]
        def _draw_graph(ax, node_color, title, cax):
            graph = to_networkx(g).to_undirected()
            nodes = nx.draw_networkx_nodes(
                graph, pos, node_color=node_color, node_size=250,
                cmap=plt.cm.viridis, ax=ax, vmin=vmin, vmax=vmax, node_shape="s",
            )
            nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax)
            ax.set_title(title, fontsize=24)
            fig.colorbar(nodes, cax=cax)
        def update(frame):
            for ax in axes.flat:
                ax.clear()
            t_label = (frame + 1) * tps
            pred_vals = rollout_preds[frame].numpy()
            gt_vals = ground_truth_list[frame].numpy()
            err_vals = torch.abs(rollout_preds[frame] - ground_truth_list[frame]).numpy()
            _draw_graph(axes[0, 0], pred_vals, f"Time {t_label:.2f}h — Prediction", cax1)
            _draw_graph(axes[0, 1], gt_vals, f"Time {t_label:.2f}h — Ground Truth", cax2)
            _draw_graph(axes[1, 0], err_vals, f"Time {t_label:.2f}h — Absolute Error", cax3)
            times = [(i + 1) * tps for i in range(frame + 1)]
            axes[1, 1].plot(times, rmse_list[:frame + 1], color="b", linewidth=3, label="Water Depth RMSE")
            axes[1, 1].set_title("RMSE Over Time", fontsize=24)
            axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
            axes[1, 1].set_ylabel("RMSE", fontsize=24)
            axes[1, 1].legend(fontsize=20)
            axes[1, 1].grid(True)
        ani = animation.FuncAnimation(fig, update, frames=len(rollout_preds), repeat=False)
        ani.save(output_path, writer="pillow", fps=2)
        plt.close(fig)

    def _plot_overall_rmse(self, all_rmse_all):
        all_rmse_tensor = torch.tensor(all_rmse_all)
        overall_mean = torch.mean(all_rmse_tensor, dim=0)
        overall_std = torch.std(all_rmse_tensor, dim=0)
        timesteps = [(i + 1) * self.cfg["time_per_step"] for i in range(self.rollout_length)]
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, overall_mean.numpy(), label="Mean RMSE", linewidth=3)
        plt.fill_between(
            timesteps,
            (overall_mean - overall_std).numpy(),
            (overall_mean + overall_std).numpy(),
            alpha=0.3, label="± Std",
        )
        plt.xlabel("Time (Hours)", fontsize=20)
        plt.ylabel("RMSE (Water Depth)", fontsize=20)
        plt.title("Overall RMSE Over Rollout", fontsize=24)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["animation_output_dir"], "overall_rmse.png"), dpi=150)
        plt.show()
    def run(self):
        all_rmse_all = []
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                g, rollout_data = self.dataset[idx]
                g = g.to(self.device)
                rollout_preds, ground_truth_list, rmse_list = self._rollout_sample(g, rollout_data)
                all_rmse_all.append(rmse_list)
                sample_id = self.dataset.dynamic_data[idx].get("hydro_id", idx)
                anim_path = os.path.join(self.cfg["animation_output_dir"], f"animation_{sample_id}.gif")
                self._create_animation(rollout_preds, ground_truth_list, g.cpu(), rmse_list, anim_path)
        self._plot_overall_rmse(all_rmse_all)
