import torch
import torch.nn.functional as F
try:
    from torch_scatter import scatter_sum as _scatter_sum
    _HAS_SCATTER = True
except ImportError:
    _HAS_SCATTER = False

def _scatter_sum_fallback(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Pure-PyTorch scatter sum (slower than torch_scatter but always available)."""
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    out.scatter_add_(0, index, src)
    return out


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if _HAS_SCATTER:
        return _scatter_sum(src, index, dim=0, dim_size=dim_size)
    return _scatter_sum_fallback(src, index, dim_size)


def compute_physics_loss(
    pred: torch.Tensor,
    physics_data: dict,
    graph,
    delta_t: float = 1200.0,
) -> torch.Tensor:
    """
    Vectorised physics-based continuity loss in the de-normalised domain.

    All per-sample operations are executed as batched tensor ops — no Python
    loop over graph samples. This is significantly faster when batch_size > 4.

    Continuity constraint (per sample s):
        pred_total_vol_s = past_vol_s + volume_std_s * Σ pred_vol_diff_{i∈s}

        term1 = ReLU( (pred_total_vol_s - (past_vol_s + Δt·(avg_inflow_s + avg_precip_s·inf_area_s))) / area_s )²
        term2 = ReLU( (fut_vol_s - pred_total_vol_s - Δt·(next_inflow_s + next_precip_s·inf_area_s)) / area_s )²

    physics_loss = mean(term1 + term2)

    Args:
        pred          : (N_total_nodes, ≥2) — column 1 is normalised volume diff
        physics_data  : dict of (batch_size,) tensors from the dataloader
        graph         : PyG Batch — provides graph.batch (node→sample index)
        delta_t       : time step in seconds

    Returns:
        Scalar tensor — mean physics loss over all samples in the batch.
    """
    batch_index = graph.batch                             # (N,) int, node → sample idx
    num_samples = int(batch_index.max().item()) + 1

    pred_vol_diff = pred[:, 1]                            # (N,)
    pred_diff_sum = scatter_sum(pred_vol_diff, batch_index, dim_size=num_samples)
    past_vol_norm  = physics_data["past_volume"]           # (B,)
    fut_vol_norm   = physics_data["future_volume"]         # (B,)
    avg_inflow     = physics_data["avg_inflow"]            # (B,)
    avg_precip     = physics_data["avg_precipitation"]     # (B,)
    next_inflow    = physics_data["next_inflow"]           # (B,)
    next_precip    = physics_data["next_precip"]           # (B,)
    volume_mean    = physics_data["volume_mean"]           # (B,)
    volume_std     = physics_data["volume_std"]            # (B,)
    num_nodes_t    = physics_data["num_nodes"].float()     # (B,)
    area_sum       = physics_data["area_sum"]              # (B,)
    inf_area_sum   = physics_data["infiltration_area_sum"] # (B,)
    past_vol_denorm = past_vol_norm * volume_std + num_nodes_t * volume_mean
    fut_vol_denorm  = fut_vol_norm  * volume_std + num_nodes_t * volume_mean
    pred_total_vol  = past_vol_denorm + volume_std * pred_diff_sum
    precip_term      = avg_precip  * inf_area_sum
    next_precip_term = next_precip * inf_area_sum
    residual1 = (pred_total_vol - (past_vol_denorm + delta_t * (avg_inflow + precip_term))) / area_sum
    term1     = F.relu(residual1) ** 2
    residual2 = (fut_vol_denorm - pred_total_vol - delta_t * (next_inflow + next_precip_term)) / area_sum
    term2     = F.relu(residual2) ** 2

    return (term1 + term2).mean()

def custom_loss(pred: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Decomposed MSE loss for water depth and volume predictions.

    Args:
        pred    : (N, 2)  — column 0 = depth prediction, column 1 = volume diff
        targets : (N, 2)  — ground truth

    Returns:
        dict with keys: total_loss, loss_depth, loss_volume
        All values are detached scalars suitable for logging.
    """
    loss_depth  = F.mse_loss(pred[:, 0], targets[:, 0])
    loss_volume = F.mse_loss(pred[:, 1], targets[:, 1])
    total       = loss_depth + loss_volume

    return {
        "total_loss":  total,
        "loss_depth":  loss_depth.detach(),
        "loss_volume": loss_volume.detach(),
    }
