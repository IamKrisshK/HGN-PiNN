import sys
import time
import hydra
from omegaconf import DictConfig
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging.console import PythonLogger
from physicsnemo.launch.logging import RankZeroLoggingWrapper
from trainer import trainer
from mopper import Mopper

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    rank_zero_logger.info("Starting training...")
    wrapper = Mopper(cfg, dist, rank_zero_logger)
    model, dataloader, start_epoch = wrapper.build_all()

    Trainer = trainer(
        cfg,
        model,
        dataloader,
        dist,
        rank_zero_logger,
        wrapper,
    )

    for epoch in range(start_epoch, cfg.epochs):
        total_loss = 0.0
        steps = 0
        start_time = time.time()
        for batch in dataloader:
            loss, loss_dict = Trainer.step(batch)
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        if dist.rank == 0:
            rank_zero_logger.info(
                f"Epoch {epoch} | Avg Loss: {avg_loss:.4e}"
            )
        Trainer.scheduler.step()
        if dist.rank == 0 and cfg.get("ckpt_path", None):
            from physicsnemo.launch.utils import save_checkpoint
            from hydra.utils import to_absolute_path
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=Trainer.model,
                optimizer=Trainer.optimizer,
                scheduler=Trainer.scheduler,
                scaler=Trainer.scaler,
                epoch=epoch,
            )
        if dist.world_size > 1:
            import torch.distributed as dist_torch
            dist_torch.barrier()
        if dist.rank == 0:
            elapsed = time.time() - start_time
            rank_zero_logger.info(f"Epoch time: {elapsed:.2f}s")
    rank_zero_logger.info("Training complete.")

if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    main()
