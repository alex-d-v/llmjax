"""W&B and TensorBoard logging utilities."""

import os
from pathlib import Path


class WandbLogger:
    """Weights & Biases logger."""

    def __init__(self, project: str = "llm-jax", config: dict = None):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, config=config)
            self.enabled = True
            print(f"W&B logging enabled: {self.run.url}")
        except ImportError:
            print("wandb not installed. Skipping W&B logging.")
            self.enabled = False

    def log(self, metrics: dict, step: int):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.wandb.finish()


class TensorBoardLogger:
    """TensorBoard logger using JAX's built-in summary writer."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
            print(f"TensorBoard logging to: {log_dir}")
        except ImportError:
            print("tensorboard not installed. Skipping TB logging.")
            self.enabled = False

    def log(self, metrics: dict, step: int):
        if self.enabled:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
            self.writer.flush()

    def finish(self):
        if self.enabled:
            self.writer.close()


class Logger:
    """Combined logger that dispatches to W&B and/or TensorBoard."""

    def __init__(self, cfg: dict):
        self.loggers = []
        log_cfg = cfg.get("logging", {})

        if log_cfg.get("wandb", False):
            self.loggers.append(WandbLogger(
                project=log_cfg.get("wandb_project", "llm-jax"),
                config=cfg,
            ))

        if log_cfg.get("tensorboard", False):
            self.loggers.append(TensorBoardLogger(
                log_dir=log_cfg.get("tb_log_dir", "runs"),
            ))

        if not self.loggers:
            print("No external loggers enabled. Logging to stdout only.")

    def log(self, metrics: dict, step: int):
        for logger in self.loggers:
            logger.log(metrics, step)

    def finish(self):
        for logger in self.loggers:
            logger.finish()