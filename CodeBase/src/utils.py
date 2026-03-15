from dataclasses import dataclass
from pathlib import Path
import logging
import yaml

@dataclass(frozen = True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    data_dir: str = "./data" 
    log_file: str = "training.log"
    ckpt_path: str = "mnist_mlp_state.pt"
    best_ckpt_path: str = "best_model.pt" 
    metrics_path: str = "metrics.csv"
    patience: int = 3
    log_every: int = 100 
    num_workers: int = 2

def load_config(config_path: str | Path) -> TrainConfig:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return TrainConfig(
        epochs = int(cfg["epochs"]),
        batch_size = int(cfg["batch_size"]),
        learning_rate = float(cfg["learning_rate"]),
        hidden_dim = int(cfg["hidden_dim"]),
        data_dir = str(cfg.get("data_dir", "./data")),
        log_file = str(cfg.get("log_file", "training.log")),
        ckpt_path = str(cfg.get("ckpt_path", "mnist_mlp_state.pt")),
        best_ckpt_path = str(cfg.get("best_ckpt_path", "best_model.pt")),
        log_every = int(cfg.get("log_every", 100)),
        num_workers = int(cfg.get("num_workers", 2)),
        patience = int(cfg.get("patience", 3)),
        metrics_path = str(cfg.get("metrics_path", "metrics.csv")),
    )


#----------
# Logging
#----------

def setup_logger(name: str = "trainer", log_file: str = "training.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid dup handlers (re-runs and imports)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    console = logging.Streamhandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger