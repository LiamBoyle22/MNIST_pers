from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#----------
# config
#----------

@dataclass(frozen = True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    data_dir: str = "./data" 
    log_file: str = "training.log"
    ckpt_path: str = "mnist_mlp_state.pt"
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
        log_every = int(cfg.get("log_every", 100)),
        num_workers = int(cfg.get("num_workers", 2)),
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

#----------
# Data
#----------

def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root = cfg.data_dir, train = True, download = True, transform = transform)
    val_ds = datasets.MNIST(root = cfg.data_dir, train = False, download = True, transform = transform)

    train_loader = DataLoader(
        train_ds,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_workers = cfg.num_workers,
        pin_memory = True,
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = cfg.num_workers,
        pin_memory = True,
        )
    
    return train_loader, val_loader

#----------
# Model
#----------

class MLP(nn.Module):
    def __init__ (self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) # Flatten
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)
    
@torch.no_grad()
def batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

#---------
# Train/ Validate
#---------

@torch.no_grad()
def Validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += batch_accuracy(logits, y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    log_every: int,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none= True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += batch_accuracy(logits, y)
        n_batches += 1

        if step % log_every ==0:
            logger.info(
                f"Step {step} | train_loss={total_loss/n_batches:.4f} | train_acc={total_acc/n_batches:.4f}"
            )

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

#----------
# Orchestration
#----------

def run_training(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    logger = setup_logger(log_file=cfg.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    logger.info(
        f"epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.learning_rate} hidden_dim={cfg.hidden_dim}"
    )

    train_loader, val_loader = make_loaders(cfg)

    model = MLP(cfg.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    logger.info("starting training...")

    for epoch in range (1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model = model, 
            loader = train_loader, 
            criterion = criterion, 
            optimizer = optimizer, 
            device = device, 
            logger = logger, 
            log_every = cfg.log_every
        )
        
        val_loss, val_acc = Validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch} | "
            f"train_loss = {tr_loss:.4f} train_acc{tr_acc:.4f} | "
            f"val_loss = {val_loss:.4f} val_acc = {val_acc:.4f}"
        )

    torch.save(model.state_dict(), cfg.ckpt_path)
    logger.info(f"Saved model -> {cfg.ckpt_path}")