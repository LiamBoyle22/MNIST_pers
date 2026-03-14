from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import yaml
import csv

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
    patience: int = 3
    log_every: int = 100 
    num_workers: int = 2
    metrics_path: str = "metrics.csv"

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
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)
    
@torch.no_grad()
def batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item() #Returns the average accuracy for a batch of predictions and true labels

#---------
# Train/ Validate
#---------

    #----------
    # Validation set evaluation (no grad, no dropout)
    #----------

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

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)    #Returns average loss and accuracy for validation set

    #---------
    # Train for one epoch (grad, dropout, logging)
    #---------

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

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)    #Returns average loss and accuracy for singular epoch

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

    best_val_acc = float("-inf")    #Initialize best validation accuracy to negative infinity to ensure any improvement is captured
    epochs_wout_improve = 0     #Counter for epochs without improvement in validation accuracy

    logger.info("starting training...")

    with open(cfg.metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        logger.info(f"Saved metrics -> {cfg.metrics_path}")

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

        writer.writerow([epoch, tr_loss, tr_acc, val_loss, val_acc])    #Log training and validation metrics to CSV for later analysis
        metrics_file.flush()    #Ensure metrics are written to disk after each epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_wout_improve = 0
            logger.info(f"New best val acc: {best_val_acc:.4f}")
        else:
            epochs_wout_improve += 1
            logger.info(
                f"No improvement in val acc for {epochs_wout_improve} epoch(s)"
                f" (patience={cfg.patience})"
            )
        
        if epochs_wout_improve >= cfg.patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break   #Early stopping if validation accuracy does not improve for 'patience' epochs
            
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": cfg,
        },
        cfg.ckpt_path
    )
    
    logger.info(f"Saved model -> {cfg.ckpt_path}")

    model2 = MLP(cfg.hidden_dim).to(device)
    model2.load_state_dict(torch.load(cfg.ckpt_path)["model_state_dict"])
    model2.eval()
    logger.info(f"Loaded model <- {cfg.ckpt_path}")     #Checkpoint is saved and loaded successfully, ensuring reusability