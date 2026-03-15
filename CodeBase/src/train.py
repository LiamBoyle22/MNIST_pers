import csv 
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import make_loaders
from src.model import MLP
from src.utils import load_config, setup_logger

    
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

    train_loader, val_loader = make_loaders(
        batch_size = cfg.batch_size,
        data_dir = cfg.data_dir,
        num_workers = cfg.num_workers
    )

    model = MLP(cfg.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_acc = float("-inf")    #Initialize best validation accuracy to negative infinity to ensure any improvement is captured
    epochs_wout_improve = 0     #Counter for epochs without improvement in validation accuracy

    logger.info("starting training...")

    with open(cfg.metrics_path, "w", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

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

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                    },
                    cfg.best_ckpt_path
                )  
                
                logger.info(f"New best val acc: {best_val_acc:.4f} | "
                            f"Saved checkpoint -> {cfg.best_ckpt_path}")
                
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
        model.state_dict(),
        cfg.ckpt_path
    )

    logger.info(f"Saved model -> {cfg.best_ckpt_path}")

    best_checkpoint = torch.load(cfg.best_ckpt_path, map_location=device)
    
    model2 = MLP(cfg.hidden_dim).to(device)
    model2.load_state_dict(best_checkpoint["model_state_dict"])
    model2.eval()

    logger.info(
        f"Loaded best model <- {cfg.best_ckpt_path}"
        f"(epoch = {best_checkpoint['epoch']}, val_acc = {best_checkpoint['val_acc']:.4f})"
    )