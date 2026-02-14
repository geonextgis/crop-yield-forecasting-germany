import os
import time

import config as cfg
import numpy as np
import torch
from config import tft_config, train_config
from dataset import YieldFormerDataset
from loss import MSELoss, QuantileLoss
from model import YieldFormer
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import set_seed

device = train_config["device"]
set_seed(42)


# Datasets and dataloaders
train_dataset = YieldFormerDataset(cfg, mode="train", scale=True, sample_grids=True)
val_dataset = YieldFormerDataset(cfg, mode="val", scale=True, sample_grids=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config["batch_size"],
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_config["batch_size"],
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


# Model, optimizer, and loss
model = YieldFormer(
    tft_config,
    hidden_dim=train_config["hidden_dim"],
    quantiles=train_config["quantiles"],
)
model = nn.DataParallel(model).to(device)
# criterion = QuantileLoss(quantiles=train_config["quantiles"]).to(train_config["device"])
criterion = MSELoss(quantiles=train_config["quantiles"]).to(train_config["device"])
optimizer = Adam(
    model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"]
)
num_epochs = train_config.get("num_epochs", 50)
patience = train_config.get("early_stopping_patience", 10)
batch_size = train_config.get("batch_size", 32)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",  # minimize validation loss
    factor=0.5,  # reduce LR by 50%
    patience=3,  # wait for 3 epochs before reducing
    threshold=1e-4,  # minimal improvement threshold
    min_lr=1e-6,  # lower bound for learning rate
)


# Train function
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    scheduler=None,
    checkpoint_dir="checkpoints",
):
    """
    Performs the entire training, validation, and logging process.
    Optimized for training with a full batch size per forward pass (no gradient accumulation needed).
    """

    # 1. Setup Logging & Checkpoints
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_id = f"run_{timestamp}"

    # TensorBoard logs
    log_dir = os.path.join("runs", log_id)
    writer = SummaryWriter(log_dir=log_dir)

    # Model Checkpoints folder: checkpoints/run_YYYYMMDD-HHMMSS/
    save_folder = os.path.join(checkpoint_dir, log_id)
    os.makedirs(save_folder, exist_ok=True)

    print(f"📘 TensorBoard logs: {log_dir}")
    print(f"💾 Checkpoints will be saved to: {save_folder}")

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # --- 1. Training Phase ---
        model.train()

        # Trackers for separate losses
        train_metrics = {"loss": 0.0}

        # Optimization: Gradients are zeroed once per batch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            optimizer.zero_grad()

            # 1. Prepare Inputs (move entire batch to device)
            batch_inputs = batch["inputs"].to(device)
            batch_identifier = batch["identifier"].to(device)
            batch_masks = batch["mask"].to(device)
            batch_targets = batch["target"].to(device)

            batch_var_mask = batch.get("variable_mask")
            if batch_var_mask is not None:
                batch_var_mask = batch_var_mask.to(device)

            preds = model(
                {
                    "inputs": batch_inputs,
                    "identifier": batch_identifier,
                    "mask": batch_masks,
                    "variable_mask": batch_var_mask,
                }
            )

            # 3. Calculate Loss
            loss = criterion(preds, batch_targets)

            # 4. Backward Pass and Step
            loss.backward()
            optimizer.step()

            train_metrics["loss"] += loss.item()

        # Calculate mean training loss over the entire dataset
        train_loss = train_metrics["loss"] / len(train_loader)

        # --- 2. Validation Phase ---
        model.eval()
        val_metrics = {"loss": 0.0}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch_inputs = batch["inputs"].to(device)
                batch_identifier = batch["identifier"].to(device)
                batch_masks = batch["mask"].to(device)
                batch_targets = batch["target"].to(device)

                batch_var_mask = batch.get("variable_mask")
                if batch_var_mask is not None:
                    batch_var_mask = batch_var_mask.to(device)

                preds = model(
                    {
                        "inputs": batch_inputs,
                        "identifier": batch_identifier,
                        "mask": batch_masks,
                        "variable_mask": batch_var_mask,
                    }
                )

                # Calculate Loss
                loss = criterion(preds, batch_targets)

                # Accumulate
                val_metrics["loss"] += loss.item()

        # Calculate mean validation loss over the entire dataset
        val_loss = val_metrics["loss"] / len(val_loader)

        # --- 3. Logging, Scheduling, and Early Stopping ---
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s | LR: {current_lr:.2e}"
        )

        # TensorBoard
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Scheduler Step
        if scheduler:
            scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()

            save_path = os.path.join(save_folder, "best_model.pt")
            torch.save(best_model_state, save_path)
            print(f"✨ New best val loss: {best_val_loss:.5f}. Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch}")
                break

    writer.close()
    print(f"Training complete. Best Val Loss: {best_val_loss:.5f}")
    return best_model_state


if __name__ == "__main__":
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        patience,
        scheduler,
    )
