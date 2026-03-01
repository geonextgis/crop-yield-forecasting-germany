import argparse
import importlib
import json
import os
import sys
import time
import uuid

source_folder = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src"
sys.path.append(source_folder)

import numpy as np
import torch
from dataset.dataset import CropFusionNetDataset
from loss.loss import QuantileLoss
from models.CropFusionNet.model import CropFusionNet
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import evaluate_and_save_outputs, load_config, set_seed

# Crop
crop = "winter_rapeseed"
cfg, model_config, train_config = load_config(crop)

device = model_config["device"]
set_seed(42)


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")

    # 1. Identity & Logging
    parser.add_argument(
        "--job_id", type=str, default=str(uuid.uuid4())[:8], help="Unique Job ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Where to save results"
    )

    # 2. Hyperparameters to tune (add more as needed)
    parser.add_argument("--lr", type=float, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, help="Batch Size")
    parser.add_argument("--hidden_dim", type=int, help="LSTM Hidden Dimension")
    parser.add_argument("--lstm_layers", type=int, help="LSTM Layers")
    parser.add_argument("--attn_heads", type=int, help="Attention Heads")
    parser.add_argument("--pooling_heads", type=int, help="Pooling Heads")
    parser.add_argument("--embedding_dim", type=int, help="Embedding Dimension")
    parser.add_argument("--dropout", type=float, help="Dropout Rate")
    parser.add_argument("--seq_length", type=int, help="Sequence Length")

    return parser.parse_args()


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
    exp_name="CropFusionNet_experiment",
):
    # 1. Setup Logging
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_id = f"run_{exp_name}_{timestamp}"
    log_dir = os.path.join("runs", log_id)
    writer = SummaryWriter(log_dir=log_dir)

    save_folder = os.path.join(checkpoint_dir, log_id)
    os.makedirs(save_folder, exist_ok=True)

    print(f"📘 TensorBoard logs: {log_dir}")
    print(f"💾 Checkpoints: {save_folder}")

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        train_loss_accum = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            # Move inputs to device
            inputs = {
                "inputs": batch["inputs"].to(device),
                "identifier": batch["identifier"].to(device),
                "mask": batch["mask"].to(device),
                "variable_mask": (
                    batch.get("variable_mask").to(device)
                    if batch.get("variable_mask") is not None
                    else None
                ),
            }
            targets = batch["target"].to(device)

            # Forward Pass
            output_dict = model(inputs)
            preds = output_dict["prediction"]

            # Loss Calculation
            loss = criterion(preds, targets)

            # Backward Pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Optimization Step
            optimizer.step()

            train_loss_accum += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {
                    "inputs": batch["inputs"].to(device),
                    "identifier": batch["identifier"].to(device),
                    "mask": batch["mask"].to(device),
                    "variable_mask": (
                        batch.get("variable_mask").to(device)
                        if batch.get("variable_mask") is not None
                        else None
                    ),
                }
                targets = batch["target"].to(device)

                output_dict = model(inputs)
                preds = output_dict["prediction"]

                loss = criterion(preds, targets)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        # --- LOGGING & SCHEDULING ---
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e} | T: {elapsed:.1f}s"
        )

        writer.add_scalars(
            "Loss", {"Train": avg_train_loss, "Val": avg_val_loss}, epoch
        )
        writer.add_scalar("LR", current_lr, epoch)

        if scheduler:
            scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(save_folder, "best_model.pt"))
            print(f"✨ New best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    writer.close()
    return best_val_loss


if __name__ == "__main__":

    # 1. Parse Arguments
    args = parse_args()

    # 2. UPDATE CONFIGURATION (Must be done FIRST)
    if args.lr:
        train_config["lr"] = args.lr
    if args.batch_size:
        train_config["batch_size"] = args.batch_size
    if args.hidden_dim:
        model_config["lstm_hidden_dimension"] = args.hidden_dim
    if args.lstm_layers:
        model_config["lstm_layers"] = args.lstm_layers
    if args.attn_heads:
        model_config["attn_heads"] = args.attn_heads
    if args.pooling_heads:
        model_config["pooling_heads"] = args.pooling_heads
    if args.embedding_dim:
        model_config["embedding_dim"] = args.embedding_dim
    if args.dropout:
        model_config["dropout"] = args.dropout
    if args.seq_length:
        model_config["seq_length"] = args.seq_length

    # Update experiment name
    train_config["exp_name"] = f"{train_config.get('exp_name', 'exp')}_{args.job_id}"

    print(f"🚀 Starting Job {args.job_id}")
    print(f"   LR: {train_config['lr']}")
    print(f"   Hidden: {model_config['lstm_hidden_dimension']}")
    print(f"   Batch: {train_config['batch_size']}")

    # ---------------------------------------------------------
    # 3. INITIALIZE OBJECTS
    # ---------------------------------------------------------

    # Re-initialize Datasets & Loaders
    train_dataset = CropFusionNetDataset(cfg, mode="train", scale=True)
    val_dataset = CropFusionNetDataset(cfg, mode="val", scale=True)
    test_dataset = CropFusionNetDataset(cfg, mode="test", scale=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Re-initialize Model
    model = CropFusionNet(model_config).to(device)

    # Re-initialize Optimizer (to pick up new lr)
    optimizer = Adam(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config.get("weight_decay", 1e-5),
    )

    # Re-initialize Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6
    )

    criterion = QuantileLoss(quantiles=model_config["quantiles"]).to(device)

    # ---------------------------------------------------------
    # 4. RUN TRAINING
    # ---------------------------------------------------------
    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config.get("num_epochs", 50),
        patience=train_config.get("early_stopping_patience", 10),
        scheduler=scheduler,
        exp_name=train_config["exp_name"],
    )

    # Save the trained model
    model_save_path = os.path.join(args.output_dir, f"model_{args.job_id}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Trained model saved to {model_save_path}")

    # ---------------------------------------------------------
    # 5. EVALUATE AND SAVE OUTPUTS
    # ---------------------------------------------------------
    # print("🔍 Evaluating and saving outputs...")

    # # Evaluate and save outputs for train, validation, and test datasets
    # evaluate_and_save_outputs(
    #     model, train_loader, criterion, device, args.output_dir, "train"
    # )
    # evaluate_and_save_outputs(
    #     model, val_loader, criterion, device, args.output_dir, "validation"
    # )
    # evaluate_and_save_outputs(
    #     model, test_loader, criterion, device, args.output_dir, "test"
    # )

    # ---------------------------------------------------------
    # 6. SAVE TUNING RESULTS
    # ---------------------------------------------------------
    results = {
        "job_id": args.job_id,
        "train_config": train_config,
        "model_config": model_config,
        "metrics": {
            "best_val_loss": best_val_loss,
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"result_{args.job_id}.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {save_path}")
