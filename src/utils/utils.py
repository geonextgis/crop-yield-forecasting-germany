import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_start_end_doy(data_path):
    """
    Get the start DOY, end DOY, and full DOY window for sowing–harvest period.

    Parameters
    ----------
    data_path : str
        Path to a CSV file with columns 'Sowing_DOY', 'Flowering_DOY', 'Harvest_DOY'.

    Returns
    -------
    dict
        {
            'sow_month_start_doy': int,   # DOY of the first sowing month (day 1 of month)
            'harvest_month_end_doy': int, # DOY of the last harvest month (day 31 of month)
            'window_length_in_days': int, # Total days in the sowing–harvest window
            'window_doys': list[int]      # List of DOYs in the window
        }
    """
    REF_YEAR = 1970  # Non-leap reference year for DOY calculation

    # Read phenology data
    phenology_df = pd.read_csv(data_path)

    # Rename columns for clarity
    phenology_df.rename(
        columns={
            "Sowing_DOY": "Sowing_DATE",
            "Flowering_DOY": "Flowering_DATE",
            "Harvest_DOY": "Harvest_DATE",
        },
        inplace=True,
    )

    # Get earliest sowing month and latest harvest month
    sow_month = pd.to_datetime(phenology_df["Sowing_DATE"]).dt.month.min()
    harvest_month = pd.to_datetime(phenology_df["Harvest_DATE"]).dt.month.max()

    # Convert months to DOY
    sow_doy = datetime(REF_YEAR, sow_month, 1).timetuple().tm_yday
    harvest_doy = datetime(REF_YEAR, harvest_month, 31).timetuple().tm_yday

    # Build DOY window, handling year wrap-around
    if harvest_doy < sow_doy:
        window_doys = list(range(sow_doy, 366)) + list(range(1, harvest_doy + 1))
    else:
        window_doys = list(range(sow_doy, harvest_doy + 1))

    return {
        "sow_month_start_doy": sow_doy,
        "harvest_month_end_doy": harvest_doy,
        "window_length_in_days": len(window_doys),
        "window_doys": window_doys,
    }


# Function to save model outputs
def save_outputs(output_dict, output_dir, dataset_type):
    """
    Save model outputs (entire output_dict and targets) to a pickle file.

    Args:
        output_dict (dict): Output dictionary from the model.
        output_dir (str): Directory to save the outputs.
        dataset_type (str): Type of dataset (train, validation, test).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_type}_outputs.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)
    print(f"✅ {dataset_type.capitalize()} outputs saved to {output_file}")


# Function to evaluate the model and save outputs
def evaluate_and_save_outputs(
    model,
    data_loader,
    criterion,
    device,
    output_dir,
    dataset_type,
):
    """
    Evaluate the model and save:
        - predictions
        - targets
        - metadata (NUTS_ID, year)
        - any additional weights returned by model

    Saved as pickle (.pkl) file.
    """

    model.eval()
    total_loss = 0.0

    all_outputs = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_type}"):

            # ---------------------
            # Prepare inputs
            # ---------------------
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

            batch_targets = batch["target"].to(device)

            # ---------------------
            # Forward pass
            # ---------------------
            output_dict = model(inputs)
            batch_preds = output_dict["prediction"]

            # ---------------------
            # Loss
            # ---------------------
            loss = criterion(batch_preds, batch_targets)
            total_loss += loss.item()

            # ---------------------
            # Store model outputs
            # ---------------------
            for key, value in output_dict.items():

                if key not in all_outputs:
                    all_outputs[key] = []

                if torch.is_tensor(value):
                    all_outputs[key].append(value.detach().cpu())
                else:
                    all_outputs[key].append(value)

            # ---------------------
            # Store targets
            # ---------------------
            all_outputs.setdefault("target", []).append(batch_targets.detach().cpu())

            # ---------------------
            # Store metadata
            # ---------------------
            all_outputs.setdefault("NUTS_ID", []).extend(batch["NUTS_ID"])
            all_outputs.setdefault("year", []).extend(batch["year"])

    # -------------------------------------------------
    # Concatenate tensors
    # -------------------------------------------------
    for key, value in all_outputs.items():

        if not isinstance(value, list) or len(value) == 0:
            continue

        if torch.is_tensor(value[0]):

            if value[0].dim() == 0:
                tensor_out = torch.stack(value)
            else:
                tensor_out = torch.cat(value, dim=0)

            # Convert to NumPy for safer pickle storage
            all_outputs[key] = tensor_out.numpy()

    # -------------------------------------------------
    # Save as pickle
    # -------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_type}_outputs.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(all_outputs, f)

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------
    avg_loss = total_loss / len(data_loader)

    print(f"{dataset_type.capitalize()} Loss: {avg_loss:.4f}")
    print(f"Outputs saved to: {save_path}")

    return avg_loss


def save_config(train_cfg, model_cfg, output_dir):

    config = {
        "train_cfg": train_cfg,
        "model_cfg": model_cfg,
    }

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "config.json")

    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to: {save_path}")
