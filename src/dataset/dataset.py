import os
from datetime import datetime, timedelta
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CropFusionNetDataset(Dataset):
    """
    PyTorch Dataset for CropFusionNet.

    This dataset handles loading district-level (NUTS_ID) timeseries data,
    masking based on crop phenology (sowing/harvest dates), scaling features,
    and returning tensors ready for model training.

    Attributes:
        config: Configuration object containing file paths and hyperparameters.
        mode (str): Data split mode ('train', 'val', or 'test').
        scale (bool): Whether to normalize features using pre-computed scalers.
        seq_len (int): The fixed sequence length expected by the model.
        parquet_dir (str): Directory containing timeseries Parquet files.
    """

    def __init__(self, config: Any, mode: str = "train", scale: bool = False):
        """
        Initializes the YieldFormerDataset.

        Args:
            config: Configuration object with paths and feature lists.
            mode (str, optional): The subset of data to load. Defaults to "train".
            scale (bool, optional): If True, scales features using config scalers. Defaults to False.
        """
        self.config = config
        self.mode = mode
        self.scale = scale
        self.seq_len = self.config.model_config.get("seq_length")
        self.parquet_dir = self.config.TIMESERIES_PARQUET_DIR
        self.harvest_next_year = self.config.HARVEST_NEXT_YEAR

        # ---------------- Load Split and Yield ----------------
        self.split_table = pd.read_csv(config.SPLIT_FILE_PATH)
        self.split_table = self.split_table[
            self.split_table["split"] == mode
        ].reset_index(drop=True)

        self.yield_table = pd.read_csv(config.YIELD_FILE_PATH)
        self.yield_table.set_index(["NUTS_ID", "year"], inplace=True)

        valid_indices = self.split_table.apply(
            lambda row: (row["NUTS_ID"], row["year"]) in self.yield_table.index, axis=1
        )

        # We drop the rows that have no yield data
        original_len = len(self.split_table)
        self.split_table = self.split_table[valid_indices].reset_index(drop=True)
        new_len = len(self.split_table)

        if original_len != new_len:
            print(
                f"⚠️ Filtered dataset: Dropped {original_len - new_len} samples missing from Yield Table."
            )

        # ---------------- Load Phenology ----------------
        self.phenology_table = pd.read_csv(
            config.PHENOLOGY_FILE_PATH,
            parse_dates=["Sowing_DOY", "Flowering_DOY", "Harvest_DOY"],
        ).drop(columns=["STATE_ID"])
        self.phenology_table.set_index(["STATE_NAME", "Year"], inplace=True)

        # ---------------- Load Static Features ----------------
        self.static_table = pd.read_csv(config.STATIC_FILE_PATH)
        self.static_table.set_index("NUTS_ID", inplace=True)

        # ---------------- Load Scalers ----------------
        scalers = self.config.scalers

        self.real_mean = np.array(scalers["time_varying_mean"])
        self.real_std = np.array(scalers["time_varying_std"])
        self.static_mean = np.array(scalers["static_mean"])
        self.static_std = np.array(scalers["static_std"])

        self.target_mean = scalers.get(f"{self.config.target}_mean")
        self.target_std = scalers.get(f"{self.config.target}_std")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.split_table)

    def _get_phenology_dates(self, nuts_name: str, year: int):
        """
        Returns a dictionary with 'Sowing_DOY' and 'Harvest_DOY'.
        If the specific year is missing, it calculates the median DOY from available years.
        """
        idx = (nuts_name, year)

        if idx in self.phenology_table.index:
            row = self.phenology_table.loc[idx]
            sowing_date = pd.to_datetime(row["Sowing_DOY"])
            harvest_date = pd.to_datetime(row["Harvest_DOY"])

            return (sowing_date, harvest_date)

        else:
            # Impute using median from available years
            phen_dates = self.phenology_table.loc[nuts_name]
            median_sowing_doy = phen_dates["Sowing_DOY"].dt.dayofyear.median()
            median_harvest_doy = phen_dates["Harvest_DOY"].dt.dayofyear.median()

            # Determine Sowing Year
            if self.harvest_next_year:
                sowing_year = year - 1
            else:
                sowing_year = year

            # Construct Timestamps
            try:
                sowing_date = pd.to_datetime(
                    datetime(sowing_year, 1, 1)
                    + timedelta(days=int(median_sowing_doy) - 1)
                )
                harvest_date = pd.to_datetime(
                    datetime(year, 1, 1) + timedelta(days=int(median_harvest_doy) - 1)
                )
                return (sowing_date, harvest_date)

            except ValueError:
                return None

    def _enforce_sequence_length(
        self, inputs: np.ndarray, valid_mask: np.ndarray, variable_mask: np.ndarray
    ):
        """
        Truncates or pads the sequence to match `self.seq_len`.
        """
        time_steps = inputs.shape[0]

        if time_steps > self.seq_len:
            return (
                inputs[: self.seq_len],
                valid_mask[: self.seq_len],
                variable_mask[: self.seq_len],
            )
        elif time_steps < self.seq_len:
            raise ValueError(
                f"Sequence length {time_steps} is shorter than required {self.seq_len}."
            )

        return inputs, valid_mask, variable_mask

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict containing:
                - 'inputs': Tensor of shape (seq_len, num_features)
                - 'identifier': Tensor of static features
                - 'mask': Valid time steps mask
                - 'target': The yield value (standardized if scale=True)
                - Metadata (NUTS_ID, year)
        """
        row = self.split_table.loc[idx]
        nuts_id = row["NUTS_ID"]
        nuts_name = row["NUTS_NAME"]
        year = int(row["year"])

        # ---------------- 1. Extract Target ----------------
        try:
            y = self.yield_table.loc[(nuts_id, year), self.config.target]
        except KeyError:
            raise ValueError(f"No yield data found for {nuts_id} in {year}")

        if self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std

        # ---------------- 2. Extract Phenology ----------------
        sowing_date, harvest_date = self._get_phenology_dates(nuts_name, year)

        if (sowing_date is None) or (harvest_date is None):
            raise ValueError(
                f"Phenology data missing and cannot be imputed for {nuts_name} in {year}"
            )

        # ---------------- 3. Process Time-Series ----------------
        parquet_path = os.path.join(self.parquet_dir, f"{nuts_id}_{year}.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        timeseries_data = pd.read_parquet(parquet_path)
        timeseries_data["date"] = pd.to_datetime(
            timeseries_data["date"], format="%Y-%m-%d"
        )

        # Apply masking based on phenology
        timeseries_data.loc[
            timeseries_data["date"] > harvest_date, self.config.climate_features
        ] = np.nan

        # Remote Sensing: Mask BEFORE sowing AND AFTER harvest
        timeseries_data.loc[
            (timeseries_data["date"] < sowing_date)
            | (timeseries_data["date"] > harvest_date),
            self.config.remote_sensing_features,
        ] = np.nan

        # Extract real-valued features
        time_varying_real = timeseries_data[self.config.time_varying_real].values

        # Generate masks
        # valid_mask: 1 if ANY feature is present (not NaN) for that timestep
        valid_mask = (~np.isnan(time_varying_real)).any(axis=1)
        # variable_mask: 1 where specific features are present
        variable_mask = ~np.isnan(time_varying_real)

        # Scaling
        if self.scale:
            time_varying_real = (time_varying_real - self.real_mean) / self.real_std
            time_varying_real = np.nan_to_num(time_varying_real)

        x_inputs = time_varying_real

        # Enforce Sequence Length (Truncate/Pad)
        if self.seq_len:
            x_inputs, valid_mask, variable_mask = self._enforce_sequence_length(
                x_inputs, valid_mask, variable_mask
            )

        # ---------------- 4. Process Static Features ----------------
        if nuts_id not in self.static_table.index:
            raise ValueError(f"Static features missing for {nuts_id}")

        static_data = self.static_table.loc[nuts_id].values

        if self.scale:
            static_data = (static_data - self.static_mean) / self.static_std

        # Add batch dimension to static features for concatenation later if needed
        x_identifier = np.expand_dims(static_data, axis=0)

        # ---------------- 5. Return Tensors ----------------
        return {
            "NUTS_ID": nuts_id,
            "year": year,
            "inputs": torch.tensor(x_inputs, dtype=torch.float32),
            "identifier": torch.tensor(x_identifier, dtype=torch.float32),
            "mask": torch.tensor(valid_mask, dtype=torch.float32),
            "variable_mask": torch.tensor(variable_mask, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
        }
