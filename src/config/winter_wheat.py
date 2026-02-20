import json
import os

import torch

CROP = "winter_wheat"
HARVEST_NEXT_YEAR = True

ROOT_DATA_DIR = os.path.join(
    f"/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/data/processed/{CROP}"
)
SPLIT_FILE_PATH = os.path.join(ROOT_DATA_DIR, "train_test_val_split.csv")
PHENOLOGY_FILE_PATH = os.path.join(ROOT_DATA_DIR, f"{CROP}_phenology.csv")
YIELD_FILE_PATH = os.path.join(ROOT_DATA_DIR, f"{CROP}_yield.csv")
TIMESERIES_PARQUET_DIR = os.path.join(ROOT_DATA_DIR, "timeseries_parquet_7D")
STATIC_FILE_PATH = os.path.join(ROOT_DATA_DIR, f"{CROP}_static.csv")
SCALER_FILE_PATH = f"/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/scaler/scaler_{CROP}.json"

# Define features
soil_features = ["soil_quality_mean", "soil_quality_stdDev"]
topo_features = ["elevation_mean", "elevation_stdDev", "slope_mean", "slope_stdDev"]
irrigation_features = ["irrigated_fraction"]

remote_sensing_features = ["ndvi", "evi", "fpar", "lai"]
climate_features = [
    "sun_dur",
    "soil_moist",
    "soil_temp",
    "et0",
    "vpd",
    "cwb",
    "tmin",
    "tmax",
    "tavg",
    "prec",
    "rad",
]

# Define variable names
time_varying_real = remote_sensing_features + climate_features
time_varying_categorical = []
static_real_variables = soil_features + topo_features + irrigation_features
static_categorical_variables = []
target = "yield"

# Scaling parameters for standardization
with open(SCALER_FILE_PATH, "r") as f:
    scalers = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define forecasting scenarios
forecast_scenarios = {
    "Feb": 39,
    "Mar": 44,
    "Apr": 48,
    "May": 53,
    "Jun": 57,
    "Jul": 61,
    "Aug": 66,
}
forecast_month = "Jul"

# Config for TFT
model_config = {
    "device": device,
    "static_categorical_variables": len(static_categorical_variables),
    "static_real_variables": len(static_real_variables),
    "static_embedding_vocab_sizes": [],
    "time_varying_categorical_variables": len(time_varying_categorical),
    "time_varying_embedding_vocab_sizes": [],
    "time_varying_real_variables": len(time_varying_real),
    "lstm_hidden_dimension": 256,
    "lstm_layers": 2,
    "attn_heads": 4,
    "pooling_heads": 8,
    "dropout": 0.3,
    "embedding_dim": 16,
    "seq_length": forecast_scenarios[forecast_month],
    "quantiles": [0.1, 0.5, 0.9],
}

train_config = {
    "device": device,
    "batch_size": 32,
    "lr": 1e-5,
    "weight_decay": 1e-5,
    "num_epochs": 500,
    "early_stopping_patience": 10,
    "exp_name": f"exp_{CROP}_{forecast_month}",
}
