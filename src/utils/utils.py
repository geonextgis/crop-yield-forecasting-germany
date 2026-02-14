import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import torch

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
            'Sowing_DOY': 'Sowing_DATE',
            'Flowering_DOY': 'Flowering_DATE',
            'Harvest_DOY': 'Harvest_DATE'
        },
        inplace=True
    )
    
    # Get earliest sowing month and latest harvest month
    sow_month = pd.to_datetime(phenology_df['Sowing_DATE']).dt.month.min()
    harvest_month = pd.to_datetime(phenology_df['Harvest_DATE']).dt.month.max()

    # Convert months to DOY
    sow_doy = datetime(REF_YEAR, sow_month, 1).timetuple().tm_yday
    harvest_doy = datetime(REF_YEAR, harvest_month, 31).timetuple().tm_yday
    
    # Build DOY window, handling year wrap-around
    if harvest_doy < sow_doy:
        window_doys = list(range(sow_doy, 366)) + list(range(1, harvest_doy + 1))
    else:
        window_doys = list(range(sow_doy, harvest_doy + 1))
    
    return {
        'sow_month_start_doy': sow_doy,
        'harvest_month_end_doy': harvest_doy,
        'window_length_in_days': len(window_doys),
        'window_doys': window_doys
    }


def compute_dataset_scaler(
    dataset,
    time_varying_real,
    static_real_variables,
    precision=6,
    n_jobs=8,
):
    """
    Compute mean and std for:
      - Time-varying real variables (scaled efficiently across all timesteps)
      - Static real variables (computed directly)
      
    Categorical variables and indices (e.g., day, month) are ignored.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or list
        Dataset returning dicts with keys: ['inputs', 'identifier']
    time_varying_real : list of str
        Names of real-valued time-varying features.
    static_real_variables : list of str
        Names of real-valued static features.
    precision : int, default=6
        Number of decimal places to keep.
    n_jobs : int, optional
        Number of parallel workers (default=8).

    Returns
    -------
    dict
        Mean and std for time-varying and static real features.
    """

    # get one sample to infer shapes
    sample = None
    for s in dataset:
        if s is not None:
            sample = s
            break
    if sample is None:
        raise ValueError("Dataset is empty or all samples are None.")

    x_inputs, x_identifier = sample['inputs'], sample['identifier']
    n_time_features = x_inputs.shape[1]
    n_static_features = x_identifier.shape[-1]

    # Initialize accumulators
    sum_time = np.zeros(n_time_features, dtype=np.float64)
    sumsq_time = np.zeros(n_time_features, dtype=np.float64)
    count_time = np.zeros(n_time_features, dtype=np.int64)

    sum_static = np.zeros(n_static_features, dtype=np.float64)
    sumsq_static = np.zeros(n_static_features, dtype=np.float64)
    count_static = np.zeros(n_static_features, dtype=np.int64)

    # --- Parallel worker for time-varying features only ---
    def process_one(i):
        sample = dataset[i]
        if sample is None:
            return None

        x_inputs = np.asarray(sample['inputs'], dtype=np.float64)
        x_identifier = np.asarray(sample['identifier'], dtype=np.float64).reshape(-1)

        # time-varying real variables only
        mask_time = ~np.isnan(x_inputs)
        valid_count_time = mask_time.sum(axis=0)
        valid_sum_time = np.nansum(x_inputs, axis=0)
        valid_sumsq_time = np.nansum(x_inputs ** 2, axis=0)

        # static real variables only
        mask_static = ~np.isnan(x_identifier)
        valid_count_static = mask_static.astype(int)
        valid_sum_static = np.nan_to_num(x_identifier)
        valid_sumsq_static = np.nan_to_num(x_identifier ** 2)

        return (
            valid_sum_time, valid_sumsq_time, valid_count_time,
            valid_sum_static, valid_sumsq_static, valid_count_static
        )

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
        delayed(process_one)(i) for i in tqdm(range(len(dataset)))
    )

    for r in results:
        if r is None:
            continue
        s_t, ss_t, c_t, s_s, ss_s, c_s = r
        sum_time += s_t
        sumsq_time += ss_t
        count_time += c_t
        sum_static += s_s
        sumsq_static += ss_s
        count_static += c_s

    # Compute mean/std
    mean_time = sum_time / np.maximum(count_time, 1)
    var_time = (sumsq_time / np.maximum(count_time, 1)) - mean_time ** 2
    std_time = np.sqrt(np.maximum(var_time, 1e-8))

    mean_static = sum_static / np.maximum(count_static, 1)
    var_static = (sumsq_static / np.maximum(count_static, 1)) - mean_static ** 2
    std_static = np.sqrt(np.maximum(var_static, 1e-8))

    print(f"✅ Computed scalers for {len(time_varying_real)} time-varying real and {len(static_real_variables)} static real features")

    result = {
        "time_varying_mean": mean_time[:len(time_varying_real)],
        "time_varying_std": std_time[:len(time_varying_real)],
        "static_mean": mean_static[-len(static_real_variables):],
        "static_std": std_static[-len(static_real_variables):]
    }

    # Round all arrays to desired precision
    result = {
        k: np.round(v, decimals=precision).tolist() for k, v in result.items()
    }

    return result


def calculate_spatial_variance(pixel_preds):
    """
    Calculates the average Standard Deviation of pixel predictions within districts.
    
    pixel_preds: (B, num_grids, Q) - Raw predictions
    """
    # 1. Select the Median Quantile
    # If Q=3 (e.g., 0.1, 0.5, 0.9), we want index 1 (0.5 median)
    # If Q=1, we use index 0
    q_idx = pixel_preds.shape[-1] // 2 
    
    # Shape becomes: (B, num_grids)
    median_preds = pixel_preds[..., q_idx] 

    batch_stds = []
    
    # 2. Loop over the batch
    # (We loop because each district has a different number of valid pixels)
    for i in range(median_preds.shape[0]):
        # Calculate Standard Deviation
        # We need at least 2 pixels to calculate variance. 
        # If a district has 1 pixel (or 0), std dev is 0.
        if median_preds.numel() > 1:
            std = torch.std(median_preds)
            batch_stds.append(std)
        else:
            # No variance possible for single pixel
            batch_stds.append(torch.tensor(0.0, device=pixel_preds.device))
            
    # 4. Average the std dev across all districts in the batch
    return torch.stack(batch_stds).mean().item()