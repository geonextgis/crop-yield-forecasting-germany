import os
import random
import subprocess
import uuid

# --- CONFIGURATION ---
CROP = "winter_rapeseed"
FORECAST_MONTH = "jul"
NUM_JOBS = 100
job_name_prefix = f"{CROP}_tuning"
output_base_dir = f"/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train/optimization/{CROP}/{FORECAST_MONTH}"
working_dir = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train"

# --- HYPERPARAMETER SEARCH SPACE ---
# Define the ranges you want to explore
param_grid = {
    "lr": [1e-3, 1e-4, 1e-5],
    "batch_size": [32, 64, 128],
    "hidden_dim": [128, 256, 512],
    "lstm_layers": [1, 2, 3],
    "attn_heads": [4, 8, 16],
    "pooling_heads": [4, 8, 16],
    "embedding_dim": [4, 8, 16],
    "dropout": [0.3, 0.4, 0.5],
}


def get_random_params():
    """Randomly samples one value from each list in the grid."""
    return {k: random.choice(v) for k, v in param_grid.items()}


# --- SUBMISSION LOOP ---
if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists(output_base_dir):
    print(f"📁 Creating output directory: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)

for i in range(NUM_JOBS):
    # 1. Sample Parameters
    params = get_random_params()

    # 2. Generate Unique ID
    unique_id = str(uuid.uuid4())[:8]  # e.g., 'a1b2c3d4'

    # 3. Construct Python Command
    python_cmd = (
        f"python train_CropFusionNet.py "
        f"--job_id {unique_id} "
        f"--output_dir {output_base_dir} "
        f"--lr {params['lr']} "
        f"--batch_size {params['batch_size']} "
        f"--hidden_dim {params['hidden_dim']} "
        f"--lstm_layers {params['lstm_layers']} "
        f"--attn_heads {params['attn_heads']} "
        f"--pooling_heads {params['pooling_heads']} "
        f"--embedding_dim {params['embedding_dim']} "
        f"--dropout {params['dropout']}"
    )

    # 4. Create SLURM Script Content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name_prefix}_{unique_id}
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu

# Load modules or activate environment
source ~/.bashrc
conda activate torch

# Move to the correct directory
cd {working_dir}

echo "Starting job {unique_id}..."
echo "Parameters: {params}"
{python_cmd}
"""

    # 5. Write and Submit Job
    script_filename = f"job_{unique_id}.sh"
    with open(script_filename, "w") as f:
        f.write(slurm_script)

    print(f"Submitting Job {i+1}/{NUM_JOBS} (ID: {unique_id})")
    print(f"  Config: {params}")

    # Execute sbatch
    subprocess.run(["sbatch", script_filename])

    # Cleanup: Remove the .sh file after submission to keep folder clean
    os.remove(script_filename)

print("✅ All jobs submitted!")
