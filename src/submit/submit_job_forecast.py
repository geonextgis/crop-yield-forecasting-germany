import importlib
import os
import subprocess
import sys
import uuid

# --- CONFIGURATION ---
CROP = "winter_wheat"

source_folder = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src"
sys.path.append(source_folder)


def load_config(crop_name: str):
    module_path = f"config.{crop_name}"
    cfg = importlib.import_module(module_path)
    return cfg


config = load_config(CROP)
forecast_scenarios = config.forecast_scenarios

job_name_prefix = f"{CROP}_forecast"
output_base_dir = f"/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train/forecast/{CROP}"
working_dir = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src/train"

# Ensure output directory exists
if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists(output_base_dir):
    print(f"📁 Creating output directory: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)

# --- SUBMISSION LOOP ---
for month, seq_length in forecast_scenarios.items():
    # 1. Generate Unique ID
    unique_id = str(uuid.uuid4())[:8]  # e.g., 'a1b2c3d4'

    # 2. Construct Python Command
    python_cmd = (
        f"python train_CropFusionNet.py "
        f"--job_id {unique_id} "
        f"--seq_length {seq_length} "
        f"--output_dir {output_base_dir}/{month} "
    )

    # 3. Create SLURM Script Content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name_prefix}_{month}_{unique_id}
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

echo "Starting job {unique_id} for {month}..."
{python_cmd}
"""

    # 4. Write and Submit Job
    script_filename = f"job_{month}_{unique_id}.sh"
    with open(script_filename, "w") as f:
        f.write(slurm_script)

    print(f"Submitting Job for {month} (ID: {unique_id})")

    # Execute sbatch
    subprocess.run(["sbatch", script_filename])

    # Cleanup: Remove the .sh file after submission to keep folder clean
    os.remove(script_filename)

print("✅ All forecast scenario jobs submitted!")
