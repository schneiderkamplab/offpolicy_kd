#!/bin/bash
#SBATCH --job-name=my-python-job
#SBATCH --account=project_465001265       # <-- Your LUMI project ID
#SBATCH --partition=standard-g          # or standard-small, dev-g, etc.
#SBATCH --time=01:00:00                 # HH:MM:SS
#SBATCH --gpus=1                      # Number of GPUs to request
#SBATCH --mem=64G                     # Total RAM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/output_%j.txt     # %j = job ID
#SBATCH --error=logs/error_%j.txt

# Load modules (adjust Python version if needed)
module load cray-python/3.11.7

# Activate your virtual environment
source LUMIdistil/bin/activate

# Run your script
python distill.py ../../dfm-data/pre-training/scandi-wiki/documents/da.jsonl.gz --distillation
