#!/bin/bash
#SBATCH --job-name=train_best_model         # Job name
#SBATCH --output=train_best_model_%j.log    # Standard output and error log (%j = job ID)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=8                      # Number of tasks (cores)
#SBATCH --time=3-00:00:00               # Max walltime (3 days here)
#SBATCH --partition=gpu                 # Partition/queue to use
#SBATCH --mem=32G                       # Memory per node
#SBATCH --gres=gpu:v100:1               # Request 1x V100 GPU

# -------------------------------
# Environment setup
# -------------------------------
module load Anaconda3/2024.06   # Provides Python base
module load cuda/12.6.3         # Match PyTorch build


source /group/pmc052/schew/venv/bin/activate

# -------------------------------
# Run script
# -------------------------------
echo "Starting training at $(date)"
python /group/pmc052/schew/train_best_model.py
echo "Finished at $(date)"
