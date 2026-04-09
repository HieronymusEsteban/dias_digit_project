#!/bin/bash
#SBATCH --job-name=multi_object_rec_qwen3vl_analysis
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=rtx4090:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stephane.hess@students.unibe.ch

# Load BOTH modules
module load Anaconda3/2024.02-1
module load CUDA/12.6.0

# Activate your virtual environment
source ~/venvs/test-qwen-env/bin/activate

# Run your script
python rec_multi_object_qwen3vl.py
