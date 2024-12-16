#!/bin/bash
#SBATCH --job-name=fine_tuning
#SBATCH --output=err_logs/fine_tuning.%j.out
#SBATCH --error=err_logs/fine_tuning.%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuan.mo@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
set -e

# load modules
module load python-data/3.10-22.09
module load pytorch/2.4

# pip3 install numpy torch datasets transformers

# list of training data file names
DATA_LIST=(
    "train_zang_100.jsonl"
    "train_zang_188.jsonl"
    "train_nick_100.jsonl"
    "train_combined_100.jsonl"
    "train_combined_288.jsonl"
)

# loop over the list --> fine tune model on different datasets in one go
for DATA in "${DATA_LIST[@]}"; do

    # set environment variable (file name)
    export DATA="$DATA"
    
    # run python script for fine-tuing
    python3 fine_tuning_model_v0.5.py

done


echo "Finishing at `date`"