#!/bin/bash
#SBATCH --job-name=llama_finetune
#SBATCH --output=./err_logs/finetune_%j.out
#SBATCH --error=./err_logs/finetune_%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00 
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nikolay.vorontsov@helsinki.fi

echo "Starting at `date`"
set -e  # stops the script when encountering an error

source /projappl/project_2011335/peft_fine/bin/activate

source ~/.bashrc_project_2011335

#ln -s /scratch/project_2011335/PEFT/project_local ~/.local

echo "environment peft_fine is ACTIVE"

# Run the training script
srun train_llama.py
#python3 train_llama.py

echo "Finishing at `date`"

deactivate

echo "environment peft_fine deactivated"

