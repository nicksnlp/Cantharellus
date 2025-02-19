#!/bin/bash
#SBATCH --job-name=roberta-predict
#SBATCH --output=./err_logs/roberta-predict_%j.out
#SBATCH --error=./err_logs/roberta-predict_%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nikolay.vorontsov@helsinki.fi

echo "Starting at `date`"
set -e  # stops the script when encountering an error

source /projappl/project_2011335/peft_fine/bin/activate

source ~/.bashrc_project_2011335

echo "environment peft_fine is ACTIVE"

# Run the script
#srun merge_llama_and_push.py
#srun merge_llama_and_predict_TEST-SET.py
#srun train_llama.py
srun roberta_predict_TEST-SET_only_debugging.py

echo "Finishing at `date`"

deactivate

echo "environment peft_fine deactivated"

