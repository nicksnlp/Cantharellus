#!/bin/bash
#SBATCH --job-name=separator_fine-tune
#SBATCH --output=./err_logs/fine_tuning.%j.out
#SBATCH --error=./err_logs/fine_tuning.%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nikolay.vorontsov@helsinki.fi

echo "Starting at `date`"
set -e  # stops the script when encountering an error
#module load pytorch/2.4  # load module     

source /projappl/project_2011335/peft_fine/bin/activate

source ~/.bashrc_project_2011335

echo "environment peft_fine is ACTIVE"

# language(s):
LANGS="ar de en es fi fr hi it zh cs ca eu fa" # for multilingual models
# LANGS="en"  # for monolingual models

# base models:
# EN monolingual:
# MODELS=("bert-base-cased" "deepset/roberta-base-squad2" "google/flan-t5-base" "microsoft/deberta-v3-base")

# multilingual:
#MODELS=("google-bert/bert-base-multilingual-cased" "FacebookAI/xlm-roberta-large" "FacebookAI/xlm-roberta-base" "google/umt5-base" "google/umt5-small")
MODELS=("FacebookAI/xlm-roberta-large")

# iterate over models
for model in "${MODELS[@]}"; do
    echo "Fine-tuning $model, training language(s): $LANGS"
    
    # export variables to python script
    export MODEL=$model
    export LANGS="$LANGS"  
    
    # call py script
    python3 fine_tuning.py
done
 
echo "Finishing at $(date)"