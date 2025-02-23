#!/bin/bash
#SBATCH --job-name=roberta_on_all_valSet
#SBATCH --output=./err_logs/roberta_on_all_valSet.%j.out
#SBATCH --error=./err_logs/roberta_on_all_valSet.%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
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

# language(s)
LANGS="ar de en es fi fr hi it zh cs ca eu fa"

# base models to finetune
# MODELS=("bert-base-multilingual-cased-10ep-M"  "umt5-base-10ep-M" "umt5-small-10ep-M" "xlm-roberta-base-10ep-M") 
MODELS=("xlm-roberta-large-10ep-M")

for model in "${MODELS[@]}"; do
    echo "fine-tuning $model, training language(s): ${LANGS[*]}"
    export MODEL=$model
    export LANGS=$LANGS
    python3 further_fine_tune_allVal.py
done

echo "Finishing at `date`"