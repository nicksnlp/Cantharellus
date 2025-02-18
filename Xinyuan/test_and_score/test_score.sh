#!/bin/bash
#SBATCH --job-name=test_and_score
#SBATCH --output=./err_logs/test_and_score.%j.out
#SBATCH --error=./err_logs/test_and_score.%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --time=01:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xinyuan.mo@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
set -e

# load modules
module load pytorch/2.4


# base models:
# EN monolingual:
# MODELS=("bert-base-cased" "deepset/roberta-base-squad2" "google/flan-t5-base" "microsoft/deberta-v3-base")

# multilingual:
MODELS=("google-bert/bert-base-multilingual-cased" "FacebookAI/xlm-roberta-large" "FacebookAI/xlm-roberta-base" "google/umt5-base" "google/umt5-small")


# test languages
LANGS=("ar" "de" "en" "es" "fi" "fr" "hi" "it" "zh" "cs" "ca" "eu" "fa") # for multilingual models
# LANGS=("en") # for monolingual models

# # iterate over models and languages
# # 1. score models finetuned on test_lang ONLY
# for model in "${MODELS[@]}"; do
#     for lang in "${LANGS[@]}"; do
#         MODEL="${model}+${lang}Val"
#         echo "Testing model: $MODEL on language: $lang"
#         python3 test_and_score.py --model "$MODEL" --lang "$lang"
#     done
# done

# 2. score models finetuned on ALL validation sets
for model in "${MODELS[@]}"; do
    for lang in "${LANGS[@]}"; do
        MODEL="${model}+${lang}"
        # MODEL="${model}"
        echo "Testing model: $MODEL on language: $lang"
        python3 test_and_score.py --model "$MODEL" --lang "$lang"
    done
done

echo "Finishing at `date`"