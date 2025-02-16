#!/bin/bash
#SBATCH --job-name=ner_test_n_score
#SBATCH --output=./err_logs/ner_test_and_score.%j.out
#SBATCH --error=./err_logs/ner_test_and_score.%j.err
#SBATCH --account=project_2011335
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
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


# test languages
LANGS=("ar" "de" "en" "es" "fi" "fr" "hi" "it" "zh" "cs" "ca" "eu" "fa") # for multilingual models

# score NER model for ALL validation sets
for lang in "${LANGS[@]}"; do
    echo "Testing NER on language: $lang"
    python3 NER_test_and_score.py --lang "$lang"
done

echo "Finishing at `date`"