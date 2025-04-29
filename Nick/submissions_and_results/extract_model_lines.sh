#!/bin/bash
# This script is to extract scores from the scores of the models

# Define an array of model names
models=(
    "bert-base-multilingual-cased"
    "xlm-roberta-base"
    "xlm-roberta-large-10ep-M"
    "umt5-base-10ep-M"
    "umt5-small-10ep-M"
    "bert-base-cased"
    "google/flan-t5-base"
    "microsoft/deberta-v3-base"
    "deepset/roberta-base-squad2"
)

# Loop over each model and process the scores.txt file
for model in "${models[@]}"; do
    # Run awk command for each model to extract lines
    awk -v model="$model" '
    $0 ~ model { 
        print prev; 
        print $0; 
        getline; 
        print $0; 
        getline; 
        print $0 
    } 
    { prev = $0 }
    ' scores.txt > "${model}.txt"
    
    echo "Processed model: $model, output saved to ${model}.txt"
done

echo "All models processed successfully!"

