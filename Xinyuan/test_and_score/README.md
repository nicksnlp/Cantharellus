# This directory contain scripts that generate model output and scores for 1. self-fine-tuned models and 2. pretrained NER model (benchmark).

### The main script (test_score.py):
Takes a fine-tuned model, generates outputs from test set and evaluates the model's performance by comparing the model output with reference output.
Model output and scores are stored in 2 seperate files.

- Model output:
  * feeds test set to fine-tuned models from a "models" directory (due to space limit, this directory is not cintained here)
  * stored as JSONL file
- Scores:
  * imported functions from scorer.py
  * evaluate model performance by Cor and IoU scores
  * stored in a CSV file (cumulative, stores scores for all models)

### Helper function 1: tokenize_align.py
- tokenizes input text before feeding text into models

### Helper function 2: scorer.py  
- provided by the organizer of this task
- calculates Cor score (with soft labels) and IoU (with hard labels) score between model output and reference.

### test_score.sh
- SLURM script that runs test_score.py
- processes multiple models and test sets at one time, models and test sets can be adjuted based on one's need
--------------------------------------------------------
## NER benchmark:
model (multilingual): 51la5/roberta-large-NER

### NER_test_and_score.py
- generate outputs and scores for the test set

### NER_test_score.sh
- SLURM script that runs test_score.py
- processes multiple test sets at one time, models and test sets can be adjuted based on one's need
