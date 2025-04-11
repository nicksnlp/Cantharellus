# This directory contains scripts for finetuning pre-trained LLMs for token level hallucination detection, wich consist of 2 parts:

1. Model fine-tuning (subdirectory "train")
1. Model output generation and evaluation (subdirectory "test_and_score")
---------------------------------------------------------------------------

Pretrained models involved the following:

EN monolingual:
- bert-base-cased
- deepset/roberta-base-squad2
- microsoft/deberta-v3-base

Multilingual:
- google-bert/bert-base-multilingual-cased
- FacebookAI/xlm-roberta-large
- FacebookAI/xlm-roberta-base
- google/umt5-base
- google/umt5-small

