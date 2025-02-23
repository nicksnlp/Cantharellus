#!/usr/bin/env python
# coding: utf-8
# %%
import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification 

# helper functions:
from tokenize_align import tokenize_n_align
from datasets import load_dataset, concatenate_datasets

import wandb

# Hugging Face and W&B login
HUGGING_API = os.getenv("HF_TOKEN")
WANDB_key = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if HUGGING_API:
    from huggingface_hub import login
    login(token=HUGGING_API)

if WANDB_key:
    wandb.login(key=WANDB_key)
    run = wandb.init(project='xlm-roberta-hallucination', job_type="training", anonymous="allow", name="special_token_all_val")

# %%
# check if GPU's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# read data from multiple jsonl files and tokenize them
def data_preprocess(tokenizer, LANGS):
    exceptional_langs = ["cs", "ca", "eu", "fa"]
    # training data address (val set):
    original_val_set = [f'./training_data/mushroom.{lang}-val.v2.jsonl' for lang in LANGS if lang not in exceptional_langs] 
    print(f"training data pack 1: {original_val_set}")
    
    out_val_set = [f'./training_data/mushroom.{lang}-val.v2.jsonl' for lang in exceptional_langs]
    print(f"training data pack 2: {out_val_set}")

    # load training data contents
    dataset1 = load_dataset('json', data_files=original_val_set) 
    dataset2 = load_dataset('json', data_files=out_val_set) 

    # remove unwanted columns from the 2 groups of data & concatenate them together
    #relevant_columns = ['model_output_text', 'hard_labels']
    ## Nick
    relevant_columns = ["model_input", 'model_output_text', 'hard_labels']
    dataset1 = dataset1["train"].remove_columns([col for col in dataset1["train"].column_names if col not in relevant_columns])
    dataset2 = dataset2["train"].remove_columns([col for col in dataset2["train"].column_names if col not in relevant_columns])
    
    # concatenate the 2 datasets
    dataset = concatenate_datasets([dataset1, dataset2])
    dataset = dataset.shuffle() # shuffle the order of datapoints

    # tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_n_align(x, tokenizer), batched=True)
    
    # split into train and validation sets
    train_dataset, eval_dataset = tokenized_datasets.train_test_split(test_size=0.05).values()

    return train_dataset, eval_dataset

# 
def train(model_name, LANGS):

    # define labels & map labels to IDs for training
    labels = ["O", "I"]  # 'O' = outside span, 'I' = inside span
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    model_name = f"../models/{model_name}"

    # load tokenizer & model (resized with <@@>)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels)).to(device)
    model.config.id2label = id2label
    model.config.label2id = label2id

    #get data preprocessed
    train_dataset, eval_dataset = data_preprocess(tokenizer, LANGS)
    
    # define training epoch
    # diff_num_epochs = [3, 5, 10, 20]
    num_epochs = 10

    # set up the training argument & trainer
    training_args = TrainingArguments(
        output_dir = "./results",
        eval_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        num_train_epochs = num_epochs,
        weight_decay = 0.01,
        logging_dir = "./logs",
        logging_steps = 1,
        save_safetensors=False,
        report_to="wandb",
        run_name="special_token_all_val",
        save_strategy="epoch",  # Saves model at each epoch
        save_total_limit=3
    )
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer
    )
    
    # start fine-tuning
    trainer.train()
    
    # save the fine-tuned model & tokenizer to the directory "models"
    model_name_stripped = model_name.split("/")[-1]     # extract the last part of the model's name --> remove "/" to prevent the fine-tuned model to be stored in a separate directory
    fine_tuned_model_path = f"../models/{model_name_stripped}+allVal"

    model.save_pretrained(fine_tuned_model_path)
    tokenizer.save_pretrained(fine_tuned_model_path) ## also, with added token <@@>

if __name__ == "__main__":
    # get training data's name & model name from the os
    LANGS = (os.getenv("LANGS", "en")).split()  # a list of languages the model will be fine-tuned on
    model_name = os.getenv("MODEL", "default") # base model's name

    # check if the 3 variables are loaded properly
    print(f"Raw LANGS from environment: {os.getenv('LANGS')}")
    print("----------------------------")
    print(f"LANGS: {LANGS}")
    print(f"model_name: {model_name}")
    print("----------------------------")
    
    # fine tune the base model
    train(model_name, LANGS)

