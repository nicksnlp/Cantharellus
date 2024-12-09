#!/usr/bin/env python
# coding: utf-8
# %%
# pip install transformers datasets


# %%
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import os

# helper functions:
from json2dataset import file_reader, data_splitter
from tokenize_align import tokenize_n_align

# check if GPU's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# # **1. Dataset Prep:**


# start experimenting with this model
model_name = "bert-base-cased"

# get tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# define labels & map labels to IDs for training
labels = ["O", "I"]  # 'O' = outside span, 'I' = inside span
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# get training data's file name from the os
file_name = os.getenv("DATA", "default")  

# read dataset from json file
file_addr = "./training_data/"+ file_name   
dataset  = data_splitter(file_reader(file_addr),9,1)    # split train-validation at a 9:1 ratio

# fetch the trainng & vaidation data
train_data = dataset["train"]
eval_data = dataset["validation"]

# tokenize and align labels for multiple spans
train_dataset = train_data.map(tokenize_n_align, batched=True,fn_kwargs={'tokenizer': tokenizer, 'label2id': label2id})
eval_dataset = eval_data.map(tokenize_n_align, batched=True,fn_kwargs={'tokenizer': tokenizer, 'label2id': label2id})




# %%
# # **2. Model Set-up:**


# prepare the model
model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(labels)).to(device)
model.config.id2label = id2label
model.config.label2id = label2id

# set up the trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=1,
    run_name = "qa_model"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# train the model (fine-tuning)
trainer.train()

# save the fine-tuned model & tokenizer to the directory "models"
name_tag = file_name.replace("train", "").replace(".jsonl", "") #strip the file name for training dataset --> save new models seperately
fine_tuned_model_path = "./models/" + name_tag
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

