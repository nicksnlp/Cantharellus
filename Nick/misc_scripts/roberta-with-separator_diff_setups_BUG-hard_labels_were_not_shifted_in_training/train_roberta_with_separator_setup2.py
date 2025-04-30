#!/usr/bin/env python3

## some adjustments made with https://chatgpt.com/share/67b52632-d31c-800b-84a2-e60a81a82ea4
import os
import torch
import json
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
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
    run = wandb.init(project='xlm-roberta-hallucination', job_type="training", anonymous="allow", name="test_5")

# Model and dataset paths
model_name = "FacebookAI/xlm-roberta-large"
dataset_path = "./shuffled_data.jsonl"
new_model = "nicksnlp/xlm-roberta-hallucination"
checkpoint_dir = "./CHECKPOINTS_xlm_setup2/"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add custom separator <@@> if needed
special_tokens = {"additional_special_tokens": ["<@@>"]}
tokenizer.add_special_tokens(special_tokens)

# Load model (without LoRA or bitsandbytes)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, device_map="auto"
)

# Resize model embeddings to accommodate the new token
model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Function to preprocess data
def preprocess_data(batch):
    tokenized_input = tokenizer(
        batch['text'],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    ## here is a bug!! The hard labels must be shifted
    aligned_labels = []
    for i, sentence in enumerate(batch['text']):
        hard_labels = batch['hard_labels'][i]
        offset_mapping = tokenized_input['offset_mapping'][i]
        sentence_labels = []

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                sentence_labels.append(-100)
            else:
                label = 0
                for (label_start, label_end) in hard_labels:
                    if start >= label_start and end <= label_end:
                        label = 1
                        break
                sentence_labels.append(label)

        aligned_labels.append(sentence_labels)

    return {
        'input_ids': tokenized_input['input_ids'],
        'labels': torch.tensor(aligned_labels),
        'attention_mask': tokenized_input['attention_mask']
    }

# Load dataset
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

data = load_jsonl(dataset_path)
dataset = Dataset.from_dict({
    'text': [item['model_input'] + "<@@>" + item['model_output_text'] for item in data],
    'hard_labels': [item['hard_labels'] for item in data]
})

tokenized_data = dataset.map(preprocess_data, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define label names
model.config.id2label = {0: "correct", 1: "hallucinated"}
model.config.label2id = {"correct": 0, "hallucinated": 1}

# Training arguments
training_arguments = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    #gradient_accumulation_steps=8,
    #optim="adamw_torch",
    #save_steps=300,
    save_strategy="epoch",  # Saves model at each epoch
    save_total_limit=10,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    #fp16=True,  # Using fp16 instead of 4-bit quantization
    #max_grad_norm=0.3,
    #warmup_ratio=0.3,
    #group_by_length=True,
    #lr_scheduler_type="linear",
    report_to="wandb",
    run_name="test_5",
    resume_from_checkpoint=True,
    save_safetensors=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_arguments,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the fine-tuned model
new_model_local_path = checkpoint_dir + "/new_model_local"
trainer.model.save_pretrained(new_model_local_path)
wandb.finish()
model.eval()

"""
# Push to Hugging Face Hub
login(token=HUGGING_API)
model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)

"""
