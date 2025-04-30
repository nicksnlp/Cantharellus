#!/usr/bin/env python3
## THIS IS ONLY SET TO MERGE AND PUSH
## PUSHING DOES NOT WORK
import jsonlines
import os
import torch
import json
import glob
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import BitsAndBytesConfig,  AutoTokenizer, TrainingArguments , Trainer
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel, get_peft_model
from trl import SFTTrainer
import wandb
from huggingface_hub import login

# Hugging Face and W&B login
HUGGING_API = os.getenv("HF_TOKEN")
WANDB_key = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if HUGGING_API:
    from huggingface_hub import login

login(token=HF_TOKEN)

"""
if WANDB_key:
    wandb.login(key=WANDB_key)
    run = wandb.init(project='llama-7b-hallucination', job_type="training", anonymous="allow", name="test_4")
"""
# Model and dataset paths
model_name = "meta-llama/Llama-2-7b-hf"
dataset_path = "./shuffled_data.jsonl"#"/path/to/training_data.jsonl"

# Hugging face repository link to save fine-tuned model(Create new repository in huggingface,copy and paste here)
new_model = "nicksnlp/llama-7B-hallucination"

checkpoint_dir = "./CHECKPOINTS/" #"/path/to/checkpoints"
"""
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_eos_token = True
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token
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

    aligned_labels = []
    for i, sentence in enumerate(batch['text']):
        hard_labels = batch['hard_labels'][i]
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][i])
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
        'tokens': [tokenizer.convert_ids_to_tokens(ids) for ids in tokenized_input['input_ids']],
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

"""
# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.float16,
    bnb_4bit_use_double_quant= False,
)
"""
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, quantization_config=bnb_config, device_map={"": 0}
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="TOKEN_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Define label names (0 = correct, 1 = hallucinated)
model.config.id2label = {0: "correct", 1: "hallucinated"}
model.config.label2id = {"correct": 0, "hallucinated": 1}

# Training arguments
training_arguments = TrainingArguments(
    output_dir=checkpoint_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_8bit",
    save_steps=300,
    save_total_limit=3,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.3,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    run_name="test_4",
    resume_from_checkpoint=True
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_data,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
    data_collator=data_collator, # Use the data collator here
)

# Start training
trainer.train()
"""
# Save the fine-tuned model
new_model_local_path = checkpoint_dir + "/new_model_local"
"""
trainer.model.save_pretrained(new_model_local_path)
wandb.finish()
model.config.use_cache = True
model.eval()

"""
## START FROM HERE

# Load base model with quantization to reduce memory usage
base_model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
    quantization_config=bnb_config,  # Defined earlier
)

# Load the PEFT model and merge weights
model = PeftModel.from_pretrained(base_model, new_model_local_path)
#model = PeftModel.from_pretrained(base_model, "/content/drive/MyDrive/NLP/MODELS/FineTunedModel_test2/checkpoint-468")
model = model.merge_and_unload()

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

## PUSH UNCOMMENT/COMMENT OUT
model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)


"""
### HERE START PREDICTION
print("Loading merged model from Huggingface")
tokenizer = AutoTokenizer.from_pretrained(new_model)
model = AutoModelForTokenClassification.from_pretrained(new_model)

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def infer_with_model(input_text):
    # Tokenize input text with offset mapping and special tokens
    inputs = tokenizer(input_text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128,
                       return_offsets_mapping=True,
                       add_special_tokens=True
                       )

    # Move input tensors to model's device
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()  # Extract character spans
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Send to correct device

    # Predict token labels
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Model outputs
    predicted_labels = torch.argmax(logits, dim=-1)[0].tolist()  # Convert to list

    # Get tokenized words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    # Align tokens with labels and offset mappings
    labeled_tokens = []
    hallucinated_words = []

    for token, label, (start, end) in zip(tokens, predicted_labels, offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens (e.g., [CLS], [SEP])
            continue

        labeled_tokens.append((token, label, (start, end)))  # Add character positions

        if label == 1:
            hallucinated_words.append(input_text[start:end])  # Extract hallucinated word

    return hallucinated_words, labeled_tokens

import regex as re

def export_spans_using_offsets(tokens, full_text):
    # Step 1: Find the "@@" marker in full_text
    trim_marker = "<@@>"
    marker_pos = full_text.find(trim_marker)

    if marker_pos == -1:
        raise ValueError("Trim marker <@@> not found in full_text!")

    # Step 2: Determine the new start position
    answer_start = marker_pos + len(trim_marker)

    # Step 3: Trim text after "@@" marker
    trimmed_text = full_text[answer_start:]

    spans = []
    start = None

    for token, label, (char_start, char_end) in tokens:
        if char_start == 0 and char_end == 0:  # Skip special tokens
            continue

        if char_end <= answer_start:  # Ignore tokens before "@@"
            continue

        # Adjust character positions to be relative to the trimmed text
        adj_start = max(char_start - answer_start, 0)
        adj_end = char_end - answer_start

        if label == 1:
            if start is None:
                start = adj_start  # Start a new span
        else:
            if start is not None:
                spans.append((start, adj_start))  # Close span
                start = None

    if start is not None:
        spans.append((start, adj_end))  # Close last span

    # Step 4: Extract spans from the trimmed text
    extracted_texts = [trimmed_text[start:end] for start, end in spans]
    
    return trimmed_text, spans, extracted_texts

"""
    #print("Original Full Text:#"+ full_text)
    #print("Trimmed Text:#"+ trimmed_text)
    #print("Adjusted Spans of Label 1:", spans)
    #print("Extracted Answer Texts:", extracted_texts)
"""

# Example usage of the inference function
question = "Which municipalities does the Italian commune of Ponzone border?"
input_text = " YES, YES, << Ponza\n"
full_text = question+"<@@>"+input_text
hallucinated_words, labeled_tokens = infer_with_model(full_text)

# Print the list of hallucinated words
print("Hallucinated words:")
print(hallucinated_words)
print(full_text)
print(labeled_tokens)

t, ch, s = export_spans_using_offsets(labeled_tokens, full_text)
print(t)
print(ch)

t = "Yes, all arachnids have antennas. However, not all of them are visible to the naked eye."
#87, 88
#print(t[87:88])

def test_inferences(validation_file, output_file):
  with jsonlines.open(validation_file) as reader, jsonlines.open(output_file, 'w') as writer:
    for datapoint in reader:
            model_output_text = datapoint.get("model_output_text", "")
            question = datapoint.get("model_input", "")
            qa_pair = question+"<@@>"+model_output_text

            hallucinated_words, labeled_tokens = infer_with_model(qa_pair)

            _, hard_labels, _ = export_spans_using_offsets(labeled_tokens, qa_pair)

            datapoint["hard_labels"] = hard_labels
            #datapoint["id"] = datapoint["id"].replace('_unlabeled', '')

            writer.write(datapoint)

for lang in ["ar", "ca", "cs", "de", "en", "es", "eu", "fa", "fi", "fr", "hi", "it", "zh"]:

    validation_file = f"./v1/mushroom.{lang}-tst.v1.jsonl"
    output_file =     f"./v1/mushroom.{lang}-tst.v1.jsonl_llama_7b-hallucinations"

    with open(output_file, "w") as file:
        pass  # Do nothing, just ensure the file exists
    print(f"{output_file} is created or already exists.")
        
    # Make predictions:
    test_inferences(validation_file, output_file)
    
    print(f"Processed data written to {output_file}")

print("Predictions are made.")

"""
