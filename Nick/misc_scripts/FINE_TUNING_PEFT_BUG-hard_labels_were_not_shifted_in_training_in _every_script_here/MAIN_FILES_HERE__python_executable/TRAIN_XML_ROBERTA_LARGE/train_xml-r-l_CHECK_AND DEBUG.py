#!/usr/bin/env python3
## https://chatgpt.com/share/679ea6f1-0d30-800b-a963-7880da492db7
import os
import json
import torch
import jsonlines
import glob
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments
from transformers import BitsAndBytesConfig, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# Hugging Face and W&B login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# Model and dataset paths
MODEL_NAME = "xlm-roberta-large"
DATASET_PATH = "./shuffled_data.jsonl"
NEW_MODEL = "nicksnlp/xlm-roberta-hallucination"
CHECKPOINT_DIR = "./CHECKPOINTS/"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
        offset_mapping = tokenized_input['offset_mapping'][i]
        sentence_labels = []
        for start, end in offset_mapping:
            if start == end:
                sentence_labels.append(-100)
            else:
                label = any(start >= s and end <= e for s, e in hard_labels)
                sentence_labels.append(1 if label else 0)
        aligned_labels.append(sentence_labels)
    return {'input_ids': tokenized_input['input_ids'], 'labels': torch.tensor(aligned_labels), 'attention_mask': tokenized_input['attention_mask']}

# Load dataset
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

data = load_jsonl(DATASET_PATH)
dataset = Dataset.from_dict({'text': [d['model_input'] + "<@@>" + d['model_output_text'] for d in data], 'hard_labels': [d['hard_labels'] for d in data]})
tokenized_data = dataset.map(preprocess_data, batched=True)

# Load model & training setup
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2, quantization_config=bnb_config, device_map={"": 0})
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(lora_alpha=8, lora_dropout=0.1, r=16, task_type="TOKEN_CLS", target_modules=["query", "key", "value", "output"])

# Training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR, num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=2,
    save_steps=300, save_total_limit=3, logging_steps=10, learning_rate=2e-4, weight_decay=0.001, 
    fp16=False, bf16=False, max_grad_norm=0.3, warmup_ratio=0.3, group_by_length=True, lr_scheduler_type="linear"
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
trainer = SFTTrainer(model=model, train_dataset=tokenized_data, peft_config=peft_config, processing_class=tokenizer, args=training_args, data_collator=data_collator)
trainer.train()

# Save & merge model
trainer.model.save_pretrained(CHECKPOINT_DIR + "/fine_tuned_model")
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR + "/fine_tuned_model").merge_and_unload()

"""
model.push_to_hub(NEW_MODEL)
tokenizer.push_to_hub(NEW_MODEL)
"""

# Prediction setup
model.to("cuda" if torch.cuda.is_available() else "cpu")

def infer_with_model(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128, return_offsets_mapping=True, add_special_tokens=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_labels = torch.argmax(logits, dim=-1)[0].tolist()
    return [(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())[i], predicted_labels[i], offset_mapping[i]) for i in range(len(predicted_labels))]

def extract_spans(tokens, full_text):
    marker_pos = full_text.find("<@@>") + len("<@@>")
    trimmed_text = full_text[marker_pos:]
    spans = []
    start = None
    for token, label, (s, e) in tokens:
        if e <= marker_pos:
            continue
        adj_start, adj_end = max(s - marker_pos, 0), e - marker_pos
        if label == 1:
            if start is None:
                start = adj_start
        else:
            if start is not None:
                spans.append((start, adj_start))
                start = None
    if start is not None:
        spans.append((start, adj_end))
    return trimmed_text, spans, [trimmed_text[s:e] for s, e in spans]

def test_inferences(validation_file, output_file):
    with jsonlines.open(validation_file) as reader, jsonlines.open(output_file, 'w') as writer:
        for datapoint in reader:
            qa_pair = datapoint.get("model_input", "") + "<@@>" + datapoint.get("model_output_text", "")
            hallucinated_tokens = infer_with_model(qa_pair)
            _, hard_labels, _ = extract_spans(hallucinated_tokens, qa_pair)
            datapoint["hard_labels"] = hard_labels
            writer.write(datapoint)

for lang in ["ar", "ca", "cs", "de", "en", "es", "eu", "fa", "fi", "fr", "hi", "it", "zh"]:
    validation_file = f"./v1/mushroom.{lang}-tst.v1.jsonl"
    output_file = f"./v1/mushroom.{lang}-tst.v1.jsonl_xlm_roberta-hallucinations"
    open(output_file, "w").close()
    print(f"Processing {lang}...")
    test_inferences(validation_file, output_file)
    print(f"Saved predictions to {output_file}")
