#!/usr/bin/env python3
import jsonlines
import os
import torch
import json
import regex as re
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Hugging Face login
HUGGING_API = os.getenv("HF_TOKEN")

if HUGGING_API:
    from huggingface_hub import login
    login(token=HUGGING_API)

# Model and dataset paths
model_name = "FacebookAI/xlm-roberta-large"
dataset_path = "./shuffled_data.jsonl"

# Fine-tuned model path
new_model_local_path = "./CHECKPOINTS_xlm/new_model_local"

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Add custom separator <@@>
special_tokens = {"additional_special_tokens": ["<@@>"]}
tokenizer.add_special_tokens(special_tokens)

# Load model (without LoRA or bitsandbytes)
model = AutoModelForTokenClassification.from_pretrained(
    new_model_local_path, num_labels=2, device_map="auto"
)

# Resize model embeddings to accommodate the new token
model.resize_token_embeddings(len(tokenizer))

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Check for CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def infer_with_model(input_text):
    inputs = tokenizer(input_text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=128,
                       return_offsets_mapping=True,
                       add_special_tokens=True)

    offset_mapping = inputs.pop("offset_mapping")[0].tolist()  # Extract character spans
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Send to correct device

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Model outputs

    predicted_labels = torch.argmax(logits, dim=-1)[0].tolist()  # Convert to list
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    labeled_tokens = []
    hallucinated_words = []

    for token, label, (start, end) in zip(tokens, predicted_labels, offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens
            continue
        labeled_tokens.append((token, label, (start, end)))  # Add character positions

        if label == 1:
            hallucinated_words.append(input_text[start:end])  # Extract hallucinated word

    return hallucinated_words, labeled_tokens

def export_spans_using_offsets(tokens, full_text):
    trim_marker = "<@@>"
    marker_pos = full_text.find(trim_marker)

    if marker_pos == -1:
        raise ValueError("Trim marker <@@> not found in full_text!")

    answer_start = marker_pos + len(trim_marker)
    trimmed_text = full_text[answer_start:]

    spans = []
    start = None

    for token, label, (char_start, char_end) in tokens:
        if char_start == 0 and char_end == 0:
            continue

        if char_end <= answer_start:
            continue

        adj_start = max(char_start - answer_start, 0)
        adj_end = char_end - answer_start

        if label == 1:
            if start is None:
                start = adj_start
        else:
            if start is not None:
                spans.append((start, adj_start))
                start = None

    if start is not None:
        spans.append((start, adj_end))

    extracted_texts = [trimmed_text[start:end] for start, end in spans]
    
    return trimmed_text, spans, extracted_texts


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
            qa_pair = question + "<@@>" + model_output_text

            hallucinated_words, labeled_tokens = infer_with_model(qa_pair)
            _, hard_labels, _ = export_spans_using_offsets(labeled_tokens, qa_pair)

            datapoint["hard_labels"] = hard_labels
            writer.write(datapoint)

# Process multiple languages
for lang in ["ar", "ca", "cs", "de", "en", "es", "eu", "fa", "fi", "fr", "hi", "it", "zh"]:
    validation_file = f"./v1/mushroom.{lang}-tst.v1.jsonl"
    output_file = f"./v1/mushroom.{lang}-tst.v1.jsonl_xml-roberta-large-separator-shuffled-all-data"

    with open(output_file, "w") as file:
        pass  # Ensure the file exists

    print(f"{output_file} is created or already exists.")

    test_inferences(validation_file, output_file)
    print(f"Processed data written to {output_file}")

print("Predictions are made.")
