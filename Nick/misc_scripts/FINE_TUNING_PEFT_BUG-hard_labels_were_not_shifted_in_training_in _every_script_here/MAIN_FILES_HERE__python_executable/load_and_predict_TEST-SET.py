#!/usr/bin/env python3
## Nikolay Vorontsov,
## Mushroom task
## Inference with fine-tuned model

import jsonlines
import re
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import login
import torch

HUGGING_API = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HUGGING_API:
    raise ValueError("HF_TOKEN is not set. Please configure your Hugging Face token.")

# Login to Hugging Face
login(token=HUGGING_API)

tokenizer = AutoTokenizer.from_pretrained("nicksnlp/llama-7B-hallucination")
model = AutoModelForTokenClassification.from_pretrained("nicksnlp/llama-7B-hallucination")

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

"""
    print("Original Full Text:#"+ full_text)
    print("Trimmed Text:#"+ trimmed_text)
    print("Adjusted Spans of Label 1:", spans)
    print("Extracted Answer Texts:", extracted_texts)
"""
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
