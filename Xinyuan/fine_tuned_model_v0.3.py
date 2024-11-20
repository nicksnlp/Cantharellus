#!/usr/bin/env python
# coding: utf-8
# %%
# pip install transformers datasets


# %%
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# some helper functions:
from json2dataset import file_reader
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


# read dataset from json file
file_addr = "./mushroom.en-val.v2.jsonl"
dataset  = file_reader(file_addr)

# fetch the trainng & vaidation data
train_data = dataset["train"]
val_data = dataset["validation"]




# %%
# tokenize and align labels for multiple spans
train_dataset = train_data.map(tokenize_n_align, batched=True,fn_kwargs={'tokenizer': tokenizer, 'label2id': label2id})
val_dataset = val_data.map(tokenize_n_align, batched=True,fn_kwargs={'tokenizer': tokenizer, 'label2id': label2id})


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

# --------------------------------
# just for testing the code! change this afterwards!
eval_dataset = train_dataset
# --------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# train the model (fine-tuning)
trainer.train()

# save the fine-tuned model & tokenizer
fine_tuned_model_path = "./qa_model"
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)



# %%
# # **3. Visualize output of the fine-tuned model**


# load the fine-tuned model & its tokenizer
new_model = BertForTokenClassification.from_pretrained(fine_tuned_model_path)
new_tokenizer = BertTokenizerFast.from_pretrained(fine_tuned_model_path)

# a test example (question & answer)
test_example = {
    "model_input": "Where is the University of Helsinki located?",
    "model_output_text": "The University of Helsinki is located in Helsinki, Poland."
}

# tokenize the example above
test_inputs = new_tokenizer(
    test_example["model_input"], # question
    test_example["model_output_text"], # answer
    truncation=True,  # split the text if it exceedes the maximum length
    padding="max_length",
    return_offsets_mapping=True, # offsets mapping = a list of (start, end), marking position of tokens in the original text
    return_tensors="pt" # return output as pyTorch tensor
)

# remove offset mappings before passing inputs to the model
offset_mapping = test_inputs.pop("offset_mapping")

# run the model & get the logits
with torch.no_grad():
    outputs = new_model(**test_inputs) # ** converts dict into argument ("key": val -> key = val)
logits = outputs.logits

# calculate probabilities of predicted labels using softmax
probs = F.softmax(logits, dim=-1)

# get predicted labels for each token using argmax
predicted_label_ids = logits.argmax(dim=-1).squeeze().tolist()  # dim = -1: appliy argmax function to the LAST dimension of the logits tensor, AKA the label indices
predicted_labels = [new_model.config.id2label[label_id] for label_id in predicted_label_ids] # map label ID to labels

# get the hallucination spans (tokens labeled as "I")
spans = []
current_span = []
current_probs = []




# %%
for label, (start, end), prob in zip(predicted_labels, offset_mapping.squeeze().tolist(), probs.squeeze()):

    # ignore special tokens' labels (e.g., [SEP], offsets with start & end as [0, 0])
    if start == 0 and end == 0:
        continue

    i_prob = prob[new_model.config.label2id["I"]].item()  # probability of label "I"

    if label == "I":
        if not current_span:
            current_span = [start, end]
            current_probs = [i_prob]
        else:
            # extend the current span to include the current token
            current_span[1] = end
            current_probs.append(i_prob)
    else:
        # end the current span when encountering an "O" label
        if current_span:
            average_prob = sum(current_probs) / len(current_probs)  # Calculate average probability for the span
            spans.append((current_span[0], average_prob, current_span[1]))
            current_span = []
            current_probs = []

# append remaining spans
if current_span:
    average_prob = sum(current_probs) / len(current_probs)
    spans.append((current_span[0], average_prob, current_span[1]))

# display the results
print("soft_labels:")
for start, prob, end in spans:
    span_text = test_example["model_output_text"][start:end]
    print(f"start: {start}, prob: {prob:.3f}, end: {end}, text: '{span_text}'")




# %%
# # Inspecting the output contents:


# import numpy as np

# # decoded tokens (with special tokens like [SEP])
# input_ids = test_inputs['input_ids'][0]  # get the input IDs tensor
# decoded_tokens = [tokenizer.decode([id]) for id in input_ids if id != tokenizer.pad_token_id]

# # get tokens with "I" label
# np_pred_labels, np_tokens = np.array(predicted_labels), np.array(decoded_tokens)
# mask = (np_pred_labels == 'I')
# result = np_tokens[mask[:len(np_tokens)]]


# print("Decoded Tokens:", decoded_tokens)
# print("Predicted Labels:", predicted_labels) # "I" or "O" for each token (include padding!)
# print("Hallucinated Token ID:",[(i, label) for i, label in enumerate(predicted_labels) if label == "I"])
# print("Hallucinated Tokens:", result)

