# Test the model:
# generate predictions (soft & hard labels) & save them to a JOSNL file
# The output JOSNL contains:
#           1. id (e.g., "val-en-1")
#           2. model_input
#           3. model_output_text
#           4. soft_lables
#           5. hard_lables
# ----------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import json
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
from transformers.pipelines.pt_utils import KeyPairDataset
from json2dataset import file_reader    # helper function: load data from a json file


# load the fine-tuned model & its tokenizer

# get model outputs for each sample within the test dataset
def get_outputs(tokenizer, model, data):
    test_inputs = tokenizer(
        data["model_input"], # question
        data["model_output_text"], # answer
        truncation=True,  # split the text if it exceedes the maximum length
        padding="max_length",
        return_offsets_mapping=True, # offsets mapping = a list of (start, end), marking position of tokens in the original text
        return_tensors="pt" # return output as pyTorch tensor
    )

    # remove offset mappings before passing inputs to the model
    offset_mapping = test_inputs.pop("offset_mapping")

    # run the model & get the logits
    with torch.no_grad():
        outputs = model(**test_inputs) # ** converts dict into argument ("key": val -> key = val)
    logits = outputs.logits

    # calculate probabilities of predicted labels using softmax
    probs = F.softmax(logits, dim=-1)

    # get predicted labels for each token using argmax
    predicted_label_ids = logits.argmax(dim=-1).squeeze().tolist()  # dim = -1: appliy argmax function to the LAST dimension of the logits tensor, AKA the label indices
    predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids] # map label ID to labels

    # get the hallucination spans (tokens labeled as "I")
    spans = []
    current_span = []
    current_probs = []


    # %%
    for label, (start, end), prob in zip(predicted_labels, offset_mapping.squeeze().tolist(), probs.squeeze()):

        # ignore special tokens' labels (e.g., [SEP], offsets with start & end as [0, 0])
        if start == 0 and end == 0:
            continue

        i_prob = prob[model.config.label2id["I"]].item()  # probability of label "I"

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
    
    return spans


def get_model_outputs(model_name):
    # model_name = os.getenv("MODEL", "default")   # get model's name from the os (for SLURM script)
    
    fine_tuned_model_path = "./models/" + model_name
    model = BertForTokenClassification.from_pretrained(fine_tuned_model_path)
    tokenizer = BertTokenizerFast.from_pretrained(fine_tuned_model_path)
    
    # load the test data
    test_file_name = "mushroom.en-val.v2.jsonl"
    test_data = file_reader("./testing_data/" + test_file_name)
    test_dataset = Dataset.from_dict({key: [d[key] for d in test_data] for key in test_data[0]}) # convert test data to Dataset object
    
    # stores all outputs 
    all_outputs = []
    
    for data in test_dataset:
        spans = get_outputs(tokenizer, model, data)
        output = {"id":data["id"], 
                  "model_input":data["model_input"],
                  "model_output_text":data["model_output_text"],
                  "soft_labels": [], 
                  "hard_labels":[]}
    
        # add predictions if hallucination detected (i.e., variable 'spans' is not empty)
        if spans:
            output["soft_labels"] = []  
            for start, prob, end in spans:
                output["soft_labels"].append({"start": start, "prob": round(prob, 1), "end": end})
        
        all_outputs.append(output)
    return all_outputs


# +
# get outputs & save to JSONL file for all models
model_names = ["bert_nick_100", "bert_zang_100", "bert_zang_188", "bert_combined_100", "bert_combined_288"]

for model_name in model_names:
    all_outputs = get_model_outputs(model_name)
    
    # save results to a JSONL file (for scoring)
    with open("./results/"+ model_name + "_results.jsonl", "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item) + "\n")
