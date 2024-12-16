# Test the model:
# generate predictions (ONLY soft labels) & save them to a JOSNL file
# The output JOSNL contains:
#           1. id (e.g., "val-en-1")
#           2. model_input
#           3. model_output_text
#           4. soft_lables
# ----------------------------------------------------------------
import os
import torch
import torch.nn.functional as F
import json
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
from transformers.pipelines.pt_utils import KeyPairDataset
from json2dataset import file_reader        # helper function 1: load data from a json file
from tokenize_align import tokenize_input   # helper function 2: tokenize input text


# load the fine-tuned model & its tokenizer

# get model outputs from a SINGLE entry (1 model_input + 1 model_output_text)
def get_outputs(tokenizer, model, data):

    # 1. tokenize the input data --------------------------------------
    test_inputs = tokenize_input(data, tokenizer)
        
    # takes offset_mappings out and store it seperately before passing inputs to the model
    offset_mapping = test_inputs.pop("offset_mapping")

    
    # 2. get model outputs --------------------------------------
    # run the model & get the logits
    with torch.no_grad():                    # disable model parameter updates when getting outputs
        outputs = model(**test_inputs)       # the "**" converts dict into argument ("key": val -> key = val)
        logits = outputs.logits
        probs = F.softmax(logits, dim = -1)    # calculate probabilities of predicted labels using softmax

    # mask tokens from model_input
    predictions = logits.argmax(dim = -1)
    token_type_ids = test_inputs['token_type_ids']
    masked_predictions = predictions * (token_type_ids == 1) 
    
    # get predicted labels from logits
    predicted_label_ids = masked_predictions.squeeze().tolist()  # dim = -1: appliy argmax function to the LAST dimension of the logits tensor, AKA the label indices
    predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids]  # map label ID to labels

    # # get predicted labels from logits
    # predicted_label_ids = logits.argmax(dim = -1).squeeze().tolist()  # dim = -1: appliy argmax function to the LAST dimension of the logits tensor, AKA the label indices
    # predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids]  # map label ID to labels

    # 3. get the hallucination spans and save them as soft labels --------------------------------------
    spans = []
    current_span = []
    current_probs = []

    for label, (start, end), prob in zip(predicted_labels, offset_mapping.squeeze().tolist(), probs.squeeze()):
        
        # 3.1 ignore any label if the current token is a [CLS] or [SEP] (which have offsets as (0, 0))
        if start == 0 and end == 0:
            continue
        
        # 3.2 otherwise, update current_span when hitting an "I" label
        else:    
            if label == "I":
                i_prob = prob[model.config.label2id["I"]].item()  # probability of predicting label "I" for the current token
                if not current_span:  
                    # initialize current_span if it's empty
                    current_span = [start, end]
                    current_probs = [i_prob]
                else:
                    # if current_span is not empty, extend it to include the current token
                    current_span[1] = end   # renew the end point of the current span
                    current_probs.append(i_prob)  # add prob of the current token 
                    
            # 3.3 stop updating the current span when hitting an "O" label
            else:
                if current_span:
                    # if there was a current_span before, store it to "spans" and empty it for the next hallucination
                    average_prob = sum(current_probs) / len(current_probs)  # calculate average prob for the entire hallucination span
                    spans.append((current_span[0], average_prob, current_span[1]))
                    current_span = []
                    current_probs = []

    
    # append the last hallucination span (if the ending label is not "O")
    if current_span:
        average_prob = sum(current_probs) / len(current_probs)
        spans.append((current_span[0], average_prob, current_span[1]))
    
    return spans


# get model outputs for ALL entries within a dataset 
def get_model_outputs(model, tokenizer, test_dataset):
    
    # store outputs for all the entries
    all_outputs = []
    
    for data in test_dataset:
        # initialize output for the current entry
        output = {"id":data["id"], 
                  "model_input":data["model_input"],
                  "model_output_text":data["model_output_text"],
                  "soft_labels": []}
        
        # add predictions if hallucination is detected (i.e., 'spans' is not empty)
        spans = get_outputs(tokenizer, model, data)
        if spans:
            for start, prob, end in spans:
                output["soft_labels"].append({"start": start, "prob": round(prob, 1), "end": end})
        
        all_outputs.append(output)
        
    return all_outputs


if __name__ == "__main__": 
    
    # 1. load the test dataset (gold standard)
    test_file_name = "mushroom.en-val.v2.jsonl"
    test_data = file_reader("./testing_data/" + test_file_name)
    test_dataset = Dataset.from_dict({key: [d[key] for d in test_data] for key in test_data[0]}) # convert test data to Dataset object

    # 2. get model outputs stored in JSONL files
    # all the models to be tested:
    model_names = ["bert_vanilla"] 
    
    for model_name in model_names:
        # load each fine-tuned model & its tokenizer
        model_path = "./models/" + model_name
        model = BertForTokenClassification.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        
        # get model outputs for all entries in test dataset
        all_outputs = get_model_outputs(model, tokenizer, test_dataset)
        
        # save results to a JSONL file (for scoring later)
        with open("./results/"+ model_name + "_results.jsonl", "w") as f:
            for item in all_outputs:
                f.write(json.dumps(item) + "\n")
