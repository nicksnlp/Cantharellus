# Test the model:
# generate predictions (ONLY soft labels) & save them to a JOSNL file
# The output JOSNL contains:
#           1. id (e.g., "val-en-1")
#           2. model_input
#           3. model_output_text
            #-----------------------------
#           4. soft_lables (new content)
# ----------------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
import json
import csv
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from scorer import main as get_scores, load_jsonl_file_to_records as load_jsonl         # helper function 1: calculate IoU & Cor scores
from tokenize_align import tokenize_input                                               # helper function 2: tokenize input text


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

    
    predictions = logits.argmax(dim = -1)

    
    # get predicted labels from logits
    predicted_label_ids = predictions.squeeze().tolist()  # dim = -1: appliy argmax function to the LAST dimension of the logits tensor, AKA the label indices
    predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids]  # map label ID to labels

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
    
    all_outputs = []   # store outputs for all the entries

    for data in test_dataset:
        # initialize contents for the model prediciton file
        output = {"id":data["id"], 
                  "model_input":data["model_input"],
                  "model_output_text":data["model_output_text"],
                  "soft_labels": []}
        
        ## Nick: compute shift for the spans, make sure to extract only those in within the range, and remove spans from questions.
        length_of_question = len(data["model_input"]) + len(" <@@> ")

        # add predictions if hallucination is detected (i.e., 'spans' is not empty)
        spans = get_outputs(tokenizer, model, data)
        if spans:
            for start, prob, end in spans:

                ## Nick, shifted spans
                start = max(start - length_of_question, 0)
                end = max(end - length_of_question, 0)

                if start == 0 and end == 0:
                    continue

                else:
                    output["soft_labels"].append({"start": start, "prob": round(prob, 1), "end": end})
        
        all_outputs.append(output)
        
    return all_outputs


# ask a model to generate predicted lables from raw data
def get_predictions(test_lang, model_name_HGface):
   
    model_real_name = model_name_HGface.split("/")[-1]
    model_addr = f"../models/{model_real_name}-10ep-M+allVal"
    tokenizer = AutoTokenizer.from_pretrained(model_addr)
    model = AutoModelForTokenClassification.from_pretrained(model_addr)
    
    # load test data (the validation set)
    #test_dataset = load_jsonl(f"./testing_data/mushroom.{test_lang}-val.v2.jsonl")

    ## Nick, Load the test-set
    test_dataset = load_jsonl(f"./v1_labeled/mushroom.{test_lang}-tst.v1.jsonl")
    # test_dataset = Dataset.from_dict({key: [d[key] for d in test_data] for key in test_data[0]}) # convert test data to Dataset object

    # get model outputs for all entries in test dataset
    all_outputs = get_model_outputs(model, tokenizer, test_dataset)
    
    # save results to a JSONL file (for scoring later)
    new_prediction_addr = f"./results/{model_real_name}_{test_lang}.jsonl"
    with open(new_prediction_addr, "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item) + "\n")

    # return the prediction file addr
    return new_prediction_addr


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run model predictions and scoring.")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--lang", required=True, help="Language code to test")
    args = parser.parse_args()

    model_name = args.model
    test_lang = args.lang
    
    # # get predicitons from the model's output jsonl file
    predictions = load_jsonl(get_predictions(test_lang, model_name)) # model predictions
    

    # get scores
    ref_dicts = load_jsonl(f"./v1_labeled/mushroom.{test_lang}-tst.v1.jsonl") # the gold standard file 
    output_file = f"./scores/{model_name}-{test_lang}-score.txt"   # scores will be stored to this address
    ious, cors = get_scores(ref_dicts, predictions, output_file)

    # write scores to a csv file
    # file_path = './scores/all_scores.csv'  # for regular models
    file_path = './scores/further_finetuned_scores.csv'  # for further-finetuned models

    model_scores = {"model_name": model_name,
                    "IoU": ious,
                    "Cor": cors,
                    "test_lang": test_lang}
    
    columns = list(model_scores.keys())
    file_exist = os.path.exists(file_path)  # check if scores.csv already exists

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exist: 
            writer.writeheader()    # add column titles if the file is created for the first time
        writer.writerow(model_scores)
