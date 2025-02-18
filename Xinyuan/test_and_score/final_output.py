# get model outputs for the final submission
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenize_align import tokenize_input                           # helper function: tokenize input text


# load the fine-tuned model & its tokenizer
def load_jsonl(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`. 
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly)."""
    df = pd.read_json(filename, lines=True)
    df = df[['id', 'model_input', 'model_output_text']]
    return df.sort_values('id').to_dict(orient='records')


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
        # add predictions if hallucination is detected (i.e., 'spans' is not empty)
        spans = get_outputs(tokenizer, model, data)
        if spans:
            for start, prob, end in spans:
                output["soft_labels"].append({"start": start, "prob": round(prob, 1), "end": end})
        all_outputs.append(output)
        
    return all_outputs


# ask a model to generate predicted lables from raw data
def get_predictions(test_lang, model_name):
    # load model & tokenizer
    model_addr = f"../models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_addr)
    model = AutoModelForTokenClassification.from_pretrained(model_addr)
    
    # load test set
    test_dataset = load_jsonl(f"./testing_data/final_submission_set/mushroom.{test_lang}-tst.v1.jsonl")

    # get model outputs for all datapoints
    all_outputs = get_model_outputs(model, tokenizer, test_dataset)
    
    # save model output to JSONL file 
    new_prediction_addr = f"./output_4_submission/{model_name}_{test_lang}.jsonl"
    with open(new_prediction_addr, "w") as f:
        for item in all_outputs:
            f.write(json.dumps(item) + "\n")



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run model predictions for final submission.")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--lang", required=True, help="Language code to test")
    args = parser.parse_args()

    model_name = args.model
    test_lang = args.lang

    print("model: ",model_name)
    print("test_lang: ",test_lang)
    
    get_predictions(test_lang, model_name)  # get model predicitons
    

