import argparse
import json
import csv
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from scorer import main as get_scores, load_jsonl_file_to_records as load_jsonl         # helper function 1: calculate IoU & Cor scores

# ask a model to generate predicted lables from raw data
def get_predictions(test_lang):
    model_name = "51la5/roberta-large-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=128, truncation=True, padding = "max_length")
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0)
    
    # load test data (the validation set) & write predictions in seperate jsonl file
    data_for_pred =  load_jsonl(f"./testing_data/test_set_labeled/mushroom.{test_lang}-tst.v1.jsonl")
    
    new_prediction_addr = f"./results/ner_testSet_{test_lang}.jsonl"
    with open(new_prediction_addr, "w") as f:
        for item in data_for_pred:
            current_line = {"id":item["id"], 
                    "model_input":item["model_input"],
                    "model_output_text":item["model_output_text"],
                    "soft_labels": []}
            
            label_predictions = ner(item['model_output_text'])
                
            if label_predictions:  # check if there's any span detected
                for l in label_predictions:
                    current_span = {"start": l["start"], 
                                "prob": round(float(l["score"]),3),  # convert score value (np.float32) into type float & round up to 3 decimals
                                "end": l["end"]}
                    
                    current_line["soft_labels"].append(current_span)
                    
            f.write(json.dumps(current_line) + "\n")
    
    # return the prediction file addr
    return new_prediction_addr


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run model predictions and scoring.")
    parser.add_argument("--lang", required=True, help="Language code to test")
    args = parser.parse_args()

    test_lang = args.lang
    
    
    # # get predicitons from the model's output jsonl file
    predictions = load_jsonl(get_predictions(test_lang)) # model predictions
    

    # get scores
    ref_dicts = load_jsonl(f"./testing_data/test_set_labeled/mushroom.{test_lang}-tst.v1.jsonl") # the gold standard file 
    output_file = f"./scores/ner-{test_lang}-score.txt"   # scores will be stored to this address
    ious, cors = get_scores(ref_dicts, predictions, output_file)

    # write scores to a csv file
    file_path = './scores/ner_scores_test.csv' 

    model_scores = {"model_name": "51la5/roberta-large-NER",
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
