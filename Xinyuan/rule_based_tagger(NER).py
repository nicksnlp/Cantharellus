from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
from json2dataset import file_reader

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # a fine-tuned model for Named Entity Recognition
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# check all labels used by this model
labels = model.config.id2label
print(labels)

# +
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

file_addr = "./testing_data/mushroom.en-val.v2.jsonl"
val_set = file_reader(file_addr)
print(val_set[0])
# -

# iterate through all lines in the validation set
# and sotre predictions into a JSONL file
with open("./results/pred_rule_based_ner.jsonl", "w") as f:
    for item in val_set:
        current_line = {"id":item["id"], 
                  "model_input":item["model_input"],
                  "model_output_text":item["model_output_text"],
                  "soft_labels": []}
        
        # get label predictions & attach them to the JSONL file (in the value of 'soft_labels')
        label_predictions = ner(item['model_output_text'])
        if label_predictions:  # check if there's any span detected
            for l in label_predictions:
                current_span = {"start": l["start"], 
                               "prob": round(float(l["score"]),3),  # convert score value (np.float32) into type float & round up to 3 decimals
                               "end": l["end"]}
                
                current_line["soft_labels"].append(current_span)
                
        f.write(json.dumps(current_line) + "\n")
