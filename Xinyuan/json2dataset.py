#!/usr/bin/env python
# coding: utf-8
# %%

# # Helper function:  read data from a jsonl file & converts to a dataset object
# --------------------------------------------------------
# 
#  - split the validation set for testing the model before we have a large labeled training set
# (there are 50 objects in val set)

# %%
# pip install datasets


# %%
import json
from datasets import Dataset, DatasetDict


# %%

# read data from jsonl file (output == 2 lists of dicts)
# and extract only "model_input", "model_output_text" and "hard_labels" from it
def file_reader(file_addr):
    # store val set objects
    data = [] #list of dicts(each sict = 1 json object)
    
    # a set of keys to be extracted for training model
    target_keys = {"id","model_input", "model_output_text", "hard_labels"}

    # Open the file and read each line
    with open(file_addr, 'r', encoding='utf-8') as file:
        for line in file:
            # access each object & extract only needed key-val pairs (input, output, hard labels)
            json_obj = json.loads(line.strip())  # strip(): removes extra whitespace or newlines
            target_data = {key: json_obj[key] for key in target_keys if key in json_obj}
            
            # remove unecessary whitespaces/newlines from input/output sentences
            stripped_target_data = {k: (v.strip() if isinstance(v, str) else v) for k, v in target_data.items()}
            data.append(stripped_target_data)
    
    return data


# split the dataset into training & validation set based on a x:y ratio (x = training set, y = validation set)
def data_splitter(data, x, y):
    # split val set based on a 9: 1 ratio
    split_idx = (len(data)*x)//(x + y)
    data_train, data_val = data[:split_idx], data[split_idx:]
    
    # lists of dicts --> Dataset objects
    train_dataset = Dataset.from_dict({k: [d[k] for d in data_train] for k in data_train[0]})
    val_dataset = Dataset.from_dict({k: [d[k] for d in data_val] for k in data_val[0]})
    
    # wrap train & val set into a DatasetDict 
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset})

    return dataset

# %%

# for testing purpose (the function above)
if __name__ == "__main__":    
    file_addr = "./mushroom.en-val.v2.jsonl"
    
    data = file_reader(file_addr)
    dataset = (data, 9, 1)

    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    print(val_data)

