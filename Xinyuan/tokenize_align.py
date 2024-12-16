#!/usr/bin/env python
# coding: utf-8
# %%
# # This is a helper function: tokenize & align lables for each SINGLE sentence-pair
# --------------------------------------------------------
# Tokenize Qestion-Answer pairs and mark tokens with corresponding lables (I or O) based on the given sratr/end index of hallucination in hard_labels.
# --------------------------------------------------------
# 
# function input = dictionary {'model_input', 'model_output_text' , 'soft_labels'}

# *   'model_input': question text
# *   'model_output_text': answer text
# *   'hard_labels': a list of lists, marking start & end of hallucination
# --------------------------------------------------------
# 
# function output = dictionary {'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'}

# *   'input_ids': token IDs
# *   'token_type_ids': ignores model_input when labeling (0 = tokens in model_input, 1 = tokens in model_output_text (to be labeled))
# *   'attention_mask': marks padding (1 = original text, 0 = paddings)
# *   'offset_mapping': (start_idx, end_idx), keeps track of start/end index of each token
# *   'labels': hallucination/not (1 =  hallucination, 0 = non-hallucination)

# %%
from transformers import BertTokenizerFast
import torch


# %%
#  tokenize input text before feeding into model
def tokenize_input(data, tokenizer):
    tokenized_inputs = tokenizer(
        data["model_input"],         # text 1
        data["model_output_text"],   # text 2
        add_special_tokens = True,   # add [CLS] and [SEP]
        max_length = 256,
        padding = "max_length",
        truncation = True,
        return_offsets_mapping = True,
        return_token_type_ids = True,
        return_tensors='pt'          # return tokenized inputs as pytorch tensors -> allows for using gpu 
    )
    
    return tokenized_inputs
    


# %%
def tokenize_n_align(data, tokenizer, label2id):
    
    # 1. tokenize input text
    tokenized_inputs = tokenize_input(data, tokenizer)
    
    # 2. manually add & align labels to the tokenized input:
    # extract labels from hard_labels & convert to "I"/"O" labels for each TOKEN
    labels = []
    for i in range(len(tokenized_inputs["input_ids"])):
        print("length of tokenized_inputs: ", len(tokenized_inputs["input_ids"]))
        
        # get offsets for the currenat tokenization
        offset_mapping = tokenized_inputs['offset_mapping'][i]

        # create a label array initialized to 'O'
        label_sequence = ['O'] * len(offset_mapping)

        # iterate over each span in hard_labels
        hard_labels = data["hard_labels"][i]
        
        if hard_labels:  # check if hard_labels is empty
            for start_end in hard_labels:
                start_char, end_char = start_end
                
                # change token lable to 'I' if token is in model_output_text & fells in the range of hard labels
                for j, (start, end) in enumerate(offset_mapping):
                    if tokenized_inputs["token_type_ids"][i][j] == 1 and (start_char <= start < end_char or start < end_char <= end): 
                        label_sequence[j] = 'I'

        labels.append([label2id[label] for label in label_sequence])

    # add labels to the tokenized_inputs in the form of a pytorch tensor
    tokenized_inputs["labels"] = torch.tensor(labels)
    
    return tokenized_inputs



# %%
# the code below are only for testing purpose
if __name__ == "__main__": 
    from transformers import BertTokenizerFast
    from datasets import Dataset

    # a toy dataset: to test if the tokenize_n_align function works
    toy_data = [{'model_output_text': 'Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.', 
                'hard_labels': [[25, 31], [45, 49], [69, 83]], 
                'id': 'val-en-1', 
                'model_input': 'What did Petra van Staveren win a gold medal for?'}]
    toy_data = Dataset.from_dict({k: [d[k] for d in toy_data] for k in toy_data[0]})

    # load tokenizer
    model_name = "bert-base-cased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # define label2id
    labels = ["O", "I"] 
    label2id = {label: i for i, label in enumerate(labels)}
    print("label2id: ",label2id)

#-----------------------------------------------
    tokenized_data = tokenize_n_align(toy_data, tokenizer, label2id)
    print("data after tokenization: ", tokenized_data)

    len(tokenized_data["labels"][0])

