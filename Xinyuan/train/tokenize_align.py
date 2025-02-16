import torch

# %%
#  tokenize input text before feeding into model
def tokenize_input(data, tokenizer):
    tokenized_inputs = tokenizer(
        data["model_output_text"],  
        add_special_tokens = True,   # add [CLS] and [SEP]
        max_length = 128,
        padding = "max_length",
        truncation = True,
        return_offsets_mapping = True,
        # return_token_type_ids = True, # used for bert-base models only
        return_tensors='pt'          # return tokenized inputs as pytorch tensors -> allows for using gpu 
    )
    
    return tokenized_inputs
    

# %%
def tokenize_n_align(data, tokenizer):
    
    # 1. tokenize input text
    tokenized_inputs = tokenize_input(data, tokenizer)
    
    # 2. manually add & align labels to the tokenized input:
    # extract labels from hard_labels & convert to "I"/"O" labels for each TOKEN
    labels = []
    
    for i in range(len(tokenized_inputs["input_ids"])):
        offset_mapping = tokenized_inputs['offset_mapping'][i]
        hard_labels = data["hard_labels"][i]
        attention_mask = tokenized_inputs['attention_mask'][i]
        
        # create a label array initialized to -100 (=="ignore this token")
        # then re-lable those whin the attention_mask as 0
        label_sequence = [-100] * len(offset_mapping)
        masked_label_sequence = [0 if m == 1 else l for l, m in zip(label_sequence, attention_mask)]

        if hard_labels:  # check if hard_labels is empty
            for idx, start_end in enumerate(offset_mapping):
                start = start_end[0]
                end = start_end[1]
                for (label_start, label_end) in hard_labels:
                    if start >= label_start and end <= label_end:
                        masked_label_sequence[idx] = 1


        labels.append(masked_label_sequence)

    # add labels to the tokenized_inputs in the form of a pytorch tensor
    tokenized_inputs["labels"] = torch.tensor(labels)
    
    return tokenized_inputs


# %%
# the code below are only for testing purpose
if __name__ == "__main__": 
    from transformers import AutoTokenizer
    from datasets import Dataset

    # a toy dataset: to test if the tokenize_n_align function works
    toy_data = [{'model_output_text': 'Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.', 
                'hard_labels': [[25, 31], [45, 49], [69, 83]], 
                'id': 'val-en-1', 
                'model_input': 'What did Petra van Staveren win a gold medal for?'}]
    
    toy_data = Dataset.from_dict({k: [d[k] for d in toy_data] for k in toy_data[0]})

    # load tokenizer
    model_name = "FacebookAI/xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)


#-----------------------------------------------
    tokenized_data = tokenize_n_align(toy_data, tokenizer)
    print("data after tokenization: ", tokenized_data)


