#!/usr/bin/env python
# coding: utf-8
# %%

# # Helper function: tokenize & align lables
# --------------------------------------------------------
# Tokenize Qestion-Answer pairs and these tokens with corresponding lables (I or O) based on the given sratring/ending position of hallucination.
# --------------------------------------------------------
# 
# function input = dictionary {'model_input', 'model_output_text' , 'soft_labels'}
# *   'model_input': question text
# *   'model_output_text': answer text
# *   'hard_labels': a list of lists, marking start & end of hallucination
# --------------------------------------------------------
# 
# function output = dictionary {'input_ids', 'attention_mask', 'token_type_ids', 'labels'}
# *   'input_ids': token ID
# *   'attention_mask': marks padding
# *   'token_type_ids': ignore question for labeling (0 or 1)
# *   'labels': hallucination/not (1 or 0)

# %%
def tokenize_n_align(examples, tokenizer, label2id):
    inputs = tokenizer(
        examples["model_input"],
        examples["model_output_text"],
        truncation=True,
        padding="max_length", # default = 512 tokens
        return_offsets_mapping=True
    )

    labels = []
    for i in range(len(inputs["input_ids"])):
        # Get offsets for the current tokenization
        offset_mapping = inputs['offset_mapping'][i]

        # Create a label array initialized to 'O'
        label_sequence = ['O'] * len(offset_mapping)

        # Iterate over each span in hard_labels
        for start_end in examples["hard_labels"][i]:
            start_char, end_char = start_end

            for j, (start, end) in enumerate(offset_mapping):
                # Check if the token spans any character in the answer span
                if start_char <= start < end_char or start < end_char <= end:
                    label_sequence[j] = 'I'

        labels.append([label2id[label] for label in label_sequence])

    # Add labels to the inputs
    inputs["labels"] = labels
    
    return inputs

