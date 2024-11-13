# The main script (fine_tuned_model_v0.3.py) fine-tunes a pre-trained LLM for hallucination detection.
Pretrained model:
- bert-cased

Training data elements:
- question text
- answer text
- hard labels

Output elements:
- hard labels:
  * 1.start of hallucination (character index)
  * 2.probability
  * 3.end of hallucination
- Hallucinated text

# json2dataset.py contains a helper function that read training data from a jsonl file and convert into DatasetDict object (contains a training and a validation set).

# tokenize_align.py contains a helper function that preprocess the data by:
- tokenizing the input question & output answer
- aligning lables to tokens
