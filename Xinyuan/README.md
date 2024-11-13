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

# Helper function 1: json2dataset.py 
read training data from a jsonl file and convert into DatasetDict object (contains a training and a validation set).

# Helper function 2: tokenize_align.py 
preprocess the data by:
- tokenizing the input question & output answer
- aligning lables to tokens
