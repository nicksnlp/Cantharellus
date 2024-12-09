# The main script (fine_tuned_model_v0.5.py) fine-tunes a pre-trained LLM for hallucination detection.
Pretrained model:
- bert-cased

Training data elements:
- question text
- answer text
- hard labels

Output elements:
- soft labels:
  * 1.start of hallucination (character index)
  * 2.probability
  * 3.end of hallucination

# Helper function 1: json2dataset.py 
read training data from a jsonl file and convert into DatasetDict object (contains a training and a validation set).

# Helper function 2: tokenize_align.py 
preprocess the data by:
- tokenizing the input question & output answer
- aligning lables to tokens

# Helper function 3: test.py
- feed test set to fine-tuned models (from the "models" directory)
- store outputs to a JSONL file
  (NOTE: this helper function only stores the outputs, but doesn't score for the model's performance)

# train.sh
Slurm script for training model on Puhti
