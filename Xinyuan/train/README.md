# In this directory:

### The main script (fine_tuned_model_v0.5.py) fine-tunes a pre-trained LLM for hallucination detection.
Model input:
- LLM-generated text (answer part)
- hard labels

Model output:
- soft labels:
  * 1.start of hallucination (character index)
  * 2.probability
  * 3.end of hallucination

### Helper function: tokenize_align.py 
preprocess the data by:
- tokenizing the input question & output answer
- aligning lables to tokens

### train.sh
Slurm script for training model on Puhti
