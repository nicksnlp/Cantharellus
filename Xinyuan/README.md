# The code here fine-tunes a pre-trained LLM for hallucination detection.
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
