# Cantharellus
A research project in language technology on hallucination detection in connection with SemEval-2025
**Task 3: Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes**.

This repository contains scripts for training & inference with various LLM models, as well as scripts for generating synthetic data used during training. It also contains some test scripts & notebooks used on some intermediate steps of the research.

Also, the system description paper is available here in latex and pdf formats.

A sample model (with base model `xlm-roberta-large`), fine-tuned within this project can be found at Hugging Face. This model was fine-tuned with synthetic data (26.3K data points, 10 epochs), and further fine-tuned human-labeled multilingual data (650 data points, 10 epochs), with combination of `"model_input"` + " <@@> " + `"model_output_text"` as training data.

https://huggingface.co/nicksnlp/xlm-roberta-large-cantharellus
