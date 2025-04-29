## This is a preliminary code to generate questions based on Wikipedia article.
## To do:  generate also answers.

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_questions(text):
    model_name = "google/t5-small-ssm"  # You can experiment with different models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=)
    outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

    questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return questions

# Example usage:
text = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars, named after the engineer Gustave Eiffel. It was built for the 1889 World's Fair, when France celebrated the centenary of the French Revolution."

questions = generate_questions(text)
for question in questions:
    print(question)