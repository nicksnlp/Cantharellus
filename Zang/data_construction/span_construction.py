import json

current_id = 1  

with open('eu-sim-val-raw.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

with open('eu-sim-val.jsonl', 'a', encoding='utf-8') as output_file:

    for entry in data:
        model_input = entry["model_input"]
        model_output_text = entry["model_output_text"]
        spans = []


        for word in entry["hallucinated_words"]:
            start_index = 0
            while True:
                start_index = model_output_text.find(word, start_index)  
                if start_index == -1:
                    break
                end_index = start_index + len(word)
                spans.append([start_index, end_index])
                start_index = end_index 

        result = {
            "id": current_id, 
            "created_with": "GPT4o",
            "lang": "eu",  
            "model_input": model_input,
            "model_output_text": model_output_text,
            "hard_labels": [[start_index, end_index] for start_index, end_index in spans]
        }

        
        current_id += 1


        json.dump(result, output_file, ensure_ascii=False)
        output_file.write("\n")
