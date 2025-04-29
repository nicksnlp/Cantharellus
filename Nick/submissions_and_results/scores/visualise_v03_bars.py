import re
import matplotlib.pyplot as plt
import numpy as np
import csv  # <--- for saving as CSV

# Define model renaming map
rename_mapping = {
    "bert-base-multilingual-cased, Multiling, ALLvalset": "google-bert/bert-base-multilingual-cased, synthetic data + all validation sets",
    ## NB 1_Valset with large case only here, other places 1_valset
    "bert-base-multilingual-cased, Multiling, 1_Valset": "google-bert/bert-base-multilingual-cased, synthetic data + single validation set",
    "bert-base-multilingual-cased, Multiling": "google-bert/bert-base-multilingual-cased, synthetic data only",
    
    "xlm-roberta-base, Multiling, ALLvalset": "FacebookAI/xlm-roberta-base, synthetic data + all validation sets",
    "xlm-roberta-base, Multiling, 1_valset": "FacebookAI/xlm-roberta-base, synthetic data + single validation set",
    "xlm-roberta-base, Multiling": "FacebookAI/xlm-roberta-base, synthetic data only",
    
    "xlm-roberta-large-10ep-M, Multiling, ALLvalset": "FacebookAI/xlm-roberta-large, synthetic data + all validation sets",
    "xlm-roberta-large-10ep-M, Multiling, 1_valset": "FacebookAI/xlm-roberta-large, synthetic data + single validation set",
    "xlm-roberta-large-10ep-M, Multiling": "FacebookAI/xlm-roberta-large, synthetic data only",

    "umt5-base-10ep-M, Multiling, ALLvalset": "google/umt5-base, synthetic data + all validation sets",
    "umt5-base-10ep-M, Multiling, 1_valset": "google/umt5-base, synthetic data + single validation set",
    "umt5-base-10ep-M, Multiling": "google/umt5-base, synthetic data only",
    
    "umt5-small-10ep-M, Multiling, ALLvalset": "google/umt5-small, synthetic data + all validation sets",
    "umt5-small-10ep-M, Multiling, 1_valset": "google/umt5-small, synthetic data + single validation set",
    "umt5-small-10ep-M, Multiling": "google/umt5-small, synthetic data only",
    
    "xlm-roberta-large-10ep-M, Multiling, ALLvalset, with Separator": "FacebookAI/xlm-roberta-large, synthetic data + all validation sets, trained with Question + <@@> + Answer",
    
}

def extract_numbers(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regex pattern to extract language codes, model names, IoU, and correlation scores
    pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
    
    data = {}
    for lang, model, iou, corr in pattern:
        # Rename the model if it exists in rename_mapping
        if model in rename_mapping:
            model = rename_mapping[model]
        
        if lang not in data:
            data[lang] = {}
        if model not in data[lang]:
            data[lang][model] = {'IoU': [], 'Correlation': []}
        data[lang][model]['IoU'].append(float(iou))
        data[lang][model]['Correlation'].append(float(corr))
    
    return data, pattern  # <--- Return also raw extracted data!

def visualize_data(data, filename):
    plt.figure(figsize=(12, 6))
    
    all_models = list(set(model for lang in data for model in data[lang]))

    # Color assignment
    def get_color(model_name):
        if '+ all validation sets' in model_name.lower():
            return 'blue'
        elif '+ single validation set' in model_name.lower():
            return 'orange'
        else:
            return 'green'  # "plain" Multiling

    # Sorting priority
    def sort_key(model_name):
        if '+ all validation sets' in model_name.lower():
            return 0  # First: Multiling + ALLvalset
        elif '+ single validation set' in model_name.lower():
            return 1  # Second: Multiling + 1_Valset
        else:
            return 2  # Third: plain Multiling

    models = sorted(all_models, key=sort_key)
    
    languages = sorted(set(lang for lang in data))
    x = np.arange(len(languages))  # X positions for languages
    width = 0.15  # Width of each bar

    
    for i, model in enumerate(models):
        iou_means = [data[lang][model]['IoU'][0] if model in data[lang] else 0 for lang in languages]
        color = get_color(model)
        plt.bar(x + i * width, iou_means, width=width, label=model, color=color)
    
    plt.xlabel('Languages')
    plt.ylabel('IoU Score')
    plt.title(f'IoU Scores by Language and Model')
    plt.xticks(x + (width * len(models) / 2), languages, rotation=45)
    plt.ylim(0.0, 0.8) 
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"pics2/{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
  

# ---------------------------------------------------------
# Collecting all data across files
all_results = []  # <--- To store (Language, Model, IoU, Correlation)

# Example usage:
#filename = "bert-base-multilingual-cased.txt"  # Change to your actual file name

filenames = {"xlm-roberta-large-10ep-M.txt", 
             "xlm-roberta-base.txt", 
             "umt5-small-10ep-M.txt", 
             "umt5-base-10ep-M.txt", 
             "bert-base-multilingual-cased.txt",
             "xlm-roberta-large-10ep-M_with_separator.txt"}

for filename in filenames:
    data, extracted = extract_numbers(filename)
    visualize_data(data, filename)
    all_results.extend(extracted)

# After processing all files, save all results into a CSV
with open("pics2/all_results.csv", mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Language", "Model", "IoU", "Correlation"])  # Header
    
    #writer.writerows(all_results)  # Write all rows, no rename mapping
    
    # Iterate through results and ensure models are renamed before saving
    for result in all_results:
        lang, model, iou, corr = result
        # Rename the model if it exists in rename_mapping
        if model in rename_mapping:
            model = rename_mapping[model]
        writer.writerow([lang, model, iou, corr])  # Write row with renamed model
