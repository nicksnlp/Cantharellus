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

    "bert-base-cased, English, monoling": "google-bert/bert-base-cased, synthetic data only",
    "bert-base-cased, English, monoling, EN + enVal": "google-bert/bert-base-cased, synthetic data + single validation set",
    
    #"google/flan-t5-base, monoling, EN",
    #"google/flan-t5-base, monoling, EN + enVal",
    
    "microsoft/deberta-v3-base, monoling, EN": "microsoft/deberta-v3-base, synthetic data only",
    "microsoft/deberta-v3-base, monoling, EN + enVal": "microsoft/deberta-v3-base, synthetic data + single validation set",
    
    "deepset/roberta-base-squad2, monoling, EN": "deepset/roberta-base-squad2, synthetic data only",
    "deepset/roberta-base-squad2, monoling, EN + enVal": "deepset/roberta-base-squad2, synthetic data + single validation set"
    
}

def extract_numbers(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regex pattern to extract language codes, model names, IoU, and correlation scores
    pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
    
    data = {}
    results = []
    
    for lang, model, iou, corr in pattern:
        # Rename the model if it exists in rename_mapping
        
        # Rename model if needed
        model_renamed = rename_mapping.get(model, model)
        
        if lang not in data:
            data[lang] = {}
        if model_renamed not in data[lang]:
            data[lang][model_renamed] = {'IoU': [], 'Correlation': []}
        data[lang][model_renamed]['IoU'].append(float(iou))
        data[lang][model_renamed]['Correlation'].append(float(corr))

        # Also save the renamed model into results
        results.append((lang, model_renamed, float(iou), float(corr)))
    
    return data, results

def visualize_data(data, filename):
    plt.figure(figsize=(12, 6))
    
    all_models = list(set(model for lang in data for model in data[lang]))

    # Color assignment
    def get_color(model_name):
        if '+ all validation sets' in model_name.lower():
            return '#4169E1'  # Royal Blue
        elif '+ single validation set' in model_name.lower():
            return '#FF8C00'  # Dark Orange
        else:
            return '#3CB371'  # Medium Sea Green  # "plain" Multiling

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
all_models = set()  # To keep track of all unique models
all_languages = set()  # To keep track of all unique languages

# Example usage:
#filename = "bert-base-multilingual-cased.txt"  # Change to your actual file name

filenames = {"xlm-roberta-large-10ep-M.txt", 
             "xlm-roberta-base.txt", 
             "umt5-small-10ep-M.txt", 
             "umt5-base-10ep-M.txt", 
             "bert-base-multilingual-cased.txt",
             "xlm-roberta-large-10ep-M_with_separator.txt",
             "Monoling_scores.txt"
             }

for filename in filenames:
    data, extracted = extract_numbers(filename)
    visualize_data(data, filename)
    all_results.extend(extracted)
    for lang, model, iou, corr in extracted:
        all_models.add(model)
        all_languages.add(lang)

# Sort models and languages
all_models = sorted(all_models)  # Sort models lexicographically
all_languages = sorted(all_languages)  # Sort languages lexicographically

# Prepare the IoU scores table with models as rows and languages as columns
table_data = []

# Header (language columns)
header = ['Model'] + all_languages
table_data.append(header)

# Fill the table row by row for each model
for model in all_models:
    row = [model]
    for lang in all_languages:
        # Get the IoU score for this model and language (default to 0 if not available)
        iou_score = 0
        for entry in all_results:
            if entry[0] == lang and entry[1] == model:
                iou_score = entry[2]  # IoU score
                break
        row.append(iou_score)
    table_data.append(row)

# Save the IoU scores table to a CSV file
with open("pics2/00_iou_scores_table.csv", mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(table_data)  # Write all rows of the table

# Save to LaTeX table with landscape
with open("pics2/00_iou_scores_table.tex", mode='w', encoding='utf-8') as tex_file:
    tex_file.write("\\begin{landscape}\n")  # <--- Start landscape
    tex_file.write("\\begin{tabular}{l" + "c" * len(all_languages) + "}\n")
    tex_file.write("\\hline\n")
    
    # Header
    header_line = 'Model & ' + ' & '.join(all_languages) + " \\\\\n"
    tex_file.write(header_line)
    tex_file.write("\\hline\n")
    
    # Table rows
    for row in table_data[1:]:  # Skip header already written
        model = row[0]
        scores = row[1:]
        formatted_scores = ['{:.3f}'.format(score) if isinstance(score, (int, float)) else str(score) for score in scores]
        line = model + ' & ' + ' & '.join(formatted_scores) + " \\\\\n"
        tex_file.write(line)
    
    tex_file.write("\\hline\n")
    tex_file.write("\\end{tabular}\n")
    tex_file.write("\\end{landscape}\n")  # <--- End landscape
