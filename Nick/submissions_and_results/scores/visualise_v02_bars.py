import re
import matplotlib.pyplot as plt
import numpy as np

def extract_numbers(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regex pattern to extract language codes, model names, IoU, and correlation scores
    pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
    
    data = {}
    for lang, model, iou, corr in pattern:
        if lang not in data:
            data[lang] = {}
        if model not in data[lang]:
            data[lang][model] = {'IoU': [], 'Correlation': []}
        data[lang][model]['IoU'].append(float(iou))
        data[lang][model]['Correlation'].append(float(corr))
    
    return data

def visualize_data(data, filename):
    plt.figure(figsize=(12, 6))
    
    models = list(set(model for lang in data for model in data[lang]))
    languages = sorted(set(lang for lang in data))

    x = np.arange(len(languages))  # X positions for languages
    width = 0.15  # Width of each bar

    for i, model in enumerate(models):
        iou_means = [data[lang][model]['IoU'][0] if model in data[lang] else 0 for lang in languages]
        plt.bar(x + i * width, iou_means, width=width, label=model)
    
    plt.xlabel('Languages')
    plt.ylabel('IoU Score')
    plt.title(f'IoU Scores by Language and Model')
    plt.xticks(x + (width * len(models) / 2), languages, rotation=45)
    plt.ylim(0.0, 0.8) 
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"pics/{filename}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
  

# Example usage:
#filename = "bert-base-multilingual-cased.txt"  # Change to your actual file name

filenames = {"xlm-roberta-large-10ep-M.txt", 
             "xlm-roberta-base.txt", 
             "umt5-small-10ep-M.txt", 
             "umt5-base-10ep-M.txt", 
             "bert-base-multilingual-cased.txt",
             "xlm-roberta-large-10ep-M_with_separator.txt"}

for filename in filenames:
    data = extract_numbers(filename)
    visualize_data(data, filename)
