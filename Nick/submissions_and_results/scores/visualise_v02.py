import re
import matplotlib.pyplot as plt
import numpy as np

best_teams = {
    "AR": {"name": "MSA", "IoU": 0.669976, "Correlation": 0.648751},
    "CA": {"name": "UCSC", "IoU": 0.721111, "Correlation": 0.777887},
    "CS": {"name": "AILSNTUA", "IoU": 0.542931, "Correlation": 0.55604},
    "DE": {"name": "UCSC", "IoU": 0.623612, "Correlation": 0.650701},
    "EN": {"name": "iai_MSU", "IoU": 0.650899, "Correlation": 0.629443},
    "ES": {"name": "ATLANTIS", "IoU": 0.531131, "Correlation": 0.0131579},
    "EU": {"name": "MSA", "IoU": 0.612919, "Correlation": 0.620171},
    "FA": {"name": "AILSNTUA", "IoU": 0.710956, "Correlation": 0.69889},
    "FI": {"name": "UCSC", "IoU": 0.648264, "Correlation": 0.649756},
    "FR": {"name": "Deloitte", "IoU": 0.646867, "Correlation": 0.618707},
    "HI": {"name": "ccnu", "IoU": 0.746571, "Correlation": 0.78466},
    "IT": {"name": "UCSC", "IoU": 0.787248, "Correlation": 0.787321},
    "ZH": {"name": "YNU-HPCC", "IoU": 0.553971, "Correlation": 0.351808}
}

baseline = {
    "AR": {"IoU": 0.361354, "Correlation": 0.00666667},
    "CA": {"IoU": 0.242314, "Correlation": 0.06},
    "CS": {"IoU": 0.263164, "Correlation": 0.1},
    "DE": {"IoU": 0.345082, "Correlation": 0.0133333},
    "EN": {"IoU": 0.348926, "Correlation": 0},
    "ES": {"IoU": 0.185334, "Correlation": 0.0131579},
    "EU": {"IoU": 0.36709, "Correlation": 0},
    "FA": {"IoU": 0.202808, "Correlation": 0.01},
    "FI": {"IoU": 0.4857, "Correlation": 0},
    "FR": {"IoU": 0.454341, "Correlation": 0},
    "HI": {"IoU": 0.271096, "Correlation": 0},
    "IT": {"IoU": 0.282615, "Correlation": 0},
    "ZH": {"IoU": 0.477155, "Correlation": 0}
}


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
    
    for model in set(model for lang in data for model in data[lang]):
        languages = []
        iou_means = []
        
        for lang in data:
            if model in data[lang]:
                languages.append(lang)
                iou_means.append(np.mean(data[lang][model]['IoU']))
        
        plt.plot(languages, iou_means, marker='o', label=model)
    
    plt.xlabel('Languages')
    plt.ylabel('IoU Score')
    plt.title(f'IoU Scores by Language and Model')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"pics/{filename}.pdf", format="pdf", bbox_inches="tight")    

# Example usage:
#filename = "bert-base-multilingual-cased.txt"  # Change to your actual file name

filenames = {"xlm-roberta-large-10ep-M.txt", 
             "xlm-roberta-base.txt", 
             "umt5-small-10ep-M.txt", 
             "umt5-base-10ep-M.txt", 
             "bert-base-multilingual-cased.txt"}

for filename in filenames:
    data = extract_numbers(filename)
    visualize_data(data, filename)
