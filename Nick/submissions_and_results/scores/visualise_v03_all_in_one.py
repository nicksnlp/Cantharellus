import re
import matplotlib.pyplot as plt

def extract_numbers(filenames):
    data = {}

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regex pattern to extract language codes, model names, IoU, and correlation scores
        pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)

        for lang, model, iou, corr in pattern:
            if lang not in data:
                data[lang] = {}
            if model not in data[lang]:
                data[lang][model] = {'IoU': [], 'Correlation': []}
            data[lang][model]['IoU'].append(float(iou))
            data[lang][model]['Correlation'].append(float(corr))

    return data

def visualize_data(data):
    plt.figure(figsize=(12, 6))

    # Get all unique models and sort them alphabetically
    models = sorted(set(model for lang in data for model in data[lang]))

    for model in models:  # Iterate through sorted models
        languages = []
        iou_scores = []

        for lang in data:
            if model in data[lang]:
                languages.append(lang)
                iou_scores.extend(data[lang][model]['IoU'])  # Use all values directly

        plt.plot(languages, iou_scores, marker='o', label=model)

    plt.xlabel('Languages')
    plt.ylabel('IoU Score')
    plt.title('IoU Scores by Language and Model')
    plt.xticks(rotation=45)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Moves legend outside the plot
    plt.tight_layout()
    plt.show()

# Example usage:
filenames = [
    "bert-base-multilingual-cased.txt",
    "xlm-roberta-base.txt",
    "xlm-roberta-large-10ep-M.txt",
    "umt5-base-10ep-M.txt",
    "umt5-small-10ep-M.txt",
    #"Monoling_scores.txt"
    # "bert-base-cased.txt",
    # "google_flan-t5-base.txt",
    # "microsoft_deberta-v3-base.txt",
    # "deepset_roberta-base-squad2.txt"
]

data = extract_numbers(filenames)
visualize_data(data)
