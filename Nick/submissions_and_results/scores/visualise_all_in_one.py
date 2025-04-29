import re
import matplotlib.pyplot as plt
import numpy as np

styles = plt.style.available

plt.style.use('seaborn-v0_8-colorblind')

for item in styles:
    print(item)

def extract_numbers(filenames):
    data = {}

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Regex pattern to extract language codes, model names, IoU, and correlation scores
        #pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]+)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
        #pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]*ALLvalset[^<]*)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
        pattern = re.findall(r'<td>([A-Z]{2})</td>\s*<td>([^<]*\b(?:ALLvalset|monoling,\s*EN\s*\+\s*enVal)\b[^<]*)</td>\s*<td>([0-9.]+)</td>\s*<td>([0-9.]+)</td>', content)
        

        for lang, model, iou, corr in pattern:
            if lang not in data:
                data[lang] = {}
            if model not in data[lang]:
                data[lang][model] = {'IoU': [], 'Correlation': []}
            data[lang][model]['IoU'].append(float(iou))
            data[lang][model]['Correlation'].append(float(corr))

    return data

def visualize_data(data):
    plt.figure(figsize=(13, 9))

    # Define the order categories
    best_models = {"Best SemEval Score"}
    baseline_models = {"SemEval Baseline (mark all)", "NER Baseline"}
    monolingual_models = {
        "google-bert/bert-base-cased",
        "microsoft/deberta-v3-base",
        "deepset/roberta-base-squad2"
    }
    multilingual_models = {
        "google-bert/bert-base-multilingual-cased",
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "google/umt5-base",
        "google/umt5-small",
        "FacebookAI/xlm-roberta-large, trained with Question + <@@> + Answer"
    }

    # Sort: Monolingual models first, then multilingual
    models = sorted(
        set(model for lang in data for model in data[lang]), 
        key=lambda x: (
            x not in best_models and x not in baseline_models,  # Best first, then baselines
            x in monolingual_models,  # Multilingual before monolingual
            x  # Default alphabetical sorting within groups
        )
    )
    #models = sorted(set(model for lang in data for model in data[lang]))  # Sort models
    all_languages = sorted(set(lang for lang in data))  # Get all languages in sorted order

    # Get the default Matplotlib color cycle
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Assign a color to each model
    # Default matlpotlib color cycle
    #model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    #More unique colors
    cmap = plt.cm.get_cmap("tab20", len(models))  # Assigns unique colors
    model_colors = {model: cmap(i) for i, model in enumerate(models)}    

    for model in models:
        iou_scores = []

        for lang in all_languages:
            if model in data.get(lang, {}):  
                iou_scores.append(data[lang][model]['IoU'][0])  # First (only) IoU score
                #iou_scores.append(data[lang][model]['Correlation'][0])  # Corr score
            else:
                iou_scores.append(np.nan)  # Use NaN for missing values (ignored in plotting)

        # Choose marker based on whether the model is monolingual or not
        marker_style = 's' if model in monolingual_models else 'o'  # Squares for monolingual, circles for multilingual

        plt.plot(all_languages, iou_scores, marker=marker_style, label=model, linestyle=':', color=model_colors[model], markersize=8)

    #plt.xlabel('Languages', fontsize='large')
    #plt.ylabel('IoU Score', fontsize=12)
    #plt.ylabel('Correlation Score')
    #plt.title('IoU Scores for Models Trained on Synthetic Data and Validation Sets', fontsize='xx-large')
    #plt.title('Correlation Scores by Language and Model')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(np.arange(0.125, 0.775, 0.025))
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Create legend with monolingual models first
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort legend handles according to category
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles_labels = sorted(
        zip(handles, labels),
        key=lambda x: (
            x[1] not in best_models and x[1] not in baseline_models,  # Best first, then baselines
            x[1] not in multilingual_models,  # Multilingual before monolingual
            x[1]  # Default alphabetical sorting within groups
        )
    )
    handles, labels = zip(*sorted_handles_labels)
    
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='medium')


    plt.tight_layout()

    # Print colors with model names
    for model, color in model_colors.items():
        print(f"{model}: {color}")

    plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")    
    plt.show()


# Example usage:
filenames = [
    "bert-base-multilingual-cased.txt",
    "xlm-roberta-base.txt",
    "xlm-roberta-large-10ep-M.txt",
    "umt5-base-10ep-M.txt",
    "umt5-small-10ep-M.txt",
    "Monoling_scores.txt",
    "xlm-roberta-large-10ep-M_with_separator.txt"
    # "bert-base-cased.txt",
    # "google_flan-t5-base.txt",
    # "microsoft_deberta-v3-base.txt",
    # "deepset_roberta-base-squad2.txt"
]

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

NER = {
    "AR": {"IoU": 0.25926758, "Correlation": 0.18480393},
    "DE": {"IoU": 0.27610984, "Correlation": 0.21869591},
    "EN": {"IoU": 0.26665087, "Correlation": 0.2226832},
    "ES": {"IoU": 0.21969407, "Correlation": 0.09701412},
    "FI": {"IoU": 0.20807314, "Correlation": 0.08292431},
    "FR": {"IoU": 0.17053855, "Correlation": 0.04828228},
    "HI": {"IoU": 0.17997527, "Correlation": 0.10226301},
    "IT": {"IoU": 0.27860097, "Correlation": 0.24279016},
    "ZH": {"IoU": 0.2303202, "Correlation": 0.17799222},
    "CS": {"IoU": 0.19221815, "Correlation": 0.13039754},
    "CA": {"IoU": 0.25712428, "Correlation": 0.16710121},
    "EU": {"IoU": 0.20627569, "Correlation": 0.09996681},
    "FA": {"IoU": 0.26505734, "Correlation": 0.26262703}
}


data = extract_numbers(filenames)
#print(data["EN"])

for lang, values in best_teams.items():
    if lang not in data:
        data[lang] = {}
    data[lang]["Best SemEval Score"] = {"IoU": [values["IoU"]], "Correlation": [values["Correlation"]]}

for lang, values in baseline.items():
    if lang not in data:
        data[lang] = {}
    data[lang]["SemEval Baseline (mark all)"] = {"IoU": [values["IoU"]], "Correlation": [values["Correlation"]]}

#51la5/roberta-large-NER
for lang, values in NER.items():
    if lang not in data:
        data[lang] = {}
    data[lang]["NER Baseline"] = {"IoU": [values["IoU"]], "Correlation": [values["Correlation"]]}


rename_mapping = {
    "old_model_name": "new_model_name", 
    
    "bert-base-multilingual-cased, Multiling, ALLvalset": "google-bert/bert-base-multilingual-cased",
    "xlm-roberta-base, Multiling, ALLvalset": "FacebookAI/xlm-roberta-base",
    "xlm-roberta-large-10ep-M, Multiling, ALLvalset": "FacebookAI/xlm-roberta-large",
    "umt5-base-10ep-M, Multiling, ALLvalset": "google/umt5-base",
    "umt5-small-10ep-M, Multiling, ALLvalset": "google/umt5-small",
    "xlm-roberta-large-10ep-M, Multiling, ALLvalset, with Separator": "FacebookAI/xlm-roberta-large, trained with Question + <@@> + Answer",
    
    #"bert-base-cased, English, monoling",
    "bert-base-cased, English, monoling, EN + enVal": "google-bert/bert-base-cased",
    #"google/flan-t5-base, monoling, EN + enVal",
    #"microsoft/deberta-v3-base, monoling, EN",
    "microsoft/deberta-v3-base, monoling, EN + enVal" : "microsoft/deberta-v3-base",
    #"deepset/roberta-base-squad2, monoling, EN",
    "deepset/roberta-base-squad2, monoling, EN + enVal": "deepset/roberta-base-squad2"


}

#print("Before renaming:")
#for lang in data:
#    print(f"{lang}: {list(data[lang].keys())}")

for lang in data:
    updated_entries = {}
    for old_key, new_key in rename_mapping.items():
        if old_key in data[lang]:
            updated_entries[new_key] = data[lang].pop(old_key)  # Move data to new key
    data[lang].update(updated_entries)  # Apply the updates

#print(data)


visualize_data(data)

# Identify multilingual models (anything not in monolingual_models set)
multilingual_models = {
    "google-bert/bert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "google/umt5-base",
    "google/umt5-small",
    "FacebookAI/xlm-roberta-large, trained with Question + <@@> + Answer"
}

# Store average IoU for each multilingual model
multilingual_avg_iou = {}

for model in multilingual_models:
    iou_scores = []
    
    for lang in data:
        if model in data[lang]:
            iou_scores.append(data[lang][model]['IoU'][0])  # Take first IoU score
    
    if iou_scores:  # Avoid division by zero
        multilingual_avg_iou[model] = sum(iou_scores) / len(iou_scores)  # Compute average

# Print results
for model, avg_iou in multilingual_avg_iou.items():
    print(f"{model}: {avg_iou:.3f}")


