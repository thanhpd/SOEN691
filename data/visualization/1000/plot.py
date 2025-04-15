import pandas as pd
import matplotlib.pyplot as plt


def plot_barh(metric, lang):
    """
        X Axis = metric

    """
    df = pd.read_csv("output.csv")
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    data = {
        "codellama_7b": [0,0,0],
        "gemma2_9b": [0,0,0],
        "llama2_70b": [0,0,0],
        "llama2_7b": [0,0,0],
        "llama3.2_3b": [0,0,0],
    }
    for _, row in df.iterrows():
        if row["Lang"] == lang:
            model = row["Model"]
            if row["Temp"] == 0:
                data[model][0] = row[metric]
            elif row["Temp"] == 0.5:
                data[model][1] = row[metric]
            elif row["Temp"] == 1.0:
                data[model][2] = row[metric]

    metric_labels = ["0.0", "0.5", "1.0"]
    df = pd.DataFrame.from_dict(data, orient='index', columns=metric_labels)

    # Plot a horizontal bar chart for each metric
    df.plot(kind='barh', figsize=(10, 6))
    plt.title(f"{metric} values across temperatures for {lang}")
    plt.xlabel("Score")
    plt.ylabel("Model Name")
    plt.legend(title="Metric")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


metrics = ["B-Moses","B-Norm","B-NLTK","Rouge-L","METEOR"]
languages = ["java", "js", "py"]
for m in metrics:
    for l in languages:
        plot_barh(m, l)