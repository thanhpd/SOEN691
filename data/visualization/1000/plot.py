import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_all = pd.read_csv("./output_1000.csv")

models = ["codellama_7b", "gemma2_9b", "llama2_70b", "llama2_7b", "llama3.2_3b"]
metrics = ["B-Moses", "B-Norm", "B-NLTK", "Rouge-L", "METEOR","BERTScore F1"]
languages = ["java", "js", "py"]
temperatures = ["0.0", "0.5", "1.0"]

def get_scores(df, model, metric, lang):
    scores = []
    for temp in temperatures:
        val = df[(df["Model"] == model) &
                 (df["Lang"] == lang) &
                 (df["Temp"] == float(temp))][metric]
        scores.append(val.values[0] if not val.empty else 0)
    return scores

# Set up grid: rows = models, cols = metrics
fig, axes = plt.subplots(nrows=len(models), ncols=len(metrics), figsize=(20, 15), sharey='col')

bar_width = 0.2
x = np.arange(len(languages))

for i, model in enumerate(models):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        all_scores = [get_scores(df_all, model, metric, lang) for lang in languages]
        all_scores = np.array(all_scores)  # shape: (languages, temperatures)

        for k, temp in enumerate(temperatures):
            ax.bar(x + k * bar_width, all_scores[:, k], width=bar_width, label=f"Temp {temp}" if i == 0 and j == 0 else "")

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(languages, rotation=30)
        ax.set_title(f"{metric}" if i == 0 else "")
        if j == 0:
            ax.set_ylabel(f"{model}", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

# Legend and layout
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)
fig.suptitle("Scores by Model, Metric, Language & Temperature", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
