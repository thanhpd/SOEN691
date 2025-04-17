import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bert_score_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['BERTScore F1'] = pd.to_numeric(df['BERTScore F1'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='BERTScore F1', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of BERTScore F1 by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("BERTScore F1")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

def plot_bmoses_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['B-Moses'] = pd.to_numeric(df['B-Moses'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='B-Moses', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of B-Moses by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("B-Moses")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

def plot_bnltk_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['B-NLTK'] = pd.to_numeric(df['B-NLTK'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='B-NLTK', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of B-NLTK by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("B-NLTK")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

def plot_bnorm_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['B-Norm'] = pd.to_numeric(df['B-Norm'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='B-Norm', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of B-Norm by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("B-Norm")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

def plot_rougel_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['Rouge-L'] = pd.to_numeric(df['Rouge-L'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='Rouge-L', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of Rouge-L by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("Rouge-L")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

def plot_meteor_boxplot(csv_file, lang):
    df = pd.read_csv(csv_file)
    df['METEOR'] = pd.to_numeric(df['METEOR'], errors='coerce')

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ModelName', y='METEOR', data=df)
    plt.xticks(rotation=45)
    plt.title(f"Box Plot of METEOR by Model ({lang})")
    plt.xlabel("Model Name")
    plt.ylabel("METEOR")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()

plot_bmoses_boxplot("./replication-lines/output_lines_java.csv", "java")
# plot_bnorm_boxplot("./replication-lines/output_lines_java.csv", "java")
# plot_bnltk_boxplot("./replication-lines/output_lines_java.csv", "java")
# plot_rougel_boxplot("./replication-lines/output_lines_java.csv", "java")
# plot_meteor_boxplot("./replication-lines/output_lines_java.csv", "java")
# plot_bert_score_boxplot("./replication-lines-bert/output_lines_java.csv", "java")

# plot_bert_score_boxplot("/content/data/output_bert_per_line_java.csv", "java")
# plot_bert_score_boxplot("/content/data/output_bert_per_line_js.csv", "js")
# plot_bert_score_boxplot("/content/data/output_bert_per_line_py.csv", "py")

# plot_bmoses_boxplot("./output-lines/output_lines-moses_java.csv", "java")
# plot_bmoses_boxplot("./output-lines/output_lines-moses_js.csv", "js")
# plot_bmoses_boxplot("./output-lines/output_lines-moses_py.csv", "py")

# plot_bnltk_boxplot("/content/data/output_lines-nltk_java.csv", "java")
# plot_bnltk_boxplot("/content/data/output_lines-nltk_js.csv", "js")
# plot_bnltk_boxplot("/content/data/output_lines-nltk_py.csv", "py")

# plot_bnorm_boxplot("/content/data/output_lines-norm_java.csv", "java")
# plot_bnorm_boxplot("/content/data/output_lines-norm_js.csv", "js")
# plot_bnorm_boxplot("/content/data/output_lines-norm_py.csv", "py")
