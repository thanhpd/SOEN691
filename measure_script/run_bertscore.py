import os
import subprocess
import csv
import shutil
import re

# Define paths
GEN_FOLDER = "processed_msg"
REF_FILE = os.path.join(GEN_FOLDER, "label.msg")
OUTPUT_FILE = "bertscore_output.csv"
BERT_SCORE_REPO = "https://github.com/Tiiiger/bert_score"
BERT_SCORE_DIR = "bert_score"

# Setup Environment
print("Setting up the environment...")

# Clone and install BERTScore if not already installed
if not os.path.exists(BERT_SCORE_DIR):
    print("Cloning BERTScore repository...")
    subprocess.run(["git", "clone", BERT_SCORE_REPO], check=True)

print("Installing BERTScore...")
subprocess.run(["pip", "install", "-e", BERT_SCORE_DIR], check=True)

# Check if the input folder exists
if not os.path.isdir(GEN_FOLDER):
    print(f"Error: Folder '{GEN_FOLDER}' not found!")
    exit(1)

DEST_FOLDER = os.path.join("bert_score", "processed_msg")

# Ensure destination folder exists
os.makedirs(DEST_FOLDER, exist_ok=True)

# Copy the folder
try:
    shutil.copytree(GEN_FOLDER, DEST_FOLDER, dirs_exist_ok=True)
    print(f"Successfully copied '{GEN_FOLDER}' to '{DEST_FOLDER}'")
except Exception as e:
    print(f"Error copying folder: {e}")

os.chdir(BERT_SCORE_DIR)

# Initialize CSV file with headers
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Filename", "Precision", "Recall", "F1"])

# Process each file in the folder
for filename in os.listdir(GEN_FOLDER):
    input_file = os.path.join(GEN_FOLDER, filename)

    # Skip the reference file
    if filename == "label.msg":
        continue

    print(f"Processing {filename}...")

    # Ensure the file exists
    if not os.path.isfile(input_file):
        print(f"Error: Candidate file '{filename}' does not exist! Skipping...")
        continue

    try:
        # Run BERTScore and capture output
        result = subprocess.run(
            ["bert-score", "-r", REF_FILE, "-c", input_file, "--lang", "en"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        print("Raw BERTScore output:", output)

        # Extract precision, recall, and F1-score using regex
        match = re.search(r"P:\s*([\d.]+)\s*R:\s*([\d.]+)\s*F1:\s*([\d.]+)", output)
        if match:
            P, R, F1 = match.groups()
        else:
            print(f"Error extracting scores for {filename}. Skipping...")
            continue

        # Append results to CSV
        with open(OUTPUT_FILE, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([filename, P, R, F1])

    except subprocess.CalledProcessError as e:
        print(f"Error processing {filename}: {e}")

print(f"Processing completed. Results saved in {OUTPUT_FILE}.")
input("Press any key to exit...")
