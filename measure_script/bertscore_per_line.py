import os
import subprocess
import csv
import sys
import bert_score  # Importing bert_score

print("Setting up the environment...")
os.system("pip install bert-score")  # Install bert-score

# Define input folder, reference file, and output CSV file
GEN_FOLDER = "generated_msg"
OUTPUT_FILE = "output_bert_per_line.csv"

# Check if the folder exists
if not os.path.isdir(GEN_FOLDER):
    print(f"Error: Folder '{GEN_FOLDER}' not found!")
    sys.exit(1)

# Initialize CSV file with headers
headers = ["Foldername", "BERTScore Precision", "BERTScore Recall", "BERTScore F1"]
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

# Function to run commands and capture output
def run_and_capture(command):
    """Runs a command and captures its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")
        output = result.stdout.strip().replace("\n", " ").replace(",", " ")
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error message: {result.stderr}")
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize BERTScore scorer without rescaling with baseline
scorer = bert_score.BERTScorer(lang='en', rescale_with_baseline=False)

# Process each folder inside GEN_FOLDER
for root, dirs, files in os.walk(GEN_FOLDER):
    for dir_name in dirs:
        # Skip directories that don't contain the necessary msg files
        folder_path = os.path.join(root, dir_name)
        label_file = os.path.join(folder_path, "label.msg")
        
        if not os.path.isfile(label_file):
            continue  # Skip folders without a label.msg file

        # Process each generated msg file (excluding label.msg)
        for filename in os.listdir(folder_path):
            if filename == "label.msg":
                continue  # Skip reference file

            gen_file = os.path.join(folder_path, filename)
            print(f"Processing {filename} in {folder_path}...")

            # Read the generated and reference files (label.msg)
            with open(gen_file, "r", encoding="utf-8", errors="ignore") as gen_f:
                generated_text = [line.strip() for line in gen_f.readlines() if line.strip()]  # Remove empty or whitespace-only lines
            
            with open(label_file, "r", encoding="utf-8", errors="ignore") as ref_f:
                reference_text = [line.strip() for line in ref_f.readlines() if line.strip()]  # Remove empty or whitespace-only lines

            # Check if we have valid non-empty texts
            if not generated_text or not reference_text:
                print(f"Warning: Skipping {filename} due to empty generated or reference text.")
                continue  # Skip this file if the text is empty

            # Process each pair of generated and reference sentences
            for gen_line, ref_line in zip(generated_text, reference_text):
                # Compute BERTScore for each line
                P, R, F1 = scorer.score([gen_line], [ref_line])  # Single sentence comparison
                
                # Append results to CSV
                with open(OUTPUT_FILE, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([folder_path, round(P.item(), 2), round(R.item(), 2), round(F1.item(), 2)])

print(f"Processing completed. Results saved in {OUTPUT_FILE}.")
