import os
import subprocess
import csv
import sys

print("Setting up the environment...")

# Install required Python packages
print("Installing required Python packages...")
os.system("pip install --upgrade pip")
os.system("pip install nltk")
os.system("pip install sumeval sacrebleu==1.5.1")

# Define input folder, reference file, and output CSV file
GEN_FOLDER = "Processed_msg"
OUTPUT_FILE = "output.csv"

# Check if the folder exists
if not os.path.isdir(GEN_FOLDER):
    print(f"Error: Folder '{GEN_FOLDER}' not found!")
    sys.exit(1)

# Initialize CSV file with headers
headers = ["Foldername","B-Moses", "B-Norm", "B-NLTK", "Rouge-L", "METEOR"]
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

def run_and_capture(command):
    """Runs a command and captures its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip().replace("\n", " ").replace(",", " ")
        return output
    except Exception as e:
        return f"Error: {str(e)}"

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

            # Capture output from each command
            moses_output = run_and_capture(f"cat {gen_file} | perl B-Moses.perl {label_file}")
            norm_output = run_and_capture(f"python B-Norm.py {label_file} {gen_file}")
            nltk_output = run_and_capture(f"python B-NLTK.py -r {label_file} -g {gen_file}")
            rouge_output = run_and_capture(f"python Rouge.py -r {label_file} -g {gen_file}")
            meteor_output = run_and_capture(f"python Meteor.py -r {label_file} -g {gen_file}")
            
            # Append results to CSV
            with open(OUTPUT_FILE, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([folder_path, moses_output, norm_output, nltk_output, rouge_output, meteor_output])

print(f"Processing completed. Results saved in {OUTPUT_FILE}.")
