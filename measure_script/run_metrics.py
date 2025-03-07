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
GEN_FOLDER = "processed_msg"
REF_FILE = os.path.join(GEN_FOLDER, "label.msg")
OUTPUT_FILE = "output.csv"

# Check if the folder exists
if not os.path.isdir(GEN_FOLDER):
    print(f"Error: Folder '{GEN_FOLDER}' not found!")
    sys.exit(1)

# Initialize CSV file with headers
headers = ["Filename", "B-Moses", "B-Norm", "B-NLTK", "Rouge-L", "METEOR"]
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

# Process each file in the folder
for filename in os.listdir(GEN_FOLDER):
    if filename == "label.msg":
        continue  # Skip reference file
    
    gen_file = os.path.join(GEN_FOLDER, filename)
    print(f"Processing {filename}...")

    # Capture output from each command
    moses_output = run_and_capture(f"cat {gen_file} | perl B-Moses.perl {REF_FILE}")
    norm_output = run_and_capture(f"python B-Norm.py {REF_FILE} {gen_file}")
    nltk_output = run_and_capture(f"python B-NLTK.py -r {REF_FILE} -g {gen_file}")
    rouge_output = run_and_capture(f"python Rouge.py -r {REF_FILE} -g {gen_file}")
    meteor_output = run_and_capture(f"python Meteor.py -r {REF_FILE} -g {gen_file}")
    
    # Append results to CSV
    with open(OUTPUT_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename,  moses_output, norm_output,nltk_output, rouge_output, meteor_output])

print(f"Processing completed. Results saved in {OUTPUT_FILE}.")