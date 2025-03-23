import os
import subprocess
import csv
import sys

print("Setting up the environment...")

# Install required Python packages
print("Installing required Python packages...")
os.system("pip install --upgrade pip")
os.system("pip install nltk")
os.system("pip install numpy")
os.system("pip install sumeval sacrebleu==1.5.1")

# Define input folder, reference file, and output CSV file
GEN_FOLDER = "processed_msg"
OUTPUT_FILE = "output_lines.csv"

# Check if the folder exists
if not os.path.isdir(GEN_FOLDER):
    print(f"Error: Folder '{GEN_FOLDER}' not found!")
    sys.exit(1)

# Initialize CSV file with headers
headers = ["Foldername", "Line Number", "B-Moses", "B-Norm", "B-NLTK", "Rouge-L", "METEOR"]
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

def run_and_capture(command, input_data=None):
    """Runs a command and captures its output."""
    try:
        result = subprocess.run(command, input=input_data, shell=True, capture_output=True, text=True)
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

            # Read the label and generated files line by line with utf-8 encoding
            with open(label_file, 'r', encoding="utf-8") as label_f, open(gen_file, 'r', encoding="utf-8") as gen_f:
                for line_number, (label_line, gen_line) in enumerate(zip(label_f, gen_f), start=1):
                    # Print the current line number and the lines being compared for debugging
                    print(f"Processing line {line_number}: Label Line: {label_line.strip()} Generated Line: {gen_line.strip()}")

                    # Capture output from each command using the lines directly
                    moses_command = f"echo \"{label_line.strip()}\" | perl B-Moses_per_line.perl -lc \"{gen_line.strip()}\""
                    moses_output = run_and_capture(moses_command)
                    
                    norm_command = f"python B-Norm_per_line.py --refs \"{label_line.strip()}\" --gen \"{gen_line.strip()}\""

                    # Run the command and capture the output
                    norm_output = run_and_capture(norm_command)
                    
                    nltk_command = f"python B-NLTK_per_line.py --ref \"{label_line.strip()}\" --gen \"{gen_line.strip()}\""
                    nltk_output = run_and_capture(nltk_command)
                    
                    rouge_command =  f"python Rouge_per_line.py --ref \"{label_line.strip()}\" --gen \"{gen_line.strip()}\""
                    rouge_output = run_and_capture(rouge_command)
                    
                    meteor_command = f"python Meteor_per_line.py --ref \"{label_line.strip()}\" --gen \"{gen_line.strip()}\""
                    meteor_output = run_and_capture(meteor_command)

                    # Append results to CSV with line number and corresponding lines
                    with open(OUTPUT_FILE, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([folder_path, line_number, moses_output, norm_output, nltk_output, rouge_output, meteor_output])

print(f"Processing completed. Results saved in {OUTPUT_FILE}.")
