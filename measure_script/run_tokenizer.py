import os
import subprocess

# Define input and output folders
INPUT_FOLDER = "./generated_msg"  # Change this to your actual input folder
OUTPUT_FOLDER = "./processed_msg"
TOKENIZER_SCRIPT = "./post_processing/tokenizer.perl"
LANGUAGE = "en"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process each file in the input folder
for filename in os.listdir(INPUT_FOLDER):
    input_file = os.path.join(INPUT_FOLDER, filename)
    output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.msg")

    # Run the tokenizer command
    try:
        with open(input_file, "r") as in_f, open(output_file, "w") as out_f:
            subprocess.run([TOKENIZER_SCRIPT, "-l", LANGUAGE], stdin=in_f, stdout=out_f, check=True)
        print(f"Processed: {filename} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")

print("Processing completed.")
