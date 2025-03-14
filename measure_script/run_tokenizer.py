import os
import subprocess

# Define input and output folders
INPUT_FOLDER = "./generated_msg"  # Change this if needed
OUTPUT_FOLDER = "./processed_msg"
TOKENIZER_SCRIPT = "./post_processing/tokenizer.perl"
LANGUAGE = "en"

# Ensure the base output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Recursively process all .msg files
for root, _, files in os.walk(INPUT_FOLDER):
    relative_path = os.path.relpath(root, INPUT_FOLDER)
    output_dir = os.path.join(OUTPUT_FOLDER, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    for filename in files:
        if filename.endswith(".msg"):
            input_file = os.path.join(root, filename)
            output_file = os.path.join(output_dir, filename)

            # Run the tokenizer using Perl explicitly
            try:
                with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
                    subprocess.run(["perl", TOKENIZER_SCRIPT, "-l", LANGUAGE], stdin=in_f, stdout=out_f, check=True)

                print(f"Processed: {input_file} -> {output_file}")

            except subprocess.CalledProcessError as e:
                print(f"Error processing {input_file}: {e}")
            except FileNotFoundError:
                print(f"Error: File {input_file} not found!")
            except UnicodeDecodeError as e:
                print(f"Unicode decode error with file {input_file}: {e}")

print("Processing completed.")
