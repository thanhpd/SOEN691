#!/bin/bash

#!/bin/bash

echo "🚀 Setting up the environment..."

# Install required Python packages
echo "📦 Installing required Python packages..."
pip install --upgrade pip
pip install nltk
pip install sumeval sacrebleu==1.5.1






# Define input folder, reference file, and output CSV file
GEN_FOLDER="generated_msg"  # Folder containing generated files
REF_FILE="generated_msg/label.msg"
OUTPUT_FILE="output.csv"

# Check if the folder exists
if [ ! -d "$GEN_FOLDER" ]; then
    echo "Error: Folder '$GEN_FOLDER' not found!"
    exit 1
fi

# Initialize CSV file with headers
echo "Filename,B-NLTK,B-Norm,B-Moses,Rouge-L" > "$OUTPUT_FILE"

# Function to run a command and capture output
run_and_capture() {
    local gen_file="$1"
    shift
    output=$("$@" 2>&1 | tr '\n' ' ' | tr ',' ' ')  # Capture output, handle newlines and commas
    echo "$output"
}

# Process each file in the folder
for GEN_FILE in "$GEN_FOLDER"/*; do
    FILENAME=$(basename "$GEN_FILE")
    if [ "$FILENAME" == "label.msg" ]; then
        continue  # Skip reference file
    fi
    echo "Processing $FILENAME..."

    # Capture output from each command
    NLTK_OUTPUT=$(run_and_capture "$GEN_FILE" python B-NLTK.py -r "$REF_FILE" -g "$GEN_FILE")
    NORM_OUTPUT=$(run_and_capture "$GEN_FILE" python B-Norm.py "$REF_FILE" "$GEN_FILE")
    MOSES_OUTPUT=$(run_and_capture "$GEN_FILE" bash -c "cat $GEN_FILE | perl B-Moses.perl $REF_FILE")
    ROUGE_OUTPUT=$(run_and_capture "$GEN_FILE" python Rouge.py -r "$REF_FILE" -g "$GEN_FILE")

    # Append results to CSV
    echo "$FILENAME,$NLTK_OUTPUT,$NORM_OUTPUT,$MOSES_OUTPUT,$ROUGE_OUTPUT" >> "$OUTPUT_FILE"

done

echo "Processing completed. Results saved in $OUTPUT_FILE."
