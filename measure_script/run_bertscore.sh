#!/bin/bash

echo "Setting up the environment..."

# Install required Python packages
echo "Installing required Python packages..."
git clone https://github.com/Tiiiger/bert_score
cd bert_score
pip install .

# Define input folder, reference file, and output CSV file
GEN_FOLDER="generated_msg"  # Folder containing generated files
REF_FILE="generated_msg/label.msg"
OUTPUT_FILE="bertscore_output.csv"

# Check if the folder exists
if [ ! -d "$GEN_FOLDER" ]; then
    echo "Error: Folder '$GEN_FOLDER' not found!"
    exit 1
fi

# Initialize CSV file with headers
echo "Filename,Precision,Recall,F1" > "$OUTPUT_FILE"

# Process each file in the folder
for GEN_FILE in "$GEN_FOLDER"/*; do
    FILENAME=$(basename "$GEN_FILE")

    # Skip the reference file
    if [ "$FILENAME" == "label.msg" ]; then
        continue
    fi

    echo "Processing $FILENAME..."

    # Ensure the file exists
    if [ ! -f "$GEN_FILE" ]; then
        echo "Error: Candidate file '$GEN_FILE' does not exist!"
        continue
    fi

    # Run BERTScore and filter output
    BERTSCORE_OUTPUT=$(bert-score -r "$REF_FILE" -c "$GEN_FILE" --lang en 2>&1)

    # Extract only Precision, Recall, and F1-score using awk
    P=$(echo "$BERTSCORE_OUTPUT" | awk '/P:/ {print $NF}')
    R=$(echo "$BERTSCORE_OUTPUT" | awk '/R:/ {print $NF}')
    F1=$(echo "$BERTSCORE_OUTPUT" | awk '/F1:/ {print $NF}')

    # Check if values were extracted correctly
    if [[ -z "$P" || -z "$R" || -z "$F1" || "$P" == "P:" || "$R" == "R:" || "$F1" == "F1:" ]]; then
        echo "Error extracting scores for $FILENAME. Skipping..."
        continue
    fi

    # Append results to CSV
    echo "$FILENAME,$P,$R,$F1" >> "$OUTPUT_FILE"

done

echo "Processing completed. Results saved in $OUTPUT_FILE."

# Keep terminal open
read -p "Press any key to exit..."
