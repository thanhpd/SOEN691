import os
import argparse
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score

# Ensure WordNet is available for METEOR
# Check if WordNet is already downloaded
try:
    nltk.data.find('corpora/wordnet.zip')  # Check if WordNet exists
except LookupError:
    nltk.download('wordnet', quiet=True)  # Download silently if missing

def get_meteor_nltk(ref_path, gen_path):
    """
    Computes the METEOR score for a list of reference and generated commit messages.

    Args:
        ref_path (str): Path to the reference commit messages file.
        gen_path (str): Path to the generated commit messages file.

    Returns:
        float: The average METEOR score across all commit messages.
    """
    # Read files and split by line
    gen_sentence_lst = open(gen_path, encoding="utf-8").read().strip().split("\n")
    ref_sentence_lst = open(ref_path, encoding="utf-8").read().strip().split("\n")

    # Compute METEOR scores for each sentence pair
    meteor_scores = [
        meteor_score([ref_sentence.split()], gen_sentence.split()) 
        for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)
    ]

    # Compute average METEOR score
    avg_meteor = np.mean(meteor_scores)
    return avg_meteor   # Scale to percentage

if __name__ == "__main__":

    ##### Get parameters #####
    parser = argparse.ArgumentParser(description='Calculate METEOR Score')

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help="Path to the reference commit messages file", required=True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help="Path to the generated commit messages file", required=True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        meteor_score_value = get_meteor_nltk(args.ref_path, args.gen_path)
        print(f"{meteor_score_value:.2f}")

    else:
        print("File not found.")
