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

def get_meteor_nltk(ref_sentence, gen_sentence):
    """
    Computes the METEOR score for a pair of reference and generated sentences.

    Args:
        ref_sentence (str): The reference sentence.
        gen_sentence (str): The generated sentence.

    Returns:
        float: The METEOR score for the sentence pair.
    """
    # Compute METEOR score for the sentence pair
    meteor_score_value = meteor_score([ref_sentence.split()], gen_sentence.split())
    return meteor_score_value

if __name__ == "__main__":

    ##### Get parameters #####
    parser = argparse.ArgumentParser(description='Calculate METEOR Score')

    # These parameters now accept direct sentences instead of file paths
    parser.add_argument("--refs", metavar="REFERENCE_SENTENCE", 
                        help="Reference sentence", required=True)
    parser.add_argument("--gen", metavar="GENERATED_SENTENCE", 
                        help="Generated sentence", required=True)

    args = parser.parse_args()

    # Get the METEOR score for the given sentences
    meteor_score_value = get_meteor_nltk(args.refs, args.gen)
    print(f"{meteor_score_value:.2f}")
#python Meteor.py --refs "this is the reference text" --gen "this is the generated text"