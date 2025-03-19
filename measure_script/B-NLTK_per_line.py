import os
import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

#nltk.download('punkt')

def get_bleu_nltk(ref_sentences, gen_sentences):
    # Calculate BLEU score for each sentence
    sentence_bleu_lst = [
        sentence_bleu([ref.split()], gen.split(), smoothing_function=SmoothingFunction().method5)
        for ref, gen in zip(ref_sentences, gen_sentences)
    ]
    
    # Calculate the average BLEU score
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu * 100

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='Calculate B-NLTK BLEU score')

    parser.add_argument("-r", "--ref", metavar="REFERENCE_SENTENCES",
                        help="Space-separated reference sentences", required=True)
    parser.add_argument("-g", "--gen", metavar="GENERATED_SENTENCES",
                        help="Space-separated generated sentences", required=True)

    args = parser.parse_args()

    # Split input sentences into lists
    ref_sentences = args.ref.split('|')  # assuming sentences are separated by "|"
    gen_sentences = args.gen.split('|')  # assuming sentences are separated by "|"

    # Ensure both lists have the same length
    if len(ref_sentences) != len(gen_sentences):
        print("Error: The number of reference sentences must match the number of generated sentences.")
    else:
        bleu_score = get_bleu_nltk(ref_sentences, gen_sentences)
        print(f"{bleu_score:.2f}")
#python script.py --ref "this is the reference text|this is another reference sentence" --gen "this is the generated text|this is another generated sentence"
