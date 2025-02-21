import argparse
import os
from bert_score import score

def get_bertscore(ref_path, gen_path):
    """
    Computes BERTScore for a list of reference and generated sentences.

    Args:
        ref_path (str): Path to the reference sentences file.
        gen_path (str): Path to the generated sentences file.

    Returns:
        float: The average BERTScore F1 score across all sentences.
    """
    # Read files and split by line
    with open(gen_path, encoding="utf-8") as f:
        gen_sentence_lst = f.read().strip().split("\n")

    with open(ref_path, encoding="utf-8") as f:
        ref_sentence_lst = f.read().strip().split("\n")

    # Compute BERTScore
    P, R, F1 = score(gen_sentence_lst, ref_sentence_lst, lang="en", verbose=True)

    # Calculate average F1 score
    avg_bertscore = F1.mean().item() * 100  # Scale to percentage
    return avg_bertscore

if __name__ == "__main__":

    ##### Get parameters #####
    parser = argparse.ArgumentParser(description="Calculate BERTScore")

    parser.add_argument("-r", "--ref_path", metavar="test.ref.txt",
                        help="Path to the reference sentences file", required=True)
    parser.add_argument("-g", "--gen_path", metavar="test.gen.txt",
                        help="Path to the generated sentences file", required=True)

    args = parser.parse_args()

    if os.path.exists(args.ref_path) and os.path.exists(args.gen_path):
        bertscore_value = get_bertscore(args.ref_path, args.gen_path)
        print(f"{bertscore_value:.2f}")
    else:
        print("File not found.")
