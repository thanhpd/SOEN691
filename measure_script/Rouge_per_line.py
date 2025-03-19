import argparse
import json
import os

### Rouge
def get_rouge(ref_text, gen_text, is_sentence=True):
    if is_sentence:
        # Evaluating directly as a sentence, no file needed
        evaluate_cmd = "sumeval r-nl \"{}\" \"{}\"".format(gen_text, ref_text)
    else:
        # If needed, you could extend for file-based comparison
        evaluate_cmd = "sumeval r-nl -f \"{}\" \"{}\" -in".format(gen_text, ref_text)
    
    rouge_score = json.load(os.popen(evaluate_cmd))["averages"]
    for key in rouge_score.keys():
        rouge_score[key] = round(rouge_score[key], 3)
    return rouge_score

if __name__ == "__main__":

    ##### get parameters #####
    parser = argparse.ArgumentParser(description='Calculate Rouge-1, Rouge-2, Rouge-N by sumeval')

    # These parameters will accept the actual reference and generated sentences as input
    parser.add_argument("--refs", metavar="REFERENCE_SENTENCE", 
                        help='Reference sentence for comparison', required=True)
    parser.add_argument("--gen", metavar="GENERATED_SENTENCE", 
                        help='Generated sentence for evaluation', required=True)

    args = parser.parse_args()

    # Get the ROUGE score for the given reference and generated sentences
    print(f"{get_rouge(args.refs, args.gen)['ROUGE-L']}")
#python Rouge.py --refs "this is the reference text" --gen "this is the generated text"