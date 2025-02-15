import nltk
import sacrebleu
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Ensure NLTK resources are downloaded
nltk.download("punkt")

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()

def calculate_bleu_nltk(reference, hypothesis):
    gen_sentence_lst = open("test.gen.txt").read().split("\n")
    ref_sentence_lst = open("test.ref.txt").read().split("\n")
    sentence_bleu_lst = [sentence_bleu([ref_sentence.split()], gen_sentence.split(), smoothing_function=SmoothingFunction().method5) for ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu*100

def calculate_bleu_moses(reference, hypothesis):
    return sacrebleu.corpus_bleu([hypothesis], [[reference]]).score

def calculate_bleu_norm(reference, hypothesis):
    return sacrebleu.corpus_bleu([hypothesis], [[reference]], tokenize="intl").score

def calculate_rouge_l(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure

def evaluate_texts(reference_file, hypothesis_file):
    references = read_file(reference_file)
    hypotheses = read_file(hypothesis_file)
    
    bleu_nltk_scores = []
    bleu_moses_scores = []
    bleu_norm_scores = []
    rouge_l_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        bleu_nltk_scores.append(calculate_bleu_nltk(ref.strip(), hyp.strip()))
        bleu_moses_scores.append(calculate_bleu_moses(ref.strip(), hyp.strip()))
        bleu_norm_scores.append(calculate_bleu_norm(ref.strip(), hyp.strip()))
        rouge_l_scores.append(calculate_rouge_l(ref.strip(), hyp.strip()))
    
    print(f"BLEU-NLTK Score: {sum(bleu_nltk_scores) / len(bleu_nltk_scores):.4f}")
    print(f"BLEU-Moses Score: {sum(bleu_moses_scores) / len(bleu_moses_scores):.4f}")
    print(f"BLEU-Norm Score: {sum(bleu_norm_scores) / len(bleu_norm_scores):.4f}")
    print(f"ROUGE-L Score: {sum(rouge_l_scores) / len(rouge_l_scores):.4f}")

if __name__ == "__main__":
    reference_file = "test.ref.txt"  # Change to your actual reference file
    hypothesis_file = "test.gen.txt"  # Change to your actual hypothesis file
    evaluate_texts(reference_file, hypothesis_file)