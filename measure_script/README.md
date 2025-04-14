# Evaluation Metrics Measurement

## How to start the measurement
### 1. Tokenization
1. After generating the outputs as .msg file then copy all the data to `generated_msg` folder
2. Run tokenizer: `python run_tokenizer.py`
3. Once finished running, find the output files under the folder called `processed_msg`

### 2. Collect summary metrics
To collect BLEU, ROUGE-L, METEOR metrics, run the following command. Once finished running, see the output in `output.csv`.
```sh
# Run the metric collection
python run_metrics.py
```

To collect BERTScore metric, run the following commands. Once finished running, see the output in `output_bert.csv`.
```sh
# Install bert-score library
pip install bert-score

# Run the metric collection
python run_bertscore.py
```

### 3. Collect per line metrics for statistical testing
To collect the BLEU, ROUGE-L, METEOR metrics for each pair of label and generated commit message lines, run the following command. Once finished running, see the output in `output_lines.csv`.
```sh
# Run the metric collection
python run_metrics_per_line.py
```

To collect BERTScore metric for each pair of label and generated commit message lines, run the following commands. Once finished running, see the output in `output_bert_per_line.csv`.
```sh
# Install bert-score library
pip install bert-score

# Run the metric collection
python bertscore_per_line.py
```

## Reference
- BLEU, ROUGE-L metric evaluation scripts comes from https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical
- METEOR metric evaluation script comes from https://github.com/facebookresearch/vizseq/blob/main/vizseq/scorers/meteor.py
- BERTScore metric evaluation script: https://huggingface.co/spaces/evaluate-metric/bertscore
