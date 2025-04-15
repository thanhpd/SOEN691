# Evaluating Lightweight Large Language Models for Git Commit Message Generation
This is the replication package for our work in the course SOEN 691: Generative AI in Software Engineering at Concordia University, Winter 2025.

## Experimental Models
- Baseline models: [CommitGen(CmtGen)][CommitGen], [NMT][NMT], [CoDiSum][CoDiSum], [PtrGNCMsg][PtrGNCMsg], [NNGen][NNGen]
- LLM models on [Ollama](https://ollama.com/): [`llama2:7b`](https://ollama.com/library/llama2), [`llama2:70b`](https://ollama.com/library/llama2:70b), [`gemma2:9b`](https://ollama.com/library/gemma2), [`llama3.2:3b`](https://ollama.com/library/llama3.2), [`codellama:7b`](https://ollama.com/library/codellama)

## Experimental Datasets
- Existing dataset: [Using Large Language Models for Commit Message Generation: A Preliminary Study](https://zenodo.org/records/10491384)
- Modified and filtered [CommitBench dataset](https://github.com/Maxscha/commitbench): Available on [Zenodo](https://zenodo.org/records/15220466)

## Evaluation Metrics
- [B-Moses](measure_script/B-Moses.perl)
- [B-Norm](measure_script/B-Norm.py)
- [B-NLTK](measure_script/B-NLTK.py)
- [ROUGE-L](measure_script/Rouge.py)
- [METEOR](measure_script/Meteor.py)
- [BERTScore](measure_script/run_bertscore.py)

Usage demo about the metrics can be found [here](measure_script/README.md).

## Research Questions
### RQ1: How replicable are the quantitative results of the study "Using Large Language Models for Commit Message Generation: A Preliminary Study"?
See RQ1 results here.

### RQ2: What is the impact of different temperature settings on the performance of LLMs commit message generation?
See RQ2 results [here](data/temperature_rq2/evaluation/output.csv).

### RQ3: How does the performance of smaller OLLMs compare in generating commit messages across different programming languages?
See RQ3 results [here](data/language_rq3//evaluation).

## Repository structure
- [`replication`](/replication) folder contains the code for RQ1
- [`generate_script`](/generate_script) folder contains the code for generating Git commit messages used in RQ2 and RQ3
- [`measure_script`](/measure_script) folder contains the code for getting the quantitative measurements of data, used in all 3 RQs
- [`data`](/data) folder contains all the data sources, generated outputs, and evaluation results of all RQs

[CommitGen]: https://sjiang1.github.io/commitgen/ "S. Jiang, A. Armaly, and C. McMillan, “Automatically generating commit messages from diffs using neural machine translation,” in ASE, 2017."

[NMT]: https://github.com/epochx/commitgen "P. Loyola, E. Marrese-Taylor, and Y. Matsuo, “A neural architecture for generating natural language descriptions from source code changes,” in ACL. Association for Computational Linguistics, 2017, pp. 287–292."

[CoDiSum]: https://github.com/SoftWiser-group/CoDiSum "S. Xu, Y. Yao, F. Xu, T. Gu, H. Tong, and J. Lu, “Commit message generation for source code changes,” in IJCAI. ijcai.org, 2019, pp. 3975–3981."

[PtrGNCMsg]: https://zenodo.org/record/2542706#.XECK8C277BJ "Q. Liu, Z. Liu, H. Zhu, H. Fan, B. Du, and Y. Qian, “Generating commit messages from diffs using pointer-generator network,” in MSR. IEEE / ACM, 2019, pp. 299–309."

[NNGen]: https://github.com/Tbabm/nngen "Z. Liu, X. Xia, A. E. Hassan, D. Lo, Z. Xing, and X. Wang, “Neuralmachine-translation-based commit message generation: how far are we?” in ASE. ACM, 2018, pp. 373–384."
