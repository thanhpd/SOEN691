#!/bin/sh

# Clone the repository
git clone https://github.com/Tbabm/nngen.git --depth 1

# Change to the cloned directory
cd nngen

# Pull the docker images
docker pull tbabm/nngen:0.1

# ===========================
# Verify that the docker images produce the same results as published in the paper
echo "run nngen on the dataset provided by NNGen"
docker run -it --rm -v $(pwd):/root/nngen --name run-nngen tbabm/nngen:0.1 \
       python3 -m nngen main ./data/cleaned.train.diff ./data/cleaned.train.msg ./data/cleaned.test.diff

# evaluate
echo "Evaluate the results: BLEU = 16.42, 27.6/16.8/13.4/11.8"
./scripts/multi-bleu.perl ./data/cleaned.test.msg < ./nngen.cleaned.test.msg

# ===========================
# Extract diffs from the data provided by original study
cd ..
echo "Extract diffs from the data provided by original study"
mkdir -p ../data
python extract_diff.py --input=../../data/replication_rq1/data_source/all_result.json --output=./nngen/data/paper_extracted_diffs.diff --labelin=../../data/replication_rq1/data_source/msg/label.msg --labelout=./nngen/data/paper_extracted_diffs.msg
cd nngen

# Run the docker image on the original study
echo "run nngen on the dataset of the original study"
docker run -it --rm -v $(pwd):/root/nngen --name run-nngen tbabm/nngen:0.1 \
       python3 -m nngen main ./data/cleaned.train.diff ./data/cleaned.train.msg ./data/paper_extracted_diffs.diff

# evaluate the results
echo "Evaluate the results"
./scripts/multi-bleu.perl ./data/paper_extracted_diffs.msg < ./nngen.paper_extracted_diffs.msg
