# Generate Git commit messages from CommitBench subset

## Environment setup in a new machine
```sh
# 1. Pull the repo
git clone https://github.com/thanhpd/SOEN691

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.6.0 sh

# 3. Install Miniconda + Python env
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# 4. Install library dependencies
cd SOEN691/generate_script
pip install -r requirements.txt

# 5. Download the modified CommitBench subset and extract the file to the `generate_script` folder
# https://zenodo.org/records/15220466

# 6. Run Ollama model
ollama run gemma2

# 7. Execute the data collection
python generate.py --model=gemma2 --lang=py --temp=0.5
```

# Running the python script (for single LLM)
```python
$ python generate.py --m=<model_name> --l=<programming_language> --t=<temperature>
$ python generate.py --m=llama3.2:3b --l=py --t=0.5 # example
(or)
$ python generate.py --model=llama3.2:3b --lang=py --temperature=0.5
```
accepted programming languages: py, go, js, rb, php, java
