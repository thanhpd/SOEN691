```sh
# 0. Pull the repo
git clone https://github.com/thanhpd/SOEN691

# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.6.0 sh

# 2. Install Miniconda + Python env
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# 3. Install library dependencies
cd SOEN691/generate_script
pip install -r requirements.txt

# Pull the dataset to the same folder
?

# Run Ollama model
ollama run gemma2

# Execute the data collection
python generate.py --model=gemma2 --lang=py --temp=0.5
```
