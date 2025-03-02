# Replication of the original study:

## Environment installation
- Install Python 3
- Install Ollama and pull the necessary models
- Install missing dependencies
- TODOs

## Run commands
```sh
# Run step 1 with a specific model provided by Ollama.
python step1.py --model=llama2

# IMPORTANT: The script uses structured response by default, there's a bug from Ollama that may render the response to be empty.
# Ref: https://github.com/ollama/ollama/issues/7603
# To address this, a retry was added to collect the unstructured response.
# Run step 2 to produce the file results ready for data analysis.
python step2.py --model=llama2
```
