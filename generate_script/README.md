## Installation
```python
$ pip install ollama datasets
```

### Usage

Running the shell script (for multiple LLMs)
```sh
$ chmod +x run.sh
$ ./run.sh
```

Running the python script (for single LLM)
```python
$ python generate.py --m=<model_name> --l=<programming_language> --t=<temperature>
$ python generate.py --m=llama3.2:1b --l=py --t=0.5 # example
(or)
$ python generate.py --model=llama3.2:1b --lang=py --temperature=0.5
```
accepted programming languages: py, go, js, rb, php, java