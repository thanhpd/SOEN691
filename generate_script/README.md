## Installation
```python
$ pip install requests datasets
```

### Usage

Running the shell script (for multiple LLMs)
```sh
$ chmod +x run.sh
$ ./run.sh
```

Running the python script (for single LLM)
```python
$ python generate.py --m=<model_name> --l=<programming_language>
$ python generate.py --m=llama3.2:1b --l=py # example
(or)
$ python generate.py --model=llama3.2:1b --lang=py 
```
accepted programming languages: py, go, js, rb, php, java