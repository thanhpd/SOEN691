import getopt, sys
import requests
import json

from datasets import load_dataset
ds = load_dataset("Maxscha/commitbench")

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42


def call_ollama_model(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.5, "seed": SEED}}
    response = requests.post(OLLAMA_URL, data=json.dumps(payload))

    # Parse and display the response
    try:
        if response.status_code == 200:
            result = response.json()
            parsed_response = result["response"].split("\n\n")[1]

            return parsed_response.strip("\"")
    except:
        return f"err: {response.text}"

def main():
    options, _ = getopt.getopt(sys.argv[1:], "m:", "model=")
    model_name = ""
    if len(options) == 0:
        print(f"missing --m option \n\nExample Usage: `python generate.py --m=llama3.2:1b`")
        return
    else:
        model_name = options[0][1]

    with open(model_name + ".msg", "w") as op:
        for data in ds["test"]:
            diff = data["diff"]

            prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
            generated_commit_message = call_ollama_model(model_name, prompt)
            op.write(generated_commit_message + "\n")



if __name__ == "__main__":
    main()
