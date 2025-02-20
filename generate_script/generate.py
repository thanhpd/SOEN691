import getopt, sys
import requests
import json
import os

from datasets import load_dataset

ds = load_dataset("Maxscha/commitbench")

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42


def call_ollama_model(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "format": {"type": "object", "properties": {"message": {"type": "string"}}},
        "stream": False,
        "options": {"temperature": 0.5, "seed": SEED},
    }
    response = requests.post(OLLAMA_URL, data=json.dumps(payload))

    try:
        if response.status_code == 200:
            result = response.json()
            parsed_response = json.loads(result["response"])

            return parsed_response["message"]
    except:
        return f"err: {response.text}"


def main():
    options, _ = getopt.getopt(sys.argv[1:], "ml:", ["model=", "lang="])
    model_name = ""
    lang = ""
    if len(options) != 2:
        print(
            f"missing/invalid --m or --l options \n\nExample Usage: `python generate.py -m=llama3.2:1b -l=py`",
        )
        return
    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("l", "--lang"):
            lang = arg
        else:
            print(
                f"missing/invalid --m or --l options \n\nExample Usage: `python generate.py -m=llama3.2:1b -l=py`",
                f"\naccepted programming languages: py, go, js, rb, php, java",
            )

    filename = f"{lang}/{model_name}.msg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(f"{lang}/{model_name}" + ".msg", "w", encoding="utf-8") as op:
        for data in ds["test"]:
            if data["diff_languages"] == lang:
                diff = data["diff"]
                prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
                generated_commit_message = call_ollama_model(model_name, prompt)
                op.write(generated_commit_message + "\n")


if __name__ == "__main__":
    main()
