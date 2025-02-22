import getopt, sys
import requests
import os
import time
import json

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

    # Parse and display the response
    try:
        if response.status_code == 200:
            result = response.json()
            parsed_response = json.loads(result["response"])
            print(parsed_response)
            return parsed_response["message"]
    except:
        return f"err: {response.text}"


def main():
    start_time = time.time()
    options, _ = getopt.getopt(sys.argv[1:], "m:", "model=")
    model_name = ""

    if len(options) == 0:
        print(
            f"missing --m option \n\nExample Usage: `python generate.py --m=llama3.2:1b`"
        )
        return
    else:
        model_name = options[0][1]

    with open("all_result.json", "r") as f:
        ds = json.load(f)

    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg

    filename = f"output/{model_name.replace(':', '_')}.msg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as op:
        # global row_count
        for _, data in enumerate(ds):
            diff = data["diff"]

            prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
            generated_commit_message = call_ollama_model(model_name, prompt)
            op.write(generated_commit_message + "\n")

        # print(f"processed {row_count} row(s) for {model_name} in {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
