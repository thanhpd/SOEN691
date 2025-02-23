import getopt, sys
import requests
import os
import time
import json

from ollama import chat

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42

def call_ollama_model2(model: str, prompt: str) -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format={"type": "object", "properties": {"message": {"type": "string"}}},
        options={"temperature": 0.5, "seed": SEED},
    )
    try:
        parsed_response = json.loads(response.message.content)
        return True, parsed_response["message"]
    except Exception as e:
        print(f"err: cannot parse json response: {response}")
        return False, "err: empty json response"


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

    sluggified_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"output/{sluggified_model_name}.msg"
    filename_log = f"output/{sluggified_model_name}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(filename_log), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as op, open(filename_log, "w", encoding="utf-8") as log:
        n = len(ds)
        row_count = 0
        print(f"""Processing {n} diffs via the model {model_name}""")
        for i in range(n):
            row_count += 1
            data = ds[i]
            diff = data["diff"]

            prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
            is_success, response = call_ollama_model2(model_name, prompt)

            if is_success and len(response) > 0:
                print(f"{i}: {response}")
                op.write(f"{repr(response)[1:-1]}\n")
            else:
                print(f"Failed to generate commit message for {i}th diff")
                log.write(f"{i},\"{repr(response)[1:-1]}\"\n")


        print(f"processed {row_count} row(s) for {model_name} in {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
