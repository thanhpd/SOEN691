import getopt, sys
import json
import os
import time

from datasets import load_dataset
from ollama import chat

ds = load_dataset("Maxscha/commitbench", split="test", streaming=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42


def call_ollama_model(model: str, prompt: str) -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format={"type": "object", "properties": {"message": {"type": "string"}}},
        options={"temperature": 0.5, "seed": SEED},
    )
    try:
        parsed_response = json.loads(response.message.content)
        return True, parsed_response["message"]
    except:
        print(f"err: cannot parse json response: {response}")
        return False, "err: empty json response"


def main():
    start_time = time.time()
    options, _ = getopt.getopt(sys.argv[1:], "ml:", ["model=", "lang="])
    model_name = ""
    lang = ""
    if len(options) != 2:
        print(
            f"missing/invalid --m or --l options \n\nExample Usage: `python generate.py --model=llama3.2:1b --lang=py`",
        )
        return
    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("l", "--lang"):
            lang = arg
        else:
            print(
                f"missing/invalid --m or --l options \n\nExample Usage: `python generate.py --model=llama3.2:1b --lang=py`",
                f"\naccepted programming languages: py, go, js, rb, php, java",
            )

    sluggified_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"{lang}/{sluggified_model_name}.msg"
    filename_log = f"output/{sluggified_model_name}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(filename_log), exist_ok=True)

    tasks = []
    row_count = 0
    with open(filename, "w", encoding="utf-8") as op, open(
        filename_log, "w", encoding="utf-8"
    ) as log:
        for i, data in enumerate(ds):
            if data["diff_languages"] == lang:
                print(f"commit_hash: {data["hash"]}")
                row_count += 1
                diff = data["diff"]
                prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
                is_success, response = call_ollama_model(model_name, prompt)
                # if row_count == 100:
                #     break

                if is_success:
                    op.write(repr(response)[1:-1] + "\n")

                log.write(
                    f'{i},"{data["hash"]}","' + repr(response)[1:-1] + '"\n'
                )
    print(
        f"processed {row_count} row(s) for {lang}/{model_name} in {time.time() - start_time} seconds"
    )


if __name__ == "__main__":
    main()
