import getopt, sys
import json
import os
import time

from datasets import load_from_disk
from ollama import Client

# ds = load_dataset("Maxscha/commitbench", split="test", streaming=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42
ROW_COUNT = 20000

client = Client(
    host="http://localhost:11434",
)


def call_ollama_model(model: str, prompt: str, temp: float):
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format={"type": "object", "properties": {"message": {"type": "string"}}},
            options={"temperature": temp, "seed": SEED},
        )
        parsed_response = json.loads(response.message.content)
        return True, parsed_response["message"]
    except:
        print("err: cannot parse empty json response", flush=True)
        return False, ""


def main():
    start_time = time.time()
    options, _ = getopt.getopt(sys.argv[1:], "mlt:", ["model=", "lang=", "temp="])
    model_name = ""
    lang = ""
    temperature = 0.5
    if len(options) <= 2:
        print(
            f"missing/invalid --m or --l options \n\nExample Usage: `python generate.py --model=llama3.2:1b --lang=py`",
        )
        return
    for opt, arg in options:
        if opt in ("m", "--model"):
            model_name = arg
        elif opt in ("l", "--lang"):
            lang = arg
        elif opt in ("t", "--temp"):
            temperature = float(arg)
        else:
            print(
                f"missing/invalid --m or --l  options \n\nExample Usage: `python generate.py --model=llama3.2:1b --lang=py --temp=0.5`",
                f"\naccepted programming languages: py, go, js, rb, php, java",
            )

    sluggified_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = (
        f"{lang}/{sluggified_model_name}/{temperature}/{sluggified_model_name}.msg"
    )
    label_filename = f"{lang}/{sluggified_model_name}/{temperature}/label.msg"
    filename_log = f"output/{lang}_{sluggified_model_name}_{temperature}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(label_filename), exist_ok=True)
    os.makedirs(os.path.dirname(filename_log), exist_ok=True)

    ds = load_from_disk(f"./commitbench_{lang}")

    row_counter = 0
    with open(filename, "w", encoding="utf-8") as op, open(
        filename_log, "w", encoding="utf-8"
    ) as log, open(label_filename, "w", encoding="utf-8") as label:
        log.write("id,hash,original_msg,generated_msg,project\n")
        
        for i, data in enumerate(ds):
            if data["diff_languages"] == lang:
                print(f"model: {model_name} | lang: {lang} | temp: {temperature} | commit_hash: {data["hash"]}", flush=True)
                diff = data["diff"]
                prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
                is_success, response = call_ollama_model(
                    model_name, prompt, temperature
                )

                if is_success:
                    ...
                row_counter += 1
                op.write(repr(response)[1:-1] + "\n")
                label.write(repr(data["message"])[1:-1] + "\n")

                log.write(
                    f'{i},"{data["hash"]}","'
                    + repr(data["message"])[1:-1]
                    + '","'
                    + repr(response)[1:-1]
                    + f'","{data["project"]}"'
                    + '\n'
                )
                if row_counter == ROW_COUNT:
                    break
        end_process_str = f"processed {row_counter} row(s) for {lang}/{model_name} in {time.time() - start_time} seconds"
        log.write(end_process_str + ",,,\n")
        print(end_process_str, flush=True)


if __name__ == "__main__":
    main()
