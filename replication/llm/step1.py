import getopt, sys
import os
import time
import json

from ollama import chat

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42
EMPTY_RESPONSE_CODE = "ERR::EMPTY_RESPONSE"

def get_prompt(diff: str) -> str:
    return f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""

def call_ollama_model(model: str, prompt: str, temperature: float) -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format={"type": "object", "properties": {"message": {"type": "string"}}},
        options={"temperature": temperature, "seed": SEED},
    )
    try:
        parsed_response = json.loads(response.message.content)
        if "message" not in parsed_response:
            print(f"err: cannot parse json response: {response}")
            return False, EMPTY_RESPONSE_CODE

        return True, parsed_response["message"]
    except Exception as e:
        print(f"err: cannot parse json response: {response}")
        return False, str(e)

def call_ollama_model2(model: str, prompt: str, temperature: float) -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "seed": SEED},
    )
    try:
        parsed_response = response.message.content
        return True, parsed_response
    except Exception as e:
        print(f"err: response: {response}")
        return False, "err: empty response"


def main():
    start_time = time.time()
    options, _ = getopt.getopt(sys.argv[1:], "", ["model=", "temp="])
    model_name = ""
    temperature = 0.5

    if len(options) == 0:
        print(
            f"missing --m option \n\nExample Usage: `python step1.py --m=llama3.2:1b`"
        )
        return
    else:
        model_name = options[0][1]

    with open("all_result.json", "r") as f:
        ds = json.load(f)

    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg
        elif opt in ("--temp"):
            try:
                temperature = float(arg)
            except ValueError:
                print("Error: Temperature must be a number.")
                sys.exit(2)

    sluggified_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"output/{sluggified_model_name}.json"
    filename_log = f"output/{sluggified_model_name}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(filename_log), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as result_file, open(filename_log, "w", encoding="utf-8") as log:
        results = []
        n = len(ds)
        row_count = 0
        print(f"""Processing {n} diffs via the model {model_name}""")
        for i in range(n):
            row_count += 1
            data = ds[i]
            diff = data["diff"]

            prompt = get_prompt(diff)
            is_success, response = call_ollama_model(model_name, prompt, temperature)

            if is_success and len(response) > 0:
                print(f"{i}: {response}")
                results.append({
                    "diff": diff,
                    "label": data["label"],
                    "message": f"{repr(response)[1:-1]}",
                    "is_retried": False
                })
                # op.write(f"{repr(response)[1:-1]}\n")
            else:
                print(f"Failed to generate commit message for {i}th diff")
                log.write(f"{i},\"{repr(response)[1:-1]}\"\n")

                if response == EMPTY_RESPONSE_CODE:
                    # Retry with none
                    is_retry_success, retry_response = call_ollama_model2(model_name, prompt, temperature)
                    if is_retry_success and len(retry_response) > 0:
                        print(f"{i}: {retry_response}")
                        results.append({
                            "diff": diff,
                            "label": data["label"],
                            "message": f"{repr(retry_response)[1:-1]}",
                            "is_retried": True
                        })
                    else:
                        print(f"Failed to generate commit message for {i}th diff")
                        log.write(f"{i},\"{repr(retry_response)[1:-1]}\"\n")

        json.dump(results, result_file, indent=4)

        print(f"processed {row_count} row(s) for {model_name} in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
