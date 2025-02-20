import getopt, sys
import asyncio
import json
import os
import time

from datasets import load_dataset
from ollama import AsyncClient

ds = load_dataset("Maxscha/commitbench", split="test", streaming=False)

OLLAMA_URL = "http://localhost:11434/api/generate"
SEED = 42
client = AsyncClient()


async def call_ollama_model(model: str, prompt: str) -> str:
    response = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format={"type": "object", "properties": {"message": {"type": "string"}}},
        options={"temperature": 0.5, "seed": SEED},
    )
    parsed_response = json.loads(response.message.content)
    return parsed_response["message"]


async def main():
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

    filename = f"{lang}/{model_name}.msg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    tasks = []
    row_count = 0
    for _, data in enumerate(ds):
        row_count += 1
        if data["diff_languages"] == lang:
            diff = data["diff"]
            prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
            tasks.append(call_ollama_model(model_name, prompt))
        if row_count == 1000:
            break
    
    results = await asyncio.gather(*tasks)
    with open(filename, "w", encoding="utf-8") as op:
        for _, commit_msg in enumerate(results):
            op.write(commit_msg + "\n")
            op.flush()
    print(f"processed {row_count} row(s) for {lang}/{model_name} in {time.time() - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
