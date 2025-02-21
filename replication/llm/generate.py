import getopt, sys
import asyncio
import json
import os
import time
import json

from ollama import AsyncClient

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
    print(parsed_response)
    return parsed_response["message"]


async def main():
    start_time = time.time()
    options, _ = getopt.getopt(sys.argv[1:], "ml:", ["model="])
    model_name = ""

    with open("all_result.json", "r") as f:
        ds = json.load(f)

    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg

    filename = f"output/{model_name.replace(':', '_')}.msg"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    tasks = []
    row_count = 0
    for _, data in enumerate(ds):
        row_count += 1
        diff = data["diff"]
        prompt = f"""The following is a diff which describes the code changes in a commit, Your task is to write a short commit message accordingly. {diff} According to the diff, the commit message should be:"""
        tasks.append(call_ollama_model(model_name, prompt))

    results = await asyncio.gather(*tasks)
    with open(filename, "w", encoding="utf-8") as op:
        for _, commit_msg in enumerate(results):
            op.write(commit_msg + "\n")
    print(f"processed {row_count} row(s) for {model_name} in {time.time() - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
