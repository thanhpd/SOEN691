import getopt, sys
import time
import json

def write_result_for_structured_response(data: str, model_name: str):
    n = len(data)
    filename = f"output/{model_name}.msg"
    filename_label = f"output/label-{model_name}.msg"

    with open(filename, "w", encoding="utf-8") as result_file, open(filename_label, "w", encoding="utf-8") as label_file:
        for i in range(n):
            current = data[i]
            if current["is_retried"]:
                continue
            result_file.write(current["message"] + "\n")
            label_file.write(current["label"] + "\n")

def write_result_for_fixed_response(data: str, model_name: str):
    n = len(data)
    filename = f"output/{model_name}-fix.msg"
    filename_label = f"output/label-{model_name}-fix.msg"

    with open(filename, "w", encoding="utf-8") as result_file, open(filename_label, "w", encoding="utf-8") as label_file:
        for i in range(n):
            current = data[i]
            result_file.write(current["message"] + "\n")
            label_file.write(current["label"] + "\n")

def main():
    answer = input("Make sure you have fixed the empty response issues. Do you want to continue? (Y/n): ").strip().lower()

    if answer in ["y", "yes", ""]:
        print("Continuing...")
    elif answer in ["n", "no"]:
        print("Exiting...")
        return

    # Load the json file containing the diffs
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

    for opt, arg in options:
        if opt in ("-m", "--model"):
            model_name = arg

    sluggified_model_name = model_name.replace(":", "_").replace("/", "_")
    filename = f"output/{sluggified_model_name}.json"

    with open(filename, "r") as f:
        ds = json.load(f)

    print(f"Processing {len(ds)} diffs via the model {model_name}")

    print("Writing structured results...")
    write_result_for_structured_response(ds, sluggified_model_name)

    print("Writing fixed structured results...")
    write_result_for_fixed_response(ds, sluggified_model_name)

    print(f"done in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
