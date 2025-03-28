import csv
import os

FILEPATHS = [
    "20000_op/output/java_codellama_7b_0.0.csv",
    "20000_op/output/java_gemma2_9b_0.0.csv",
    "20000_op/output/java_llama2_7b_0.0.csv",
    "20000_op/output/java_llama2_70b_0.0.csv",
    "20000_op/output/java_llama3.2_3b_0.0.csv",
    "20000_op/output/js_codellama_7b_0.0.csv",
    "20000_op/output/js_gemma2_9b_0.0.csv",
    "20000_op/output/js_llama2_7b_0.0.csv",
    "20000_op/output/js_llama2_70b_0.0.csv",
    "20000_op/output/js_llama3.2_3b_0.0.csv",
    "20000_op/output/py_codellama_7b_0.0.csv",
    "20000_op/output/py_gemma2_9b_0.0.csv",
    "20000_op/output/py_llama2_7b_0.0.csv",
    "20000_op/output/py_llama2_70b_0.0.csv",
    "20000_op/output/py_llama3.2_3b_0.0.csv",
]

OUTPUT_LABEL_FILEPATH = "label_modified.msg"
OUTPUT_FILEPATH = "llama2_7b_modified.msg"

for fp in FILEPATHS:
    FILEPATH = fp

    filename = fp.split("/")[-1]
    lang = filename.split("_")[0]
    model = filename.split("_")[1] + '_' + filename.split("_")[2]
    temp = filename.split("_")[-1].rstrip(".csv")
    os.makedirs(
        os.path.dirname(f"20000_op_filtered/{lang}/{model}/{temp}/"), exist_ok=True
    )

    OUTPUT_LABEL_FILEPATH = f"20000_op_filtered/{lang}/{model}/{temp}/label.msg"
    OUTPUT_FILEPATH = f"20000_op_filtered/{lang}/{model}/{temp}/{model}.msg"

    with open(FILEPATH) as f, open(OUTPUT_FILEPATH, 'w') as op, open(OUTPUT_LABEL_FILEPATH, 'w') as op_l:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)
        for data in reader:
            if data[3] != '':
                op_l.write(repr(data[2])[1:-1] + '\n')
                op.write(repr(data[3])[1:-1] + '\n')
