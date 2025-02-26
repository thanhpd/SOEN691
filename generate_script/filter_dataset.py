from datasets import load_dataset

SPLIT = "test"
LANGUAGE = "py"
ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds.to_csv(f"./commitbench_{LANGUAGE}.csv")
