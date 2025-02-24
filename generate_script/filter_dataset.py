from datasets import load_dataset

ds = load_dataset("Maxscha/commitbench", split="test")
ds.filter(lambda data: data["diff_languages"] == "py")
ds.to_csv("./commitbench_py.csv")