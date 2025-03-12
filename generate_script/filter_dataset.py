"""Loading dataset from Huggingface"""
from datasets import load_dataset

SPLIT = "train+test"
N = 30000


# --
LANGUAGE = "py"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)

# --
LANGUAGE = "go"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)

# --
LANGUAGE = "java"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)

# --
LANGUAGE = "js"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)

# --
LANGUAGE = "php"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)

# --
LANGUAGE = "rb"
SAVE_PATH = f"./commitbench_{LANGUAGE}"

ds = load_dataset("Maxscha/commitbench", split=SPLIT)
ds = ds.shuffle(seed=42)
ds = ds.filter(lambda data: data["diff_languages"] == LANGUAGE)
ds = ds.take(N)
ds.save_to_disk(SAVE_PATH)