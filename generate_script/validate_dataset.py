from datasets import load_from_disk

lang = "py"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}\n---\n")

# -- 
lang = "go"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}\n---\n")

# -- 
lang = "js"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}\n---\n")

# -- 
lang = "php"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}\n---\n")

# -- 
lang = "java"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}\n---\n")

# -- 
lang = "rb"
path = f"./commitbench_{lang}"
ds = load_from_disk(path)
ds = ds.with_format("pandas")
print(f"{ds["project"].nunique()}: unique {lang} project count\ndataset path: {path}")