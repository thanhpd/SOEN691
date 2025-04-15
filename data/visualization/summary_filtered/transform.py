import csv

with open("./output_bert.csv", "r") as file, open(
    "output__bert_java.csv", "w"
) as o_java, open("output__bert_js.csv", "w") as o_js, open(
    "output__bert_py.csv", "w"
) as o_py:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] == "Foldername":
            # o_java.write("ModelName,B-Moses,B-Norm,B-NLTK,Rouge-L,METEOR" + "\n")
            # o_js.write("ModelName,B-Moses,B-Norm,B-NLTK,Rouge-L,METEOR" + "\n")
            # o_py.write("ModelName,B-Moses,B-Norm,B-NLTK,Rouge-L,METEOR" + "\n")
            
            o_java.write("ModelName,BERTScore Precision (Mean),BERTScore Recall (Mean),BERTScore F1 (Mean)" + "\n")
            o_js.write("ModelName,BERTScore Precision (Mean),BERTScore Recall (Mean),BERTScore F1 (Mean)" + "\n")
            o_py.write("ModelName,BERTScore Precision (Mean),BERTScore Recall (Mean),BERTScore F1 (Mean)" + "\n")
            continue
        
        path = row[0].split("/")
        lang = path[2]
        model = path[3]
        
        # if lang == "java":
        #     o_java.write(f"{model},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}" + "\n")
        # elif lang == "js":
        #     o_js.write(f"{model},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}" + "\n")
        # elif lang == "py":
        #     o_py.write(f"{model},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}" + "\n")

        if lang == "java":
            o_java.write(f"{model},{row[1]},{row[2]},{row[3]}" + "\n")
        elif lang == "js":
            o_js.write(f"{model},{row[1]},{row[2]},{row[3]}" + "\n")
        elif lang == "py":
            o_py.write(f"{model},{row[1]},{row[2]},{row[3]}" + "\n")
