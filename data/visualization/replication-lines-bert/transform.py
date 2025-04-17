import csv

with open("./output_bert_per_line.csv", "r") as file, open(
    "output_bert_java.csv", "w"
) as o_java:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] == "Foldername":

            o_java.write("ModelName,BERTScore Precision (Mean),BERTScore Recall (Mean),BERTScore F1 (Mean)" + "\n")
            continue

        path = row[0].split("/")
        lang = "java"
        model = path[-1]

        if lang == "java":
            o_java.write(f"{model},{row[1]},{row[2]},{row[3]}" + "\n")
