import csv

with open("./output_lines.csv", "r") as file, open(
    "output_lines_java.csv", "w"
) as o_java:
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[0] == "Foldername":
            o_java.write(
                "ModelName,Line Number,B-Moses,B-Norm,B-NLTK,Rouge-L,METEOR" + "\n"
            )
            continue

        path = row[0].split("/")
        model = path[2]

        o_java.write(
            f"{model},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}" + "\n"
        )
