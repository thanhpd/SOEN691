# Given a json file including an array of objects, output a file containing the diff (a property of each object) and respect new lines

import json
import os
import sys
import getopt
import shutil

def main():
  options, _ = getopt.getopt(sys.argv[1:], "", ["input=", "output=", "labelin=", "labelout="])
  # Initialize input and output file paths
  gen_input_file = ""
  gen_output_file = ""
  label_input_file = ""
  label_output_file = ""

  if len(options) == 0:
      print(
          f"missing --input and --output option \n\nExample Usage: `python extract_diff.py --input=all_result.json --output=output.diff --labelin=label.json --labelout=label.diff`"
      )
      return

  for opt, arg in options:
        if opt in ("i", "--input"):
            gen_input_file = arg
        elif opt in ("o", "--output"):
            gen_output_file = arg
        elif opt in ("li", "--labelin"):
            label_input_file = arg
        elif opt in ("lo", "--labelout"):
            label_output_file = arg

  with open(gen_input_file, "r") as f:
      data = json.load(f)

  with open(gen_output_file, "w", encoding="utf-8") as op:
      for item in data:
          diff = item["diff"]
          op.write(repr(diff)[1:-1] + "\n")

  print(f"Diffs extracted to {gen_output_file}")

  # Copy label in to label out using shutil
  if os.path.isfile(label_input_file):
      shutil.copy(label_input_file, label_output_file)
      print(f"Label file copied to {label_output_file}")
  else:
      print(f"Error: Label input file '{label_input_file}' not found!")
      sys.exit(1)


if __name__ == "__main__":
  main()
