for i in \
    llama3.2:1b \
    llama3.2:3b \
    llama3.1:8b
do
    ollama run $i
    python generate.py --m=$i
    ollama stop $i

done