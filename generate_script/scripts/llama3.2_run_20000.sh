# llama3.2:3b -> temp 0
echo "Running llama3.2:3b for py with temp=0.0\n"
python generate.py --model=llama3.2:3b --lang=py --temp=0.0
echo "Completed run-> llama3.2:3b for py with temp=0.0\n"

echo "Running llama3.2:3b for java with temp=0.0\n"
python generate.py --model=llama3.2:3b --lang=java --temp=0.0
echo "Completed run-> llama3.2:3b for java with temp=0.0\n"

echo "Running llama3.2:3b for js with temp=0.0\n"
python generate.py --model=llama3.2:3b --lang=js --temp=0.0
echo "Completed run-> llama3.2:3b for js with temp=0.0\n"
