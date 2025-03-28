# llama2:70b -> temp 0
echo "Running llama2:70b for py with temp=0.0\n"
python generate.py --model=llama2:70b --lang=py --temp=0.0
echo "Completed run-> llama2:70b for py with temp=0.0\n"

echo "Running llama2:70b for java with temp=0.0\n"
python generate.py --model=llama2:70b --lang=java --temp=0.0
echo "Completed run-> llama2:70b for java with temp=0.0\n"

echo "Running llama2:70b for js with temp=0.0\n"
python generate.py --model=llama2:70b --lang=js --temp=0.0
echo "Completed run-> llama2:70b for js with temp=0.0\n"
