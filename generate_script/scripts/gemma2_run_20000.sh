# gemma2 9b -> temp 0
echo "Running gemma2:9b for py with temp=0.0\n"
python generate.py --model=gemma2:9b --lang=py --temp=0.0
echo "Completed run-> gemma2:9b for py with temp=0.0\n"

echo "Running gemma2:9b for java with temp=0.0\n"
python generate.py --model=gemma2:9b --lang=java --temp=0.0
echo "Completed run-> gemma2:9b for java with temp=0.0\n"

echo "Running gemma2:9b for js with temp=0.0\n"
python generate.py --model=gemma2:9b --lang=js --temp=0.0
echo "Completed run-> gemma2:9b for js with temp=0.0\n"
