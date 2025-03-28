# codellama:7b -> temp 0
echo "Running codellama:7b for py with temp=0.0\n"
python generate.py --model=codellama:7b --lang=py --temp=0.0
echo "Completed run-> codellama:7b for py with temp=0.0\n"

echo "Running codellama:7b for java with temp=0.0\n"
python generate.py --model=codellama:7b --lang=java --temp=0.0
echo "Completed run-> codellama:7b for java with temp=0.0\n"

echo "Running codellama:7b for js with temp=0.0\n"
python generate.py --model=codellama:7b --lang=js --temp=0.0
echo "Completed run-> codellama:7b for js with temp=0.0\n"
