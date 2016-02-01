rm output.txt
for ((x=0; x < $1; x++))
do
	echo $x
	bash test.sh >> output.txt
done
rm output.csv
python parser.py
