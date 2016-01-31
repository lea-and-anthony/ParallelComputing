@echo off
del output.txt
for /l %%x in (1, 1, %1) do (
	echo %%x
	cmd /C test.bat >> output.txt
)
del output.csv
python parser.py
rem 2>&1
