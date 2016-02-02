import csv

items = [[], [], [], [], []]

with open("output.txt", "rb") as infile, open("output.csv", "wb") as outfile:
	i = -1
	for line in infile:
		if line.find('Prediction for image') != -1 :
			number = line.split(' : ')
			items[i].append(float(number[1]))
		
		if line.find('------') != -1 :
			i = (i + 1) % 5
	
	newline = 'Original version,'
	for number in items[0]:
		newline = newline + str(number) + ','
	
	newline = newline + '\n'
	outfile.writelines(newline)
	
	newline = 'Modified version,'
	for number in items[1]:
		newline = newline + str(number) + ','
	
	newline = newline + '\n'
	outfile.writelines(newline)
	
	newline = 'Cuda version,'
	for number in items[2]:
		newline = newline + str(number) + ','
	
	newline = newline + '\n'
	outfile.writelines(newline)
	
	newline = 'Cuda optimized version,'
	for number in items[3]:
		newline = newline + str(number) + ','
	
	newline = newline + '\n'
	outfile.writelines(newline)
	
	newline = 'Cuda optimized no host version,'
	for number in items[4]:
		newline = newline + str(number) + ','
	
	newline = newline + '\n'
	outfile.writelines(newline)