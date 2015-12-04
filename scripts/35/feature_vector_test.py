#!/usr/bin/python

import subprocess
import sys
import os
import fileinput

def run_mallet_input():
	os.system("rm -r ./full_phase_2_test/*")
	for x in os.listdir("./full_phase_1_test/"):
		output = subprocess.check_output("../../../../../../../Downloads/mallet-2.0.7/bin/mallet import-file --input ./full_phase_1_test/" + x + " --remove-stopwords --print-output > ./full_phase_2_test/" + x, shell=True)
		f = open("./full_phase_2_test/" + x,"r")
		lines = f.readlines()
		f.close()
		f = open("./full_phase_2_test/" + x,"w")
		bais = "Positive"
		for line in lines:
			if "name: " not in line:
				if "target: " not in line:
					if "input: " in line:
						line = line.replace("input: ","")
						templine = line
						templine = templine.split('=')
						if line != "" and len(templine) > 1:
							value = float(templine[1].strip())
							templine = line
							templine = templine.split('(')
							templine = templine[1].split(')')
							feature = templine[0]
							if bais == "Positive":
								line = "1 " + str(int(feature)+1) + ":" + str(value) + " "
							else:
								line = "-1 " + str(int(feature)+1) + ":" + str(value) + " "
							f.write(line)
					else:
						if line != "\n":
							templine = line
							templine = templine.split('=')
							value = float(templine[1].strip())
							templine = line
							templine = templine.split('(')
							templine = templine[1].split(')')
							feature = templine[0]
							line = str(int(feature)+1) + ":" + str(value) + " " 
						f.write(line)
			else:
				if "Positive" in line:
					bais = "Positive"
				else:
					bais = "Negative"
		f.close()

if __name__ == "__main__":
	run_mallet_input()

