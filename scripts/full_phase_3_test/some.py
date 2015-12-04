#!/usr/bin/python

import csv
import re
import os
import subprocess

def average():
	accuracy = []
	for x in os.listdir("./full_phase_2_test/"):
		with open(x) as f:
    		for line in f: # for each question
    			listoftags = {}
    			fobject = open('temp','w')
    			fobject.write(line)
    			fobject.close()
    			for y in os.listdir("./full_phase_3/"): # for each model
    				similarity = os.popen("../../../../../../../Downloads/svm_light/./svm_classify ./temp ./full_phase_3/" + y + " ./full_phase_3_test/" + y + " | sed -n 4p | awk '{print $5}' | cut -d '%' -f 1 ").read()
					similarity = similarity.strip()
					accuracy.append(float(similarity))
	print reduce(lambda x, y: x + y, accuracy) / len(accuracy)

if __name__ == "__main__":
	average()