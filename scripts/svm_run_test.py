#!/usr/bin/python

import csv
import re
import os
import subprocess

def calculate_accuracy():
	for x in os.listdir("./full_phase_2_test/"):
		os.system("../../svm_light/./svm_classify ./full_phase_2_test/" + x + " ./full_phase_3/" + x + " ./full_phase_3_test/" + x )
			
if __name__ == "__main__":
	calculate_accuracy()
