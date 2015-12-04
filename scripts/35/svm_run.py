#!/usr/bin/python

import csv
import re
import os
import subprocess

def create_models():
	os.system("rm -r ./full_phase_3/*")
	for x in os.listdir("./full_phase_2/"):
		os.system("../../../../../../../Downloads/svm_light/./svm_learn ./full_phase_2/" + x + " ./full_phase_3/" + x)
			
if __name__ == "__main__":
	create_models()
