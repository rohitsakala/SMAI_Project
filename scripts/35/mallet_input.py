#!/usr/bin/python

import csv
import re
import os
import subprocess

def create_formatted_input_mallet():
	os.system("rm -r ./full_phase_1/*")
	for x in range(1,6):
		with open('../data_positive/' + str(x) + '.csv', 'rb') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				f = open('./full_phase_1/'+ row['ID'],"a")
				row['Body'] = row['Body'].replace('\n',' ')
				row['Body'] = row['Body'].replace('\r','')
				row['Body'] = row['Body'].rstrip()
				re.sub(r'\W+', '', row['Body'])
				lines = ["Positive " + row['ID'] + " " + row['Body'] + "\n"]
				f.writelines(lines)
				f.close()
	for x in range(1,35):
		with open('../data_negative/' + str(x) + '.csv', 'rb') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				f = open('./full_phase_1/'+ row['Id'],"a")
				row['Body'] = row['Body'].replace('\n',' ')
				row['Body'] = row['Body'].replace('\r','')
				row['Body'] = row['Body'].rstrip()
				re.sub(r'\W+', '', row['Body'])
				lines = ["Negative " + row['Id'] + " " + row['Body'] + "\n"]
				f.writelines(lines)
				f.close()
			
if __name__ == "__main__":
	create_formatted_input_mallet()
