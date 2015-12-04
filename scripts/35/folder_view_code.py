#!/usr/bin/python

import subprocess
import sys
import csv
import os
import re
import fileinput

def folder_view():
	os.system("rm -r ./global/*")
	for x in range(1,6):
		with open('../data_positive/' + str(x) + '.csv', 'rb') as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				if not os.path.exists('./global/' + row['ID']):
    					os.makedirs('./global/' + row['ID'])
				f = open('./global/'+ row['ID'] + '/' + row["Id"],'w+')
				row['Body'] = row['Body'].replace('\n',' ')
				row['Body'] = row['Body'].replace('\r','')
				row['Body'] = row['Body'].rstrip()
				re.sub(r'\W+', '', row['Body'])
				lines = row['Body']
				f.writelines(lines)
				f.close()

if __name__ == "__main__":
	folder_view()
