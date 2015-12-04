import csv
import json
a = open("../tags.csv","rb")
data = csv.reader(a)
data = [x for x in data]
tags = {}
for x in data[1:1001]:
	tags[x[1]] = x[0]
print len(tags.keys())
with open("../tags.json","wb") as to_json:
	json.dump(tags,to_json)

