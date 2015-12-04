import json
import csv

questions = {}
tags = json.load(open("../tags.json","rb"))
for i in xrange(1,6):
	with open(str(i)+".csv","rb") as quesdata:
		a = [x for x in csv.reader(quesdata)]
		for j in xrange(1,len(a)):
			ques_id = a[j][0]
			rel_tags = a[j][16]
			rel_tags = rel_tags.split("<")
			rel_tags =  [ x[:-1] for x in rel_tags[1:]]
			questions[ques_id] = []
			for k in rel_tags:
				if tags.has_key(k):
					questions[ques_id].append(tags[k])

with open("question_tags.json","wb") as final:
	json.dump(questions,final)
		
