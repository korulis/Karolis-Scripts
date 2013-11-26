import csv
import os
import re
import sys
#cookies

labelfile=open('cookie_table_of_labels.csv','r')
all_labels = labelfile.readlines()
labelfile.close()
gender_labels=all_labels[1].split(',')
gender_labels[-1]=gender_labels[-1][:-2]
outf=dict(zip(gender_labels,range(len(gender_labels))))
writer=dict(zip(gender_labels,range(len(gender_labels))))
for i in gender_labels:
    outf[i]=open('cookies_gender_'+i+'.csv','w')
    writer[i] = csv.writer(outf[i])

for filename in os.listdir('.'):
    if re.match('cookies_train.csv',filename):
        infile = open(filename,'r')
        reader = csv.reader(infile, delimiter = ',')
        
        for row in reader:
            writer[row[2]].writerow(row)
        infile.close()

for i in gender_labels:
    outf[i].close()
print 'done'
