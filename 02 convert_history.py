import csv
import os
import re
import urlparse
import sys

def get_domain(urll):
    return urlparse.urlparse(urll).hostname


outf=open('converted_history.csv','w')
writer = csv.writer(outf)
for filename in os.listdir('.'):
    if re.match('cookie_history_1.txt',filename):
        infile = open(filename,'r')
        reader = csv.reader(infile, delimiter = '\t')
        print filename
        counter=0
        for row in reader:
            counter+=1
            user_id = abs(int(row[0]))
            url_time = row[1]
            try:
                domain = get_domain(row[2])
            except IndexError:
                print counter
                print row
                continue
            url= row[2]
            writer.writerow([user_id, url_time, domain, url])
        infile.close()
                        

outf.close()
print 'done'
