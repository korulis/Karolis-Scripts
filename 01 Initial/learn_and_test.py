import csv
import os
import re
#cookies
outf_test = open('cookies_test.csv','w')
writer_test = csv.writer(outf_test)
outf_learn = open('cookies_learn.csv','w')
writer_learn = csv.writer(outf_learn)
for filename in os.listdir('.'):
    if re.match('converted_cookies.csv',filename):
        infile = open(filename,'r')
        reader = csv.reader(infile, delimiter = ',')
        
        for row in reader:
            user_id = abs(int(row[0]))
            if user_id%10 == 0 :
                writer_test.writerow(row)
            else:
                writer_learn.writerow(row)    
        infile.close()

outf_test.close()
outf_learn.close()
        
#history
outf_test = open('history_test.csv','w')
writer_test = csv.writer(outf_test)
outf_learn = open('history_learn.csv','w')
writer_learn = csv.writer(outf_learn)
for filename in os.listdir('.'):
    if re.match('converted_history.csv',filename):
        infile = open(filename,'r')
        reader = csv.reader(infile, delimiter = ',')
        
        for row in reader:
            user_id = abs(int(row[0]))
            if user_id%10 == 0 :
                writer_test.writerow(row)
            else:
                writer_learn.writerow(row)    
        infile.close()

outf_test.close()
outf_learn.close()

