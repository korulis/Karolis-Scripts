import csv
import sys

outf = open('uvd_cookies.csv','w')
writer = csv.writer(outf)
inf=  open('uvd_data.csv', 'r')
contents = inf.readlines()
inf.close()
for i in contents:
    cur_line =i.split('\t')
    cur_line[-1]= cur_line[-1][:-2]
    writer.writerow(cur_line)
outf.close()
print 'done'
