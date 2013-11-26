import csv
import sys
from collections import Counter

#7057620,55,female,529,5,10,0,0.5,go.tv2.dk,samvirke.dk,www.degulesider.dk


    
def make_histograms(val_table):
    print 'Making histograms...'
    infile = open('cookie_table_of_labels.csv','r')
    contents=infile.readlines()
    data_dimension=len(val_table)
    histograms = [[] for i in range(data_dimension-1) ]
    val_table[-1]=[val_table[-1][i][:-2] for i in range(len(val_table[-1]))]
    all_domains=val_table[-1]+val_table[-2]+val_table[-3]
    val_table[-1]= all_domains
    val_table[-2]= all_domains
    val_table[-3]= all_domains
    if (len(contents)+1)!=data_dimension:
        print 'Make histograms: invalid table dimensions', data_dimension, len(contents)
        sys.exit()
    for j in range(data_dimension-1):
        histograms[j]=dict()
        curline=contents[j].split(',')
        curline[-1]=curline[-1][:-2]
        cnt=Counter(val_table[j+1])
        suma=sum(cnt.values())
        if j =='compaq.dk.msn.com':
            print 'bazinga'
                
        for i in curline:
            if i =='compaq.dk.msn.com':
                print 'bazinga yeah'
            if i in cnt.keys():
                histograms[j][i]=float(cnt[i])/suma
            else:
                histograms[j][i]=0
    infile.close()
    print 'Done making histograms'   
    return histograms

def make_value_table(in_filename='cookies_train.csv',out_filename='somename.csv'):
    print ' Making value table from ' ,in_filename
    inf=  open(in_filename, 'r')
    contents = inf.readlines()
    inf.close()
    data_dimension =len(contents[0].split(','))
    val_table = [[] for i in range(data_dimension) ]
    print val_table, 'empty val_table', data_dimension
    for i in contents:
        cur_line =i.split(',')
        for j in range(data_dimension):
            val_table[j].append(cur_line[j])
    val_table = [sorted(val_table[i]) for i in range(data_dimension)]
    #print set(val_table[0]),' ciaprintas'
    print ' Done making value table'
    return val_table
