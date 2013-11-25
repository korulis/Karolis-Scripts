import csv
import sys
from collections import Counter
from math import log

def write_lifts_to_file(valtable,out_filename='male_lifts.csv', nTop = 10):
    print 'Working on writing to file '+out_filename+' ...'
    outf=open(out_filename,'w')
    writer=csv.writer(outf)
    list1, list2,list3, list4=[],[],[],[]    
    for i in valtable:
        print i
        for j in i:
            list1.append(j)
            list2.append(i[j])    
    indices = [i[0] for i in sorted(enumerate(list2), key=lambda x:x[1])]
    indices.reverse()
    for i in range(nTop):
        list3.append(list1[indices[i]])
        list4.append(list2[indices[i]])
    print list1
    print list2
    print list3
    print list4
    sys.exit()
    writer.writerow(list3)
    writer.writerow(list4)
    outf.close()
    print 'done'

def write_label_table_to_file(label_table,out_filename):
    print 'Working on writing to file '+out_filename+' ...'
    outf= open(out_filename,'w')
    writer = csv.writer(outf)
    for i in range(1,len(label_table)):
        writer.writerow(label_table[i])
    outf.close()
    print 'Done writing to'+out_filename

def make_histograms(val_table):
    print 'Making histograms...'
    infile = open('cookie_table_of_labels.csv','r')
    contents=infile.readlines()
    data_dimension=len(val_table)
    histograms = [[] for i in range(data_dimension-1) ]
    val_table[-1]=[val_table[-1][i][:-2] for i in range(len(val_table[-1]))]
    if (len(contents)+1)!=data_dimension:
        print 'Make histograms: invalid table dimensions'
        sys.exit()
    for j in range(data_dimension-1):
        histograms[j]=dict()
        curline=contents[j].split(',')
        curline[-1]=curline[-1][:-2]
        cnt=Counter(val_table[j+1])
        suma=sum(cnt.values())
        
        for i in curline:
            if i in cnt.keys():
                histograms[j][i]=float(cnt[i])/suma
            else:
                histograms[j][i]=0
    infile.close()
    print 'Done making histograms'   
    return histograms

def make_value_table(in_filename='cookies_learn.csv',out_filename='somename.csv'):
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

#making histograms here
val_table=make_value_table('cookies_learn.csv')
indi_hist=make_histograms(val_table)
##print 'prg 1 #############################################################'
##print indi_hist, '<-- full list of indi histograms'
#-----------------


#female histogram load
val_table=make_value_table('cookies_gender_female.csv')
hist2=make_histograms(val_table)
#male histogram load
val_table=make_value_table('cookies_gender_male.csv')
hist1=make_histograms(val_table)
#make lifta tables for gender
male_lifts, female_lifts=[],[]
for i in range(len(indi_hist)):
    male_lifts.append(dict())
    female_lifts.append(dict())
for i in range(len(indi_hist)):
    for j in indi_hist[i]:
        male_lifts[i][j]= hist1[i][j]/indi_hist[i][j]
        female_lifts[i][j]=hist2[i][j]/indi_hist[i][j]
#write lifts to file
write_lifts_to_file(male_lifts,'male_lifts.csv')
write_lifts_to_file(female_lifts,'female_lifts.csv')

##print indi_hist[1]['male']+indi_hist[1]['female']
##print 'male browser'   
##for i in indi_hist[-1]:
##    print male_lifts[-1][i], i
##print 'male os'   
##for i in indi_hist[-2]:
##    print male_lifts[-2][i], i
##print 'female browser'   
##for i in indi_hist[-1]:
##    print female_lifts[-1][i], i
##print 'female os'   
##for i in indi_hist[-2]:
##    print female_lifts[-2][i], i
#operamobile and safari, opera
#iphone and android, blackberry chromeos linux, rest.

#male: Opera,SeaMonkey,unknown,ChromeiOS
#male:BlackberryPlaybook,ChromeOS,Linux
#female: Opera Mobile, Safari
#female: IPhone,Android,

###-----------------
##labelfile=open('cookie_table_of_labels.csv','r')
##all_labels = labelfile.readlines()
##labelfile.close()
##gender_labels=all_labels[1].split(',')
##gender_labels[-1]=gender_labels[-1][:-2]
##gender_labels=gender_labels[:-1]
##cond_hists=dict(zip(gender_labels,range(len(gender_labels))))
##for i in gender_labels:
##    val_table=make_value_table('cookies_gender_'+i+'.csv')
##    cond_hists[i]=make_histograms(val_table) 
##    print cond_hists[i], '<--histograms for '+i
###-----------------
