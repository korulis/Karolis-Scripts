import csv
import sys
from collections import Counter
from math import log
import kmodule

#cookie Id, age, gender, full time, uvd_bin, domain_bin, url_int_bin, domain_int_bin, dom1, dom2, dom3
lookupcat = ['cookieID', 'age', 'gender', 'fulltime', 'uvd_bin', 'domain_bin', 'url_int_bin','domain_int', 'dom1', 'dom2', 'dom3']
##45300226	55	female	151	5	5	5	1.0	afe2.specificclick.net	www.tvguide.dk	ekstrabladet.dk
##54389116	55	female	143	5	10	10	1.5	ams1.ib.adnxs.com	entertainment.dk.msn.com	d5p.de17a.com

def write_lifts_to_file(valtable,out_filename='male_lifts.csv', nTop = 10):
    print 'Working on writing to file '+out_filename+' ...'
    outf=open(out_filename,'w')
    writer=csv.writer(outf)
    list1, list2,list3, list4=[],[],[],[]    
    cnt=0
    for i in valtable:
        cnt += 1
        if cnt == 1 or cnt == 2 or cnt == 9 or cnt == 8: # dont take gender, age, and dom2 and dom3 cuz its all in dom1
            continue
        for j in i:
            list1.append(lookupcat[cnt]+'_'+j)
            list2.append(i[j])    
    indices = [i[0] for i in sorted(enumerate(list2), key=lambda x:x[1])]
    indices.reverse()
    for i in range(nTop):
        list3.append(list1[indices[i]])
        list4.append(list2[indices[i]])
    for i in range(nTop):
        writer.writerow([str(i),list3[i],list4[i]])
    outf.close()
    print 'done'



#making histograms here
val_table=kmodule.make_value_table('cookies_train.csv')
indi_hist=kmodule.make_histograms(val_table)

#female histogram load
val_table=kmodule.make_value_table('cookies_gender_female.csv')
hist2=kmodule.make_histograms(val_table)
#male histogram load
val_table=kmodule.make_value_table('cookies_gender_male.csv')
hist1=kmodule.make_histograms(val_table)
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
write_lifts_to_file(male_lifts,'top_male_lifts.csv')
write_lifts_to_file(female_lifts,'top_female_lifts.csv')

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
