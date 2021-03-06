import csv
import sys
from math import log
import kmodule



def write_lifts_to_file(valtable,out_filename='male_lifts.csv'):
    print 'Working on writing to file '+out_filename+' ...'
    outf=open(out_filename,'w')
    writer=csv.writer(outf)
    for i in valtable:
        list1,list2=[],[]
        for j in i:
            list1.append(j)
            list2.append(i[j])    
        writer.writerow(list1)
        writer.writerow(list2)
    outf.close()
    print 'done'


val_table=kmodule.make_value_table('cookies_train.csv')
indi_hist=kmodule.make_histograms(val_table)
##suma1,suma2=0,0
##max1,max2=0,0


#female histogram load
val_table=kmodule.make_value_table('cookies_gender_female.csv')
hist2=kmodule.make_histograms(val_table)
#male histogram load
val_table=kmodule.make_value_table('cookies_gender_male.csv')
hist1=kmodule.make_histograms(val_table)
#make lift tables for gender
male_lifts, female_lifts=[],[]
for i in range(len(indi_hist)):
    male_lifts.append(dict())
    female_lifts.append(dict())
for i in range(len(indi_hist)):
    sumele=0
    for j in indi_hist[i]:
        sumele+=hist2[i][j]+hist1[i][j]+indi_hist[i][j]
        if indi_hist[i][j]==0:
            if hist2[i][j]!=0 or hist1[i][j]!=0:
                print 'Error. Indipendent histogram value is zero while gender histogram value is not'
                sys.exit()
            else:
                male_lifts[i][j]=1
                female_lifts[i][j]=1
        else:
            male_lifts[i][j]= hist1[i][j]/indi_hist[i][j]
            female_lifts[i][j]=hist2[i][j]/indi_hist[i][j]
    print sumele, '<-very important. Must equal 3.0. Collumn:',i
#write lifts to file
write_lifts_to_file(male_lifts,'male_lifts.csv')
write_lifts_to_file(female_lifts,'female_lifts.csv')

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
