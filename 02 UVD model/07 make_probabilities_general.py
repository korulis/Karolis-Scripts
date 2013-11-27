'''very dirty and nonoptimal code'''

import csv
import sys
from math import log
import kmodule
from collections import Counter
from pylab import * 


def plotROC(fallout, recall, shouldShow = True , aucstr = ''):
    print 'plotting ROC'
    p1 = plot(fallout, recall)
    xlabel('Fallout')
    ylabel('Recall')
    title('ROC with ' + aucstr)
    grid(True)
    t=arange(0, 1.1, 0.5)
    p2=plot(t, t, 'r--')
    p3=plot(fallout, recall, 'ro')
#    legend([p2], ['ROC with AUC ' + aucstr + '.png'])
    savefig('ROC with ' + aucstr + '.png')
    if shouldShow:
        show()
    print 'done plotting ROC'

def plotRecallVSPrecision(recall, precision, shouldShow = True , aucstr = ''):
    print 'plotting Recall-Precision'
    p4 = plot(recall, precision)
    xlabel('Recall')
    ylabel('Precision')
    title('Recall-Precision with ' + aucstr)
    grid(True)
    t=[0, 0.5, 1]
    t2=[1, 0.5, 0]
    print 'asdadsfadsfds'
    p5=plot(t, t2, 'r--')
    p6=plot(recall,precision, 'ro')
#    legend([p2], ['ROC with AUC ' + aucstr + '.png'])
    savefig('Recall-Precision with ' + aucstr + '.png')
    if shouldShow:
        show()
    print 'done plotting Recall-Precision'

def saveFalloutsRecallsAndPrecisionsToFile(fallouts, recalls, precisions, aucstr = ''):
    print 'saveFalloutsRecallsAndPrecisionsToFile'
    data=np.array([fallouts,recalls,precisions])
    np.savetxt('FalloutsRecallsAndPrecisionsWith ' + aucstr + '.csv', data, delimiter=',')
    print 'done saveFalloutsRecallsAndPrecisionsToFile'


#making histograms here
val_table=kmodule.make_value_table('cookies_train.csv')
indi_hist=kmodule.make_histograms(val_table)
#print indi_hist 
print '<-- full list of indi histograms'
labelfile=open('cookie_table_of_labels.csv','r')
all_labels = labelfile.readlines()
labelfile.close()
gender_labels=all_labels[1].split(',')
gender_labels[-1]=gender_labels[-1][:-2]
#gender_labels=gender_labels[:-1]
cond_hists=dict(zip(gender_labels,range(len(gender_labels))))
for i in gender_labels:
    val_table=kmodule.make_value_table('cookies_gender_'+i+'.csv')
    cond_hists[i]=kmodule.make_histograms(val_table) 
    #print cond_hists[i]
    print '<--histograms for '+i
#-----------------
    
intestfile=open('cookies_test.csv','r')
testcontents = intestfile.readlines()
intestfile.close()
data_dim =len(testcontents[0].split(','))
foo=False
indi_hist2=dict(zip(['male','female'],[0,0]))
indi_hist2['male']=indi_hist[1]['male']/indi_hist[1]['female']+indi_hist[1]['male']
indi_hist2['female']=indi_hist[1]['female']/indi_hist[1]['female']+indi_hist[1]['male']
#model_coef=[1.1**i for i in range(-4,5)]
#model_coef=[i*1/float(8) for i in range(24+1)]
model_coef=[0]+[1.03**i for i in range(-50,50)]+[1000]
maletpcounter=dict(zip(model_coef,[0 for i in range(len(model_coef))]))
femaletncounter=dict(zip(model_coef,[0 for i in range(len(model_coef))]))
fail_counter=dict(zip(model_coef,[0 for i in range(len(model_coef))]))
for l in model_coef:
    guessdist=Counter()
    truedist=Counter()
    full_counter=0
    for i in testcontents:
        dataline = i.split(',')
        dataline[-1]=dataline[-1][:-2]
        if dataline[2]=='unknown':
            print 'unexpected gender in line', i
            sys.exit()
        full_counter+=1
        prob_prods=dict(zip(gender_labels,len(gender_labels)*[1]))
        prob_anti=dict(zip(gender_labels,len(gender_labels)*[1]))
        prob_list=dict(zip(gender_labels,[[] for ii in range(len(gender_labels))]))
        guess_gender='spam'
        true_gender=dataline[2]
        for j in gender_labels:
            for k in range(1,data_dim):
                #7 livas, 3lieavas 5,6
                #7057620,55,female,529,5,10,0,0.5,go.tv2.dk,samvirke.dk,www.degulesider.dk
                nameString = 'NaiveUvd4_8_9_10Auc'
                if k in [1,2]+[3,5,6,7]: continue
                try:
                    if cond_hists[j][k-1][dataline[k]]==0 and indi_hist[k-1][dataline[k]]==0:
                        prob_prods[j]*=1
                        prob_list[j].append(1)
                    else:
                        prob_prods[j]*=cond_hists[j][k-1][dataline[k]]/indi_hist[k-1][dataline[k]]                    
                        prob_list[j].append(cond_hists[j][k-1][dataline[k]]/indi_hist[k-1][dataline[k]])
                except KeyError:
                    #print 'keyError. Tru gender ',true_gender
                    print ' wtf keyError. Tru gender ',true_gender,j,full_counter
                    print dataline
                    print dataline[k]
                    print cond_hists[j][k-1]
                    print gender_labels
                    prob_prods[j]*=0
                    prob_list[j].append(0)
                    foo=True
                    sys.exit() #temporary
                except:
                    print cond_hists[j][k-1][dataline[k]],indi_hist[k-1][dataline[k]]
                    print k
                    print dataline[k]
                    print indi_hist[k-1]
                    print '########### nezinoma klaida eiluteje',i
                    raise
            prob_prods[j]*=indi_hist2[j]
            prob_list[j].append(indi_hist2[j])

        for j in gender_labels:
            if foo==False and prob_prods[j]==0:
                pass
        #print prob_prods['male'],' and ', prob_prods['female'], ' and ' , true_gender
        if prob_prods['male']>=l:
            guess_gender='male' 
        else:
            guess_gender='female'
        if guess_gender==true_gender:
            if true_gender=='male':
                maletpcounter[l]+=1
            elif true_gender=='female':
                femaletncounter[l]+=1
            else:
                print  'Error. true gender not in gender list.'
                sys.exit()
        else:
            fail_counter[l]+=1
##        print ' '
##        print full_counter,' fullcounter'
##        print maletpcounter[l]+femaletncounter[l], 'success counter'
##        print fail_counter[l],' fail_counter'
            
        guessdist[guess_gender]+=1
        truedist[true_gender]+=1
    print ' '
    print l, ' l here'
    print full_counter,' full counter'
    print maletpcounter[l]+femaletncounter[l],' success counter'
    print fail_counter[l],' fail counter'
    print float(maletpcounter[l]+femaletncounter[l])/full_counter, '<-- success rate'
    print guessdist,' spejimas'
    print truedist,'tikrasis'


print truedist['male']
#recall=dict(zip(model_coef,[0 for i in range(len(model_coef))]))
##fpr=[ float(truedist['female']-femaletncounter[i])/(truedist['female']) for i in model_coef]
##recall=[ float(maletpcounter[i])/(truedist['male']) for i in model_coef]
##precision=[ float(maletpcounter[i])/(truedist['female']-femaletncounter[i]+maletpcounter[i]) for i in model_coef]

fpr=[]
recall=[]
precision=[]
for i in range(len(model_coef)):
    fpr.append(float(truedist['female']-femaletncounter[model_coef[i]])/truedist['female'])
    recall.append(float(maletpcounter[model_coef[i]])/truedist['male'])
    if truedist['female']-femaletncounter[model_coef[i]]+maletpcounter[model_coef[i]]!=0:
        precision.append(float(maletpcounter[model_coef[i]])/(truedist['female']-femaletncounter[model_coef[i]]+maletpcounter[model_coef[i]]))
    else:
        precision.append(1)
                                                   
auc=abs(0.5*sum([((fpr[i]-fpr[i-1])*(recall[i]+recall[i-1])) for i in range(1,len(model_coef))]))
nameString = nameString+str(auc)
plotROC(fpr, recall,True,  nameString)
plotRecallVSPrecision(recall, precision, True, nameString )
saveFalloutsRecallsAndPrecisionsToFile(fpr, recall,precision, nameString)



    
print 'AUC'
print auc
##print 'recall'
##for i in range(len(model_coef)):
##    print recall[i]
##print 'precision'
##for i in range(len(model_coef)):
##    print precision[i]
##print ' FPR'
##for i in range(len(model_coef)):
##    print fpr[i]    
##print 'coefficient L'
##for i in range(len(model_coef)):
##    print model_coef[i]
###print recall, ' recall'
###print precision, ' precision'
##print 'AUC'
##print auc
