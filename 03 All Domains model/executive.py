from __future__ import division
import csv
import re
import sys
from collections import Counter
from collections import OrderedDict
from pylab import *
from numpy import genfromtxt
#import matplotlib.pyplot as plt


def getDomainNameFromIndex(index):
    #print 'getDomainNameFromIndex'
    inf = open('all_domains.csv','rb')
    contents=inf.readlines()
    line = contents[index].split(',')
    if int(line[0]) != index:
        print 'error.getDomainNameFromIndex.'
        sys.exit()
    domainName=line[1]
    inf.close
    return domainName

def saveFalloutsRecallsAndPrecisionsToFile(fallouts, recalls, precisions, aucstr = ''):
    print 'saveFalloutsRecallsAndPrecisionsToFile'
    data=np.array([fallouts,recalls,precisions])
    np.savetxt('FalloutsRecallsAndPrecisionsWith ' + aucstr + '.csv', data, delimiter=',')
    print 'done saveFalloutsRecallsAndPrecisionsToFile'

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
    close()
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
    p5=plot(t, t2, 'r--')
    p6=plot(recall,precision, 'ro')
#    legend([p2], ['ROC with AUC ' + aucstr + '.png'])
    savefig('Recall-Precision with ' + aucstr + '.png')
    if shouldShow:
        show()
    close()
    print 'done plotting Recall-Precision'


def removeTheBackslashes(infname='ids_with_their_domains_temp.csv'):
    print 'removeTheBackslashes'
    inf=open(infname,'r')
    contents=inf.readlines()
    outf=open(infname[:-9]+'.csv','w')
    writer=csv.writer(outf)
    for i in contents:
        i= i.strip().split(',')
        i = [ j.replace('\\', '') for j in i]
        writer.writerow(i)
    outf.close()
    inf.close()
#removeTheBackslashes()
def getListOfTheColumn(infname='all_domains.csv',col=0):
    inf=open(infname,'r')
    reader=csv.reader(inf,delimiter=',')
    alist=[]
    for i in reader:
        alist.append(i[col])
    inf.close()
    return alist
#print getListOfTheColumn()
def getModelParameters():
    return arange(0.0, 5.0, 0.05).tolist()+[1000]
def getAUC(fpr, recall):
    size=len(recall)
    auc1=abs(0.5*sum([((fpr[i]-fpr[i-1])*(recall[i]+recall[i-1])) for i in range(1,size)]))
    fpr = [1.0]+fpr+[0.0]
    recall = [1.0]+recall+[0.0]
    size=len(recall)
    auc2=abs(0.5*sum([((fpr[i]-fpr[i-1])*(recall[i]+recall[i-1])) for i in range(1,size)]))
    return [auc1,auc2]
def separateGender(infname='training_ids_with_their_domains.csv'):
    print 'separateGender'
    genders=['male','female']
    inf = open(infname,'r')
    reader=csv.reader(inf, delimiter=',')
    writer=dict()
    outf=dict()
    for i in genders:
        outf[i]=open(infname[:-4]+'_'+i+'.csv','w')
        writer[i]=csv.writer(outf[i])
    for i in reader:
        writer[i[1]].writerow(i)
    outf['male'].close()
    outf['female'].close()
    inf.close()
#separateGender()
def getDomainsOfId(idd,fname='test_ids_with_their_domains.csv'): 
    dlist=[]
    inf=open(fname,'r')
    reader=csv.reader(inf,delimiter=',')
    for i in reader:
        if idd == int(i[0]):
            dlist=i[2:]
            continue
    return dlist
def separateTestAndTraining(fname='ids_with_their_domains.csv',endint=0): 
    print 'separateTestAndTraining ',endint
    inf=open(fname,'r')
    outf, writer=dict(), dict()
    reader=csv.reader(inf,delimiter=',')
    nlist=['test','training']
    for i in nlist:
        outf[i]=open(i+'_'+fname,'w')
        writer[i]=csv.writer(outf[i])
    cnt1=0
    cnt2=0
    for i in reader:
        cnt1+=1
        if int(i[0])%10==endint:
            cnt2+=1
            writer[nlist[0]].writerow(i)
        else:
            writer[nlist[1]].writerow(i)
    for i in nlist:
        outf[i].close()
    return float(cnt2)/cnt1 # returns the percentage that corresponds to test set size
#separateTestAndTraining()
def makeTableOfFeatValsNonDom(outfname='TableOfNonDomFeatVals.csv',numOfFeat=1,infname='training_ids_with_their_domains.csv'):
    alist=[[] for i in range(numOfFeat)]
    outf=open(outfname,'w')
    writer=csv.writer(outf)
    for i in range(numOfFeat):
        alist[i]=getListOfTheColumn('ids_with_their_domains.csv',1+i)
        alist[i]=list(set(alist[i]))
        writer.writerow(alist[i])
    outf.close()
    return alist
#makeTableOfFeatValsNonDom()
def makeHistogramNonDom(outfname='HistOfNonDomFeats.csv',numOfFeat=1,infname='training_ids_with_their_domains.csv'):
    print 'makeHistogramNonDom'
    alist=[[] for i in range(numOfFeat)]
    outf=open(outfname,'w')
    writer=csv.writer(outf)
    for i in range(numOfFeat):
        alist[i]=getListOfTheColumn('ids_with_their_domains.csv',1+i)
        alist[i]=Counter(alist[i])
        suma=sum(alist[i].values())
        for j in alist[i].keys():
            alist[i][j]=float(alist[i][j])/suma
        writer.writerow(alist[i].keys())
        writer.writerow(alist[i].values())
    return alist
#print makeHistogramNonDom()
def makeHistOfDoms(infname='training_ids_with_their_domains.csv',outfname='HistOfDoms.csv'):
    print 'makeHistOfDoms'+' '+infname
    outf=open(outfname,'w')
    writer=csv.writer(outf)
    inf=open(infname,'r')
    reader=csv.reader(inf)
    domList=getListOfTheColumn('all_domains.csv',0)
    nDoms=len(domList)
    yesList=OrderedDict()
    noList=OrderedDict()
    for i in domList:
        yesList[i]=0
    cnt=0
    for i in reader:
        cnt+=1
        for j in i[2:]:
            yesList[j]+=1
    for i in domList:
        yesList[i]=yesList[i]/float(cnt)
        noList[i]=1-yesList[i]
    writer.writerow(noList.keys())
    writer.writerow(noList.values())
    writer.writerow(yesList.values())
    inf.close()
    outf.close()
    out=dict()
    out['domIndex']=noList.keys()
    out['no']=noList.values()
    out['yes']=yesList.values()
    print 'done'
    return out
#makeHistOfDoms()
def printTopLifts(topNumber, outFileName, histFileName='HistOfDoms.csv', histMaleFileName='HistOfDoms_male.csv', histFemaleFileName='HistOfDoms_female.csv'):
    print ' print Lifts'
    histData = genfromtxt(histFileName, delimiter = ',')
    histMaleData = genfromtxt(histMaleFileName, delimiter = ',')
    histFemaleData = genfromtxt(histFemaleFileName, delimiter = ',')
#    liftsMale= dict()
#    liftsFemale = dict()
    liftsMaleNo=[0 for i in range(len(histData[0]))]
    liftsMaleYes=[0 for i in range(len(histData[0]))]
    liftsFemaleNo=[0 for i in range(len(histData[0]))]
    liftsFemaleYes=[0 for i in range(len(histData[0]))]
    for i in map(int,histData[0]):
        try:
            if histData[1][i]==0:
                if histMaleData[1][i]==0:
                    liftsMaleNo[i]=1
                else:
                    print 'Error non-zero divided by zero. Domain: ',i
                    sys.exit()
                if histFemaleData[1][i]==0:
                    liftsFemaleNo[i]=1
                else:
                    print 'Error non-zero divided by zero. Domain: ',i
                    sys.exit()
            else:
                liftsMaleNo[i]=histMaleData[1][i]/histData[1][i]
                liftsFemaleNo[i]=histFemaleData[1][i]/histData[1][i]
                
            if histData[2][i]==0:
                if histMaleData[2][i]==0:
                    liftsMaleYes[i]=1
                else:
                    print 'Error non-zero divided by zero. Domain: ',i
                    sys.exit()
                if histFemaleData[2][i]==0:
                    liftsFemaleYes[i]=1
                else:
                    print 'Error non-zero divided by zero. Domain: ',i
                    sys.exit()
            else:
                liftsMaleYes[i]=histMaleData[2][i]/histData[2][i]
                liftsFemaleYes[i]=histFemaleData[2][i]/histData[2][i]
        except ZeroDivisionError:
            print 'error'
            sys.exit()            
        except RuntimeWarning:
            print 'warning-error'
            sys.exit()
            
    domains=map(int,histData[0])
    indicesMaleNo=np.argsort(liftsMaleNo)
    indicesMaleYes=np.argsort(liftsMaleYes)
    indicesFemaleNo=np.argsort(liftsFemaleNo)
    indicesFemaleYes=np.argsort(liftsFemaleYes)
    tp=topNumber
    topLiftsMaleNo = [0 for i in range(tp)]
    topLiftsMaleYes= [0 for i in range(tp)]
    topLiftsFemaleNo= [0 for i in range(tp)]
    topLiftsFemaleYes= [0 for i in range(tp)]
    topDomainsMaleNo= [0 for i in range(tp)]
    topDomainsMaleYes= [0 for i in range(tp)]
    topDomainsFemaleNo= [0 for i in range(tp)]
    topDomainsFemaleYes= [0 for i in range(tp)]
    for i in range(tp):
        topLiftsMaleNo[i]=liftsMaleNo[indicesMaleNo[-(i+1)]]
        topLiftsMaleYes[i]=liftsMaleYes[indicesMaleYes[-(i+1)]]
        topLiftsFemaleNo[i]=liftsFemaleNo[indicesFemaleNo[-(i+1)]]
        topLiftsFemaleYes[i]=liftsFemaleYes[indicesFemaleYes[-(i+1)]]
        topDomainsMaleNo[i]=indicesMaleNo[-(i+1)]
        topDomainsMaleYes[i]=indicesMaleYes[-(i+1)]
        topDomainsFemaleNo[i]=indicesFemaleNo[-(i+1)]
        topDomainsFemaleYes[i]=indicesFemaleYes[-(i+1)]
    
##        topDomainsMaleNo
##        topDomainsMaleYes
##        topDomainsFemaleNo
##        topDomaonsFemaleYes
    
##    print topDomainsMaleNo
##    print topDomainsFemaleYes
##    print topDomainsMaleYes
##    print topDomainsFemaleNo
##
##    print topLiftsMaleNo
##    print topLiftsFemaleYes
##    print topLiftsMaleYes
##    print topLiftsFemaleNo
    topDomainNamesMaleNo = map(getDomainNameFromIndex,topDomainsMaleNo)
    topDomainNamesFemaleYes = map(getDomainNameFromIndex,topDomainsFemaleYes)
    topDomainNamesMaleYes = map(getDomainNameFromIndex,topDomainsMaleYes)
    topDomainNamesFemaleNo = map(getDomainNameFromIndex,topDomainsFemaleNo)
    outf= open(outFileName,'wb')
    writer=csv.writer(outf)
    writer.writerow(['Top Lifts and their Domains: Visited domain given was male'])
    writer.writerow(topDomainNamesMaleYes)
    writer.writerow(topLiftsMaleYes)
    writer.writerow(['Top Lifts and their Domains: Did not visit domain given was male'])
    writer.writerow(topDomainNamesMaleNo)
    writer.writerow(topLiftsMaleNo)
    writer.writerow(['Top Lifts and their Domains: Visited domain given was female'])
    writer.writerow(topDomainNamesFemaleYes)
    writer.writerow(topLiftsFemaleYes)
    writer.writerow(['Top Lifts and their Domains: Did not visit domain given was female'])
    writer.writerow(topDomainNamesFemaleNo)
    writer.writerow(topLiftsFemaleNo)
    outf.close()

def getBayesResults(params, genderDist, infname='test_ids_with_their_domains.csv'):
    print 'getBayesResults'
    inf=open(infname,'r')
    reader=csv.reader(inf,delimiter=',')
    hist=makeHistOfDoms()
    condHist=dict()
    condHist['female']=makeHistOfDoms('training_ids_with_their_domains_female.csv','HistOfDoms_female.csv')
    condHist['male']=makeHistOfDoms('training_ids_with_their_domains_male.csv','HistOfDoms_male.csv')
    cnt=0
    tp,tn,fp,fn,precision,recall,fpr= dict(),dict(),dict(),dict(),OrderedDict(),OrderedDict(),OrderedDict()
    for i in params:
        tp[i],tn[i],fp[i],fn[i],precision[i],recall[i],fpr[i]= 0,0,0,0,0,0,0
    for i in reader:
        bprob=genderDist['male']
        fprob=genderDist['female']
        cnt+=1
        trueGender=i[1]
        try:
            for jk in hist['domIndex']:
                j=int(jk)
                if jk in i[2:]:
                    p=condHist['male']['yes'][j]/hist['yes'][j]
                else:
                    p=condHist['male']['no'][j]/hist['no'][j]
                if p>5:
                    print p, ' dis is a big p.. or small'
                    sys.exit()
                bprob*=p
        except ZeroDivisionError: #no information gainned
            bprob*=1
        except:
            raise
        try:
            for jk in hist['domIndex']:
                j=int(jk)
                if jk in i[2:]:
                    p=condHist['female']['yes'][j]/hist['yes'][j]
                else:
                    p=condHist['female']['no'][j]/hist['no'][j]
                if p>5:
                    print p, ' dis is a big p.. or small'
                    sys.exit()
                fprob*=p
        except ZeroDivisionError: #no information gainned
            fprob*=1
        except:
            raise
        for j in params:
            if bprob>=j*fprob:
                if trueGender=='male':
                    tp[j]+=1
                else:
                    fp[j]+=1
            else:
                if trueGender=='female':
                    tn[j]+=1
                else:
                    fn[j]+=1    
    for i in params:
        tp[i]=tp[i]/float(cnt)
        fp[i]=fp[i]/float(cnt)
        tn[i]=tn[i]/float(cnt)
        fn[i]=fn[i]/float(cnt)
        print 'success: ',tp[i]+tn[i],'param: ',i
        try:
            precision[i]=tp[i]/(tp[i]+fp[i])
            recall[i]=tp[i]/(tp[i]+fn[i])
            fpr[i]=fp[i]/(fp[i]+tn[i])
        except ZeroDivisionError:
            print '!!!the model parameters are too radical!!! Parameter: ',i
            raise
            sys.exit()
    inf.close()
    return precision, recall , fpr    
##
##def plotROC(fpr,recall,aucstr=''):
##    p1=plot(fpr,recall)
##    xlabel('FPR')
##    ylabel('Recall')
##    title('ROC curve')
##    grid(True)
##    t=arange(0,1.1,0.5)
##    p2=plot(t,t,'r--')
##    legend([p2],['ROC with auc='+aucstr+'.png'])
##    savefig('ROC with auc='+aucstr+'.png')
##    show()  



##printTopLifts(10,'TopLifts.csv')
##sys.exit()
removeTheBackslashes('ids_with_their_domains_temp.csv')
auc1=[]
auc2=[]
for i in range(10):
    separateTestAndTraining('ids_with_their_domains.csv',i)
    separateGender('training_ids_with_their_domains.csv')
    gDist=makeHistogramNonDom('HistOfNonDomFeats.csv',1,'training_ids_with_their_domains.csv')[0]
    params=getModelParameters()
    res=getBayesResults(params, gDist, 'training_ids_with_their_domains.csv')
    print getAUC(res[2].values(),res[1].values())
    aa=getAUC(res[2].values(),res[1].values())
    nameString = '00otherNaiveAllDomainsEndInt'+str(i)+'AUC'+str(aa[0])
    plotROC(res[2].values(),res[1].values(), shouldShow = False , aucstr = nameString)
    plotRecallVSPrecision(res[1].values(), res[0].values(), shouldShow = False , aucstr = nameString)
    saveFalloutsRecallsAndPrecisionsToFile(res[2].values(),res[1].values(),res[0].values(), aucstr = nameString)
    auc1.append(aa[0])
    auc2.append(aa[1])
aucc1=average(auc1)
aucc2=average(auc2)
print aucc1,aucc2
printTopLifts(10,'TopLifts.csv')
other=[0.7836644109603883, 0.78409038090151961]
#usual 0.788951664252      0.788952657117


##separateTestAndTraining('ids_with_their_domains.csv',0)
##separateGender('training_ids_with_their_domains.csv')
##gDist=makeHistogramNonDom('HistOfNonDomFeats.csv',1,'training_ids_with_their_domains.csv')[0]
##params=getModelParameters()
##res=getBayesResults(params, gDist, 'training_ids_with_their_domains.csv','bayes_results.csv')
##print getAUC(res[2].values(),res[1].values())
##plotROC(res[2].values(),res[1].values(),str(getAUC(res[2].values(),res[1].values())))





print 'end'
#this is the end of the file.


