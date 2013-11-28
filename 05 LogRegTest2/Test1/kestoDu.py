import numpy
import random
import math
import cPickle
import sys
import csv

#Karolis: perkleiau funkciju aprasymus i failo pradzia ir sukuriau funkcija kestsRoutine()


    



def kestsRoutine(absCriterion = 0.0000001, initAlpha = 0.1, initTheta = 0.5, outFileName ='kests_regression_weights.csv'):
        
    coffs_len = len(list(csv.reader(open('all_domains.csv', 'rb') ) ) ) + 1


    file = open('regression_train.b', 'rb')
    data = cPickle.load(file)
    file.close()
    """
    data - formatu
    [ [cookie_id, 'female', [0,1,4,62,234]],
    [cookie_id, 'male', [2, 4, 35, 132],
    ...
    ]
    """
    #Karolis: pakeiciu klasifier i 'male'
    classifier = 'male'
    # urls = 1..2795
    #
    data_len = float(len(data))
    #how many features you have + 1 for theta0
    #Karolis: pas mane yra 1505 skirtingu Domenu
    #coffs_len = 1505 + 1
    #Karolis: uzkomentuoju
    #coffs_len = 7443 + 1

    #Karolis: padarysiu cia, kad pradetume nuo coffs=initTheta    
    #initial vector of coefficients
    ##coffs = []
    ##for i in range(coffs_len):
    ##    coffs.append (random.random()/10 )
    ##coffs[0] = 1
    coffs = []
    for i in range(coffs_len):
        coffs.append (initTheta)
    coffs[0] = initTheta

    #Karolis:... gaunasi kad pas tave xo gali buti susumuotas 2 kartus.
    def h (coffs, data_point):
        sum = coffs[0]
        for x in data_point[2]:
    #Karolis: klaida
    #        sum += coffs[x[0]] 
            sum += coffs[x] 
            #sum += coffs[x[0]] * x[1]
        return 1.0 / ( 1.0 + math.exp(-sum))

    #ar reguliarizacija veikia - nesu tikras, kazkaip itartinai ten budavo.
    def regularization(coffs):
        sum = 0.0
        for i in coffs:
            sum += abs(i)
        return sum


    def cost_function(coffs):
        sum = 0
        #alphaR = 1/data_len
        alphaR = 1
        for i in data:
            if i[1] == classifier:
                sum += math.log(h(coffs, i))
            else:
                try:
                    #print i
                    #print h(coffs, i)
                    sum += math.log(1 - h(coffs, i))
                except ValueError:
                    print math.log(1 - h(coffs, i))
                    sys.exit()
        sum /= data_len
        #nutrink sum -= .... ir panaikinsi reguliaraizacija, tada turetu kazkiek veikt.
    #Karolis: laikinai isimu reguliarizacija, nes nera reguliarizacijos gradiento impementacijos siame kode
    #    sum -= regularization(coffs) * alphaR
        #print 'regularization: ', regularization(coffs) * alphaR
    #Karolis: panashu kad cia truksta sum*= -1, tai prirasysiu.
        sum*= -1
        return sum

    #calculating partial derrivatives.
    def update_coffs(coffs):
        coffs_n = [0] * coffs_len
    #Karolis
        #alpha = 0.1
        alpha = initAlpha
        
        for point in data:
            hh = h(coffs, point)
            coffs_n[0] += (point[1] == classifier) - hh
            for url in point[2]:
    #Karolis: klaida..
    #            coffs_n[url[0]] += (point[1] == classifier) - hh
                coffs_n[url] += (point[1] == classifier) - hh
                #coffs_n[url[0]] += ((point[1] == classifier) - hh) * url[1]
                
        for i in range(coffs_len):
            coffs_n[i] /= data_len
            coffs_n[i] *= alpha
            
        return coffs_n


    prev_cost = cost_function(coffs)
    print prev_cost

    #this is kinda gradient descent
    cnt=0
    while True:
        coffs_n = update_coffs(coffs)
        
        for i in range(coffs_len):
            coffs_n[i] = coffs[i] + coffs_n[i]
            
        new_cost = cost_function(coffs_n)
        
        coffs = coffs_n
        if cnt%1000==0:
            print prev_cost, '->', new_cost, ' cnt=', cnt
        cnt+=1
        #for i in range (11):
        # print coffs[i]

    #Karolis    
        #if abs(prev_cost - new_cost) < 0.0000001:
        if abs(prev_cost - new_cost) < absCriterion:
            break
        prev_cost = new_cost

    #dump weights
    file = open("kests_regression_weights.csv", 'w')
    for i in range (coffs_len):
        file.write(str(i) + ',' + str(coffs[i]) + '\n');
    file.close()
    return coffs



    #Karolis: cia ir baigiam..
##sys.exit()
###write test roc data
### you can run yard-plot -show--auc curves_train.txt to get auc curve.
##file = open("curves_train.txt", 'w')
##file.write('output\tmethod\n')
##for line in data:
##    if (line[1] == classifier):
##        file.write('+1\t' + str(h(coffs, line)) + '\n')
##    else:
##        file.write('-1\t' + str(h(coffs, line)) + '\n')
##file.close()
##
##  
###performing on a test set:
##file = open("regression_test.b", 'rb')
##data_test = cPickle.load(file)
##file.close
##
### you can run yard-plot -show--auc curves_test.txt to get auc curve.
##file = open("curves_test.txt", "w")
##for line in data_test:
##    if (line[1] == classifier):
##        file.write('+1\t' + str(h(coffs, line)) + '\n')
##    else:
##        file.write('-1\t' + str(h(coffs, line)) + '\n')
##file.close()
##
