from __future__ import division
from sklearn import linear_model
import sys
import csv
from collections import Counter
from collections import OrderedDict
from pylab import * ## should replace this with sth smarter
from numpy import genfromtxt
from time import clock
from numpy import linalg
from itertools import product as prd
import scipy.sparse
import kestoDu
#import cPickle as pickle
#import numpy as np



class AdjustedDataMunger:
    def __init__(self, scaleDataBool, unitRange = 1):
        self.scaleDataBool = scaleDataBool
        self.rangeVector = None
        self.averageVector = None
        self.unitRange = unitRange
        
    def scaleData(self,x):
        self.rangeVector=[]
        self.averageVector=[]
        for i in range(shape(x)[1]):
            self.rangeVector.append(max(x[:,i]) - min(x[:,i]))
            #self.averageVector.append( np.mean(x[:,i]) ) 
            self.averageVector.append((max(x[:,i]) + min(x[:,i])) / 2) 
        self.rangeVector=np.array(self.rangeVector)
        self.averageVector=np.array(self.averageVector)
        for i in range(shape(x)[1]):
            x[:,i] = self.unitRange * (x[:,i] - self.averageVector[i]) / self.rangeVector[i]
        return x
        
    def preprocessData(self,dataFile, skipHeaderInt = 1):
        #data = genfromtxt(dataFile, delimiter = ',', skip_header=skipHeaderInt, dtype = float)
        data = genfromtxt(dataFile, delimiter = ',', skip_header=1, dtype = str)
        y=data[:,2]
        y= (y=='male')*1
        data = genfromtxt(dataFile, delimiter = ',', skip_header=1, dtype = 'd')
        x=data[:,3:-3]
        if self.scaleDataBool:
            x=self.scaleData(x)
        temp = np.array([ones(len(y))])
        x=np.hstack((temp.transpose(),x))
        return x, y
    
    def postprocessData(self,x):
        pass
    def postprocessResults(self,theta):
        if self.scaleDataBool:
            newTheta=[]
            newTheta.append(theta[0] - 2 * np.dot(theta[1:],self.averageVector / self.rangeVector))
            for i in range(len(theta[1:])):
                newTheta.append(self.unitRange * theta[1:][i] / self.rangeVector[i])
            newTheta=np.array(newTheta)
            return newTheta
        else:
            return theta
class McDataMunger:
    def __init__(self, scaleDataBool):
        self.scaleDataBool = scaleDataBool
        self.rangeVector = None
        self.averageVector = None
        self.unitRange = 1
        
    def scaleData(self,x):
        self.rangeVector=[]
        self.averageVector=[]
        for i in range(shape(x)[1]):
            self.rangeVector.append(max(x[:,i]) - min(x[:,i]))
            self.averageVector.append( np.mean(x[:,i]) ) 
        self.rangeVector=np.array(self.rangeVector)
        self.averageVector=np.array(self.averageVector)
        for i in range(shape(x)[1]):
            x[:,i] = self.unitRange * (x[:,i] - self.averageVector[i]) / self.rangeVector[i]
        return x
        
    def preprocessData(self,dataFile, skipHeaderInt = 1):
        data = genfromtxt(dataFile, delimiter = ',', skip_header=skipHeaderInt, dtype = float)
        y=data[:,-1]
        x=data[:,:-1]
        if self.scaleDataBool:
            x=self.scaleData(x)
        temp = np.array([ones(len(y))])
        x=np.hstack((temp.transpose(),x))
        return x, y
    
    def postprocessData(self,theta):
        if self.scaleDataBool:
            newTheta=[]
            newTheta.append(theta[0] - 2 * np.dot(theta[1:],self.averageVector / self.rangeVector))
            for i in range(len(theta[1:])):
                newTheta.append(self.unitRange * theta[1:][i] / self.rangeVector[i])
            newTheta=np.array(newTheta)
            return newTheta
        else:
            return theta


def logisticFunction(self, z):
    return 1 / (1 + exp(-z))

class LogisticRegressionCostFunction:

    def __init__(self, regularisationCoefficient = 0, regularisationType = 2):
        self.regularisationCoefficient = regularisationCoefficient
        self.regularisationType = regularisationType
        
    def getPrediction(self, x, theta, threshold = 0):
        return (x.dot(theta) > threshold) * 1

    def logisticFunction(self, z):
        return 1 / (1 + exp(-z))
    def cummulativeCost(self,z,y): # takes y and phi times x
        return ((np.dot(1-y,log(1+exp(z))**(1-y))+np.dot(y,log(1+exp(-z))**y))) / len(y)

    def regularisationCost(self, positionVector, sampleDataSize):
        return self.regularisationCoefficient * sum(abs(positionVector[1:])**self.regularisationType) / sampleDataSize
        
    def calculate(self, featureValues, positionVector, groundTruthValues):
        summand1 = self.cummulativeCost(featureValues.values.dot(positionVector), groundTruthValues)
        summand2 = self.regularisationCost(positionVector, len(groundTruthValues))
        return summand1 + summand2
    
    def calculateRegularisationGradient(self,positionVector, sampleDataSize):
        if self.regularisationType == 2:
            return hstack(([0],positionVector[1:])) * self.regularisationCoefficient / sampleDataSize
        elif self.regularisationType == 1:
            return sign(hstack(([0],positionVector[1:]))) * self.regularisationCoefficient / sampleDataSize
        else:
            print 'the Code is implemented for L1 and L2 regularistaion. Given regularistation: L',self.regularisationType
            sys.exit()
        
    def calculateCummulativeCostGradient(self, featureValues, positionVector, groundTruthValues):
        logHypothesis = self.logisticFunction(featureValues.values.dot(positionVector))
        loss = logHypothesis - groundTruthValues 
        return featureValues.transposed().dot(loss) / len(groundTruthValues)

    def calculateGradient(self, featureValues, positionVector, groundTruthValues):
        summand1 = self.calculateCummulativeCostGradient(featureValues, positionVector, groundTruthValues) 
        summand2 = self.calculateRegularisationGradient(positionVector, len(groundTruthValues))
        return summand1 + summand2


def calculateDescentStepUltimate(oldStep, wasSuccessful, increaseFactor, decreaseFactor):
    return oldStep * increaseFactor if wasSuccessful else oldStep / decreaseFactor

def descentStepChangingBy(increaseFactor = 1, decreaseFactor = 1):
    return lambda oldStep, wasSuccessful: calculateDescentStepUltimate(oldStep, wasSuccessful, increaseFactor, decreaseFactor)

class CostFunctionPoint:
    def __init__(self, costFunction, cost, positionVector):
        self.costFunction = costFunction
        self.cost = cost
        self.positionVector = positionVector
        self.gradient = None

    def descend(self, featureValues, groundTruthValues, step):
        self.gradient = self.costFunction.calculateGradient(featureValues, self.positionVector, groundTruthValues)
        newPositionVector = self.positionVector - step * self.gradient
        newCost = self.costFunction.calculate(featureValues, newPositionVector, groundTruthValues)
        return CostFunctionPoint(self.costFunction, newCost, newPositionVector)

    def isLowerThan(self, otherDescent):
        #print 'IS LOWER PRINT'  , self.toString(), otherDescent.toString()
        return otherDescent.cost is None or otherDescent.cost > self.cost

    def printPositionVectorToFile(self, coefficientsFileName):
        outputFile = open(coefficientsFileName, 'wb')
        writer = csv.writer(outputFile)
        writer.writerow(self.positionVector)
        outputFile.close()

    def toString(self):
        return self.cost,'<-Cost. Theta->',self.positionVector

class TransposableNPArray:
    def __init__(self, values):
        self.values = values
        self.valuesTransposed = None

    def transposed(self):
        if self.valuesTransposed is None:
            self.valuesTransposed = self.values.transpose()
        return self.valuesTransposed



class GradientNormStopper:
    def __init__(self, limit, span = 1):
        self.limit = limit
        self.span = span
        self.counter = 0 

    def shouldStop(self, iteration, point, newPoint):
        if point.gradient == None:
            print 'Error. RelCauchyCriterion. Point gradient is None.'
            sys.exit()
        else:
            gradientNorm = linalg.norm(point.gradient)
        if gradientNorm <= self.limit:
            self.counter+=1
        else:
            self.counter = 0
        return True if self.counter >= self.span else False

class AbsCauchyStopper:
    def __init__(self, limit, span = 1):
        self.limit = limit
        self.span = span
        self.counter = 0 

    def shouldStop(self, iteration, point, newPoint):
        if point.cost == None or newPoint.cost == None :
            if iteration == 1:
                absCauchy = self.limit + 1    
            else:
                print 'Error. AbsCauchyCriterion. Point gradient is None. Iteration:', iteration
                sys.exit()
        else:
            absCauchy =abs(point.cost - newPoint.cost)
        if absCauchy <= self.limit:
            self.counter+=1
        else:
            self.counter = 0
        return True if self.counter >= self.span else False

class IterationsStopper:
    def __init__(self, limit, span = 1):
        self.limit = limit
        self.span = span
        self.counter = 0 

    def shouldStop(self, iteration, point, newPoint):
        return True if iteration >= self.limit else False
    

def gR(featureValues, groundTruthValues, costFunction, calculateDescentStepFunction, processStopper, descentStep = 0.1, primitiveDescent = False):
    t1=clock()
    #descentStep=0.1
    iteration = 0
    positionVector = zeros(shape(featureValues.values)[1])
    point = CostFunctionPoint(costFunction, None, positionVector)
    
    while True:
        if descentStep<0.0005 and descentStep>0.005:
            print iteration, point.toString(), descentStep
        iteration += 1
        nextPoint = point.descend(featureValues, groundTruthValues, descentStep)
        descentSuccessful = nextPoint.isLowerThan(point)
        descentStep = calculateDescentStepFunction(descentStep, descentSuccessful)
        if descentStep == 0 or processStopper.shouldStop(iteration, point, nextPoint):
            #if descentStep == 0: print 'Alpha was zero'
            print 'GradienDescend stopped at:'
            print 'Iteration:', iteration,'. Duration: ', clock()-t1
            
            return nextPoint if descentSuccessful else point
        if descentSuccessful or primitiveDescent:
            point = nextPoint
            

##        print iteration
##        print descentSuccessful
##        print descentStep
##        raw_input()



def getPredictionAccuracy(prediction, y):
    return sum((prediction == y) * 1) / len(y)

def getTP(prediction, y):
    return sum(( prediction == y*2-1) * 1)

def getFP(prediction, y):
    return sum((prediction == y+1) * 1)

def getFN(prediction, y):
    return sum((prediction  == y-1) * 1)

def getTN(prediction, y):
    return sum((prediction  == y*2) * 1)

def getRecall(prediction, y):
    tp = getTP(prediction, y)
    return tp / (tp + getFN(prediction, y))

def getPrecision(prediction, y):
    tp = getTP(prediction, y)
    return tp / (tp + getFP(prediction, y))

def getFallout(prediction, y):
    fp = getFP(prediction, y)
    return fp / (fp + getTN(prediction, y))

def getAUC(fallout, recall):
    if type(fallout)!=type([]) or type(recall)!=type([]):
        print 'Fallout and Recall types mus be LIST'
        sys.exit()
    size = len(recall)
    auc1 = abs(0.5 * sum([((fallout[i] - fallout[i - 1]) * (recall[i] + recall[i - 1])) for i in range(1,size)]))
    fallout = [1.0] + fallout + [0.0]
    recall = [1.0] + recall + [0.0]
    size = len(recall)
    auc2 = abs(0.5 * sum([((fallout[i] - fallout[i-1]) * (recall[i] + recall[i-1])) for i in range(1,size)]))
    return [auc1, auc2]
    
def plotROC(fallout, recall, shouldShow = True , aucstr = ''):
    p1 = plot(fallout, recall)
    xlabel('fallout')
    ylabel('Recall')
    title('ROC with ' + aucstr)
    grid(True)
    t=arange(0, 1.1, 0.5)
    p2=plot(t, t, 'r--')
#    legend([p2], ['ROC with AUC ' + aucstr + '.png'])
    savefig('ROC with ' + aucstr + '.png')
    if shouldShow:
        show()
        
def saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = ''):
    data=np.array([fallouts,recalls])
    np.savetxt('FalloutsAndRecallsWith ' + aucstr + '.csv', data, delimiter=',')

def loadFalloutsAndRecallsFromFile(fileName):
    data = genfromtxt(fileName, delimiter = ',')
    fallouts = data[0]
    recalls = data[1]
    return fallouts, recalls
    
class DataMunger:
    def __init__(self, scaleDataBool, unitRange = 1):
        self.scaleDataBool = scaleDataBool
        self.rangeVector = None
        self.averageVector = None
        self.unitRange = unitRange
        
    def scaleData(self,x):
        self.rangeVector=[]
        self.averageVector=[]
        for i in range(shape(x)[1]):
            self.rangeVector.append(max(x[:,i]) - min(x[:,i]))
            #self.averageVector.append( np.mean(x[:,i]) ) 
            self.averageVector.append((max(x[:,i]) + min(x[:,i])) / 2) 
        self.rangeVector=np.array(self.rangeVector)
        self.averageVector=np.array(self.averageVector)
        for i in range(shape(x)[1]):
            x[:,i] = self.unitRange * (x[:,i] - self.averageVector[i]) / self.rangeVector[i]
        return x    
    def postprocessData(self,x):
        pass
    def postprocessResults(self,theta):
        if self.scaleDataBool:
            newTheta=[]
            newTheta.append(theta[0] - 2 * np.dot(theta[1:],self.averageVector / self.rangeVector))
            for i in range(len(theta[1:])):
                newTheta.append(self.unitRange * theta[1:][i] / self.rangeVector[i])
            newTheta=np.array(newTheta)
            return newTheta
        else:
            return theta
    def getNumberOfDataColumns(self):
        tempr=csv.reader(open('all_domains.csv', 'rb'))
        return len(list(tempr))
    def getNumberOfRows(self, dataFileName, trainSetInt, getTestData = False):
        dataFile = open(dataFileName, 'rb')
        dataReader = csv.reader(dataFile)
        if trainSetInt == None:
            numOfRows = len(list(dataReader))
        else:
            numOfRows = 0
            for dataRow in dataReader:
                if ( (divmod(int(dataRow[0]),10)[1] == trainSetInt) ^ (getTestData) ):
                    numOfRows+=1
        dataFile.close()
        return numOfRows
    def getXandY(self, dataFileName, trainSetInt, getTestData = False):
        numOfCols=self.getNumberOfDataColumns()
        numOfRows=self.getNumberOfRows(dataFileName, trainSetInt, getTestData)
        dataFile = open(dataFileName, 'rb')
        dataReader = csv.reader(dataFile)
        x = scipy.sparse.lil_matrix( (numOfRows, numOfCols+1) )
        y = zeros(numOfRows)
        counter = 0
        for dataRow in dataReader:
            if ( (divmod(int(dataRow[0]),10)[1] == trainSetInt) ^ (getTestData) ) or trainSetInt == None:
                #fill in x
                visitedDomains = [j+1 for j in map(int, dataRow[2:])]
                x.rows[counter] = [0]+visitedDomains  
                x.data[counter] = [1 for j in x.rows[counter]]  
                #fill in y
                y[counter] = (dataRow[1]=='male')*1
                counter+=1
        x = scipy.sparse.csr_matrix(x)
        dataFile.close()
        return x ,y
    def preprocessData(self, dataFileName, skipHeaderInt = 1, trainSetInt = None, getTestData = False):
        x, y = self.getXandY(dataFileName, trainSetInt, getTestData)
        if self.scaleDataBool:
            print 'Sparse data is not being scaled here'
            #print x.todense()
        return x, y

###scripting onwards
def getTestXandY(trInt):
    xTest, yTest = DataMunger(False).preprocessData('ids_with_their_domains.csv',0 ,trInt ,getTestData = True)        
    return xTest, yTest      
def getThetaForTrainingSet(trInt,regCof,regType, stopLim, stepIn):
    x, y = DataMunger(False,2).preprocessData('ids_with_their_domains.csv',1, trInt)        
    lgf=LogisticRegressionCostFunction(regCof,regType)
    pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(stepIn,2), IterationsStopper(stopLim,10))
    theta = pt.positionVector
    pt.printPositionVectorToFile('theta trInt,regCof,regType,stopLim,stepIn'+str(trInt)+'.csv')
    return theta
def getFalloutsAndRecalls(regCof,regType, thresholds, xTest, yTest, theta):
    fallouts=[]
    recalls=[]
    lgf=LogisticRegressionCostFunction(regCof,regType)
    for threshold in thresholds:
        tmp=lgf.getPrediction(xTest, theta, threshold)
        fallouts.append(getFallout(tmp,yTest))
        recalls.append(getRecall(tmp,yTest))        
    return fallouts, recalls
def getAucRecallsAndFalloutsForTrainingSet(trInt,regCof,regType, stopLim, stepIn, thresholds, xTest, yTest):
    print 'getting AUC for training set integer ',trInt
    theta = getThetaForTrainingSet(trInt,regCof,regType, stopLim, stepIn)
    #print theta[0:10]
    fallouts, recalls = getFalloutsAndRecalls(regCof,regType, thresholds, xTest, yTest, theta)
    auc = getAUC(fallouts, recalls)
    print 'AUC for Training set integer' , trInt, ' is ',auc
    return auc, fallouts, recalls    
def getAverageCrossValidationFalloutsAndRecalls(falloutai, recallai):
    falloutai=np.array(falloutai)
    recallai=np.array(recallai)
    fallouts=np.mean(falloutai, axis=0)
    recalls=np.mean(recallai, axis=0)
    return fallouts, recalls
def getAUCsAndMakeROCFromCrossValidation(regCof,regType, stopLim, stepIn, thresholds):
    aucai=[]
    recallai=[]
    falloutai=[]
    trainingIntegers = range(10)
    #trainingIntegers = [None]    
    for trInt in trainingIntegers:
        xTest, yTest = getTestXandY(trInt)
        tempAns=getAucRecallsAndFalloutsForTrainingSet(trInt,regCof,regType, stopLim, stepIn, thresholds, xTest, yTest)
        aucai.append(tempAns[0])
        falloutai.append(tempAns[1])
        recallai.append(tempAns[2])
    fallouts, recalls = getAverageCrossValidationFalloutsAndRecalls(falloutai, recallai)
    myString= ','.join(['regCof, regType, AUC ',str(regCof),str(regType),str(np.mean(np.array(aucai)))])
    
    plotROC(fallouts,recalls, False, myString )
    return aucai

#def getFeaturedFromIndices(listOfIndices)

def printTopCoefficientsAbsAndTheirFeatures(topNumber, outputFileName, coefficientsFileName):
    data = genfromtxt(coefficientsFileName, delimiter = ',')
    theta=data
    descendingListOfIndices = np.argsort(abs(theta)).tolist()
    descendingListOfIndices.reverse()
    descendingListOfCoefficients = [theta[i] for i in descendingListOfIndices]
    features=descendingListOfIndices
    #features=getFeaturedFromIndices(descendingListOfIndices)
    outputFile = open(outputFileName, 'wb')
    writer = csv.writer(outputFile)
    writer.writerow(['Features of top '+str(topNumber)+' biggest absolute value coefficients, their indices and coefficients'])
    writer.writerow(features[:topNumber])
    writer.writerow(descendingListOfIndices[:topNumber])
    writer.writerow(descendingListOfCoefficients[:topNumber])
    outputFile.close()
                     


##def printCoefficients(coefficients, coefficientsFileName):
##    outputFile = open(coefficientsFileName, 'wb')
##    writer = csv.writer(outputFile)
##    writer.writerow(coefficients)
##    outputFile.close()


###scr


#kests trial
print 'Begin'


munger = DataMunger(False,2)
trInt=0
x, y = munger.preprocessData('ids_with_their_domains.csv',0,trInt)   
xTest, yTest = getTestXandY(trInt)

numCols = shape(x)[1]
print numCols
print shape(x)

#Trial parameters

thresholds=[-1000,-10,-5,-4,-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3,4,5,10,1000]
thresholds=sort([-1000]+list(np.linspace(-10,10,21))+list(np.linspace(-1,1,21))+[1000])
initLearningRate=0.1
learningRateRule='constant'
regularisationCoff=0
initTheta = ones(numCols)*0.5
numberOfIterations = 100


clf = linear_model.SGDClassifier(alpha=regularisationCoff, #regularistion coff
                                 class_weight=None, #used to "deal with unbalanced classes"
                                 epsilon=0.1, # not relevant for simple LogReg
                                 eta0=initLearningRate, # initial learning rate
                                 fit_intercept=False,#statu False nes mano duomenyse jau yra vientai
                                #sitas bajeris man veiks tik kai be reguliarizacijos...
                                 #decides wheather decision function is homogenic or with a constant member
                                 l1_ratio=0, #L1 regularisation wieght in elastic_net regularistaion
                                 learning_rate=learningRateRule, #learning rate change rule
                                 loss='log',
                                 n_iter=numberOfIterations, #number of times the algorithm will pass the whole data.
                                 #n_iter=1000000, #number of times the algorithm will pass the whole data.
                                 n_jobs=1, # number of CPU to use for 1vsAll
                                 penalty='l2', #regularistation type
                                 power_t=0.5, #does not apply
                                 random_state=None,# not relevant for simple LogReg
                                 rho=None,
                                 shuffle=False, # not relevant for simple LogReg
                                 verbose=0, #verbose level
                                 warm_start=False)


###scikit100 
##clf.n_iter=100
##initTheta = ones(numCols)*0.5
##clf.fit(x, y, coef_init=initTheta )
###print clf.score(x,y)
##sciTheta100 = clf.coef_[0]
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, sciTheta100)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial scikit100 auc '+str(auc[0]))
###print loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial scikit100 auc 0.696332813932.csv')
###plotROC(fallouts, recalls, shouldShow = False , aucstr = 'kestsTrial scikit100 auc '+str(auc[0]))
##print auc, 'scikit100'
##
###scikit10000
##clf.n_iter=10000
##initTheta = ones(numCols)*0.5
##clf.fit(x, y, coef_init=initTheta )
###print clf.score(x,y)
##sciTheta10000 = clf.coef_[0]
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, sciTheta10000)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial scikit10000 auc '+str(auc[0]))
##print auc, 'scikit10000'
##
###Karolis10000
##lgf=LogisticRegressionCostFunction(regularisationCoefficient = 0, regularisationType = 2)
##pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(1,1), IterationsStopper(10000), initLearningRate, primitiveDescent = True)
##karolisTheta10000 = pt.positionVector
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, karolisTheta10000)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial karolis10000 auc '+str(auc[0]))
##print auc, 'karolis10000'
##
##
###Karolis1000000
##lgf=LogisticRegressionCostFunction(regularisationCoefficient = 0, regularisationType = 2)
##pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(1,1), IterationsStopper(1000000), initLearningRate, primitiveDescent = True)
##karolisTheta1000000 = pt.positionVector
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, karolisTheta1000000)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial karolis1000000 auc '+str(auc[0]))
##print auc, 'karolis1000000'
##
###KarolisE3
##lgf=LogisticRegressionCostFunction(regularisationCoefficient = 0, regularisationType = 2)
##pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(1,1), AbsCauchyStopper(0.001), initLearningRate, primitiveDescent = True)
##karolisThetaE3 = pt.positionVector
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, karolisThetaE3)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial karolisE3 auc '+str(auc[0]))
##print auc, 'karolisE3'
##
###KarolisE6
##lgf=LogisticRegressionCostFunction(regularisationCoefficient = 0, regularisationType = 2)
##pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(1,1), AbsCauchyStopper(0.000001), initLearningRate, primitiveDescent = True)
##karolisThetaE6 = pt.positionVector
##fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, karolisThetaE6)
##auc = getAUC(fallouts, recalls)
##saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial karolisE6 auc '+str(auc[0]))
##print auc, 'karolisE6'


#Kests0.001
kestsThetaE3 = kestoDu.kestsRoutine(absCriterion = 0.001, initAlpha = initLearningRate, initTheta = 0.5, outFileName ='kests_regression_weightsE3.csv')
fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, kestsThetaE3)
auc = getAUC(fallouts, recalls)
saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial kestsE3 auc '+str(auc[0]))
print auc, 'kestsE3'

#Kests0.000001
kestsThetaE6 = kestoDu.kestsRoutine(absCriterion = 0.000001, initAlpha = initLearningRate, initTheta = 0.5, outFileName ='kests_regression_weightsE6.csv')
fallouts, recalls = getFalloutsAndRecalls(regularisationCoff, 2, thresholds, xTest, yTest, kestsThetaE6)
auc = getAUC(fallouts, recalls)
saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = 'kestsTrial kestsE6 auc '+str(auc[0]))
print auc, 'kestsE6'
sys.exit()

#End of trial
    
printTopCoefficientsAbsAndTheirFeatures(10,'temporaryName.csv','theta trInt,regCof,regType,stopLim,stepIn0.csv')
regCof,regType, stopLim, stepIn = 0.5,2,100000,2
#thresholds=[-1000,-10,-2,-0.5,0,0.5,2,10,1000]
thresholds=[-1000,-10,-5,-4,-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3,4,5,10,1000]
thresholds=sort([-1000]+list(np.linspace(-10,10,21))+list(np.linspace(-1,1,21))+[1000])
aucs=getAUCsAndMakeROCFromCrossValidation(regCof,regType, stopLim, stepIn, thresholds)
print aucs
aucs=np.array(aucs)
print 'average AUC ', np.mean(aucs)
print 'Done?!'







sys.exit()    
##def crossValidation():
##    #for regCof,regType, stopLim, stepIn in prd([0.1,0.5,1,2],[1,2],[0.1,0.01,0.001,0.001],[1,1.2,2]):
##    #for regCof,regType, stopLim, stepIn in prd([0.1,3],[1,2],[100000000],[2]):
##    #loop over parameters
##    for regCof,regType, stopLim, stepIn in prd([0.1],[2],[1000],[2]): #dont try to iterate here. WILL BE BAD!!!
##        #do Auc and Roc crosvalidation with given parameters
##        munger=DataMunger(False,2)
##        xTest, yTest = munger.preprocessData('ids_with_their_domains.csv',0)        
##        # define threshold array
##        threshold=[-1000,-10,-2,-0.5,0,0.5,2,10,1000]
##        #threshold=[0]
##        trainInts=range(1)
##        prediction=[[] for i in trainInts]
##        fallout=[[] for i in trainInts]
##        recall=[[] for i in trainInts]
##        auc=[[0,0] for i in trainInts]
##        falloutas=np.array([0 for i in range(len(threshold))])
##        recallas=np.array([0 for i in range(len(threshold))])
##        for tr in trainInts:
##            #get auc, recal and prediction for a given training int  
##            x, y = munger.preprocessData('ids_with_their_domains.csv',0,tr)
##            m,n =shape(x)
##            theta = zeros(n)
##            prediction[tr] = [[] for i in range(len(threshold))]
##            fallout[tr] = [[] for i in range(len(threshold))]   
##            recall[tr] = [[] for i in range(len(threshold))]
##            lgf=LogisticRegressionCostFunction(regCof,regType)
##            print 'start gr'
##            pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(stepIn,2), IterationsStopper(stopLim,10))
##            result=pt.positionVector
##            for i in range(len(threshold)):
##                prediction[tr][i] = lgf.getPrediction(xTest, result, threshold[i])
##                fallout[tr][i]=getFallout(prediction[tr][i], yTest)
##                recall[tr][i]= getRecall(prediction[tr][i], yTest)
##                print sum((prediction[tr][i]==yTest)*1)/len(yTest)
##                print len(prediction[tr][i])
##                print sum(yTest)
##                print len(yTest)
##                print fallout[tr], 'fall'
##                print recall[tr][i]
##            auc[tr]=getAUC(fallout[tr],recall[tr])
##            #print auc, regCof,regType, stopLim, stepIn
##            #plotROC(fallout[tr],recall[tr],True, makestring)
##        auc=np.array(auc)
##        aucas = np.mean(auc[:,1])  
###        sys.exit()
##        print 'aucas', aucas, auc
##        for tr in range(10):
##            print falloutas
##            print fallout[tr]
##            sys.exit()
##            falloutas = fallout[tr]+falloutas
##            recallas = recall[tr]+recallas
##        falloutas=falloutas/10
##        recallas=recallas/10
##        plotROC(falloutas,recallas,True, 'tas vienas')
##    
##
##
##crossValidation()
##sys.exit()
##munger=DataMunger(False,2)
###result [0.80857372528019555, 0.80857372528019555] 0.1 2 100000000 2
##x, y = munger.preprocessData('ids_with_their_domains.csv',0)
###print x
###print y
##m,n =shape(x)
##print m, n
##print sum(y)/len(y)
###sys.exit()
##theta = zeros(n)
##threshold=[-1000,-10,-2,-0.5,0,0.5,2,10,1000]
###threshold=[0.01,0.1,0.5,1,2,10,100,exp(100)]
##prediction = [[] for i in range(len(threshold))]
###result = [[] for i in range(len(threshold))]
##fallout = [[] for i in range(len(threshold))]
##recall= [[] for i in range(len(threshold))]
###for regCof,regType, stopLim, stepIn in prd([0.1,0.5,1,2],[1,2],[0.1,0.01,0.001,0.001],[1,1.2,2]):
###for regCof,regType, stopLim, stepIn in prd([0.1,3],[1,2],[100000000],[2]):
##for regCof,regType, stopLim, stepIn in prd([0.1],[2],[100000000],[2]):
##    lgf=LogisticRegressionCostFunction(regCof,regType)
##    pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(stepIn,2), IterationsStopper(stopLim,10))
##    result=pt.positionVector
##    for i in range(len(threshold)):
##    #for i in [1,4,7]:
####        print np.dot(x,result)
####        print lgf.logisticFunction(np.dot(x,result))
##        prediction[i] = lgf.getPrediction(x, result, threshold[i])
##        #prediction[i] = lgf.getPrediction(x, result, 0)
##        #acc[i]=getPredictionAccuracy(getPrediction(x,result, threshold(i)),y)
##        #print acc, regCof,regType, stopLim, stepIn
##        fallout[i]=getFallout(prediction[i], y)
##        recall[i]= getRecall(prediction[i], y)
##        #print sum(prediction[i])
##        #print sum(y)
####        print y
####        print fallout[i]
####        print recall[i]
####        print sum(prediction[i])
####        print 'sdfa'
##        #sys.exit()
####        raw_input()
##        
##    auc=getAUC(fallout,recall)
##    print auc, regCof,regType, stopLim, stepIn
##    plotROC(fallout,recall)
##
##sys.exit()
####minimum=np.array([-0.0316899,-0.1911447,-0.4516988,6.55483368,-1.5423271,1.20252151,
####                   0.81213118,-0.5612584,-0.2435837,0.88195461,-1.2403244,0.25063274,
####                  -0.3382761,1.20216841,-0.1771912,-0.7312406 ])
####
####rescaleData=True
####munger=McDataMunger(rescaleData)
####x, y = munger.preprocessData('we love drugs.csv')
####m,n =shape(x)
####theta = ones(n)
####lgf=LogisticRegressionCostFunction(0.5,1)
#####pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(2,2), AbsCauchyStopper(0.00001,10))
####pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(2,2), GradientNormStopper(0.01,10))
#####temp1=munger.postprocessData(pt.positionVector)
####temp1=pt.positionVector
####temp2=lgf.calculate(TransposableNPArray(x),temp1,y)
####mincost=lgf.calculate(TransposableNPArray(x),minimum,y)
####sys.exit()
####print temp2,'<--make this unchanging'
####print mincost
####pred1=getPredictionAccuracy(getPrediction(x,temp1),y)
####pred2=getPredictionAccuracy(getPrediction(x,minimum),y)
####distributionofdreams=getPredictionAccuracy(getPrediction(x,minimum),ones(m))
####print pred1
####print pred2            
####print temp1
####print minimum
####print linalg.norm(temp1-minimum)
####
####sys.exit()
