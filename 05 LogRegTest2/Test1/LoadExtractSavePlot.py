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
#import cPickle as pickle
#import numpy as np



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

        
def saveFalloutsRecallsAndPrecisionsToFile(fallouts, recalls, precisions, aucstr = ''):
    print 'saveFalloutsRecallsAndPrecisionsToFile'
    data=np.array([fallouts,recalls,precisions])
    np.savetxt('FalloutsRecallsAndPrecisionsWith ' + aucstr + '.csv', data, delimiter=',')
    print 'done saveFalloutsRecallsAndPrecisionsToFile'


def saveFalloutsAndRecallsToFile(fallouts, recalls, aucstr = ''):
    data=np.array([fallouts,recalls])
    np.savetxt('FalloutsAndRecallsWith ' + aucstr + '.csv', data, delimiter=',')

def loadFalloutsAndRecallsFromFile(fileName):
    data = genfromtxt(fileName, delimiter = ',')
    fallouts = data[0]
    recalls = data[1]
    return fallouts, recalls

def getPositiveRate(y):
    return sum(y)/len(y)

def getNegativeRate(y):
    return sum(1-y)/len(y)



##def extractFPs(fallouts, 
##
##negativeRate = getNegativeRate(y)
##
##
##tp fp
##fn tn
##
##tp + fn = mypositives
##fp + tn = my negtives
##
##
##
##
##falout = fp / fp +tn
##recall = tp / tp +fn
##tp+fp+tn+fn =1


fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial karolis10000 auc 0.742542347986.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial karolis10000 auc 0.742542347986.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial karolis1000000 auc 0.687749223654.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial karolis1000000 auc 0.687749223654.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial karolisE3 auc 0.532798483674.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial karolisE3 auc 0.532798483674.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial karolisE6 auc 0.732576190728.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial karolisE6 auc 0.732576190728.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial kestsE3 auc 0.530284544844.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial kestsE3 auc 0.530284544844.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial scikit100 auc 0.696332813932.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial scikit100 auc 0.696332813932.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial scikit10000 auc 0.656198412996.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial scikit10000 auc 0.656198412996.csv')

fallouts, recalls = loadFalloutsAndRecallsFromFile('FalloutsAndRecallsWith kestsTrial kestsE6 auc 0.534547985964.csv')
plotROC(fallouts, recalls, shouldShow = True , aucstr = 'kestsTrial kestsE6 auc 0.534547985964.csv')

##fallouts, recalls = loadFalloutsAndRecallsFromFile('')
##plotROC(fallouts, recalls, shouldShow = True , aucstr = '')
##



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
            print 'Error. Gradien Norm Criterion. Point gradient is None.'
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
    ans = tp / (tp + getFP(prediction, y))
    return ans

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
    return  auc2
    
    
class DataMunger:
    def __init__(self, scaleDataBool, unitRange = 1, omitIntercept = True):
        self.omitIntercept = omitIntercept
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
                if ( (divmod(int(dataRow[0]),10)[1] == trainSetInt) == (getTestData) ):
                    numOfRows+=1
        dataFile.close()
        return numOfRows
    def getXandY(self, dataFileName, trainSetInt, getTestData = False):
        numOfCols=self.getNumberOfDataColumns()
        numOfRows=self.getNumberOfRows(dataFileName, trainSetInt, getTestData)
        dataFile = open(dataFileName, 'rb')
        dataReader = csv.reader(dataFile)
        if self.omitIntercept:
            x = scipy.sparse.lil_matrix( (numOfRows, numOfCols+1) )
        else:
            x = scipy.sparse.lil_matrix( (numOfRows, numOfCols) )
        y = zeros(numOfRows)
        counter = 0
        for dataRow in dataReader:
            if ( (divmod(int(dataRow[0]),10)[1] == trainSetInt) == (getTestData) ) or trainSetInt == None:
                #fill in x
                if self.omitIntercept:
                    visitedDomains = [j+1 for j in map(int, dataRow[2:])]
                    x.rows[counter] = [0]+visitedDomains
                else:
                    visitedDomains = [j for j in map(int, dataRow[2:])]
                    x.rows[counter] = visitedDomains
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
def getThetaForTrainingSet(trInt, lgf , stoppingLimit, increaseFactor):
    x, y = DataMunger(False,2).preprocessData('ids_with_their_domains.csv',1, trInt)        
    #lgf=LogisticRegressionCostFunction(regCof,regType)
    pt= gR(TransposableNPArray(x), y, lgf, descentStepChangingBy(increaseFactor,2), IterationsStopper(stoppingLimit,10))
    theta = pt.positionVector
    #pt.printPositionVectorToFile('theta trInt,regCof,regType,stoppingLimit,increaseFactor'+str(trInt)+'.csv')
    return theta
def getFalloutsRecallsAndPrecisions(lgf , thresholds, xTest, yTest, theta):
    fallouts=[]
    recalls=[]
    precisions=[]
    #lgf=LogisticRegressionCostFunction(regCof,regType)
    for threshold in thresholds:
        tmp=lgf.getPrediction(xTest, theta, threshold)
        precisions.append(getPrecision(tmp,yTest))
        fallouts.append(getFallout(tmp,yTest))
        recalls.append(getRecall(tmp,yTest))        
    return fallouts, recalls , precisions
def getAucFalloutsRecallsAndPrecisionsForTrainingSet(trInt,lgf, stoppingLimit, increaseFactor, thresholds, xTest, yTest):
    print 'getting AUC for training set integer ',trInt
    #lgf=LogisticRegressionCostFunction(regularisationCoefficient = regCof, regularisationType = regType)
    theta = getThetaForTrainingSet(trInt,lgf, stoppingLimit, increaseFactor)
    #print theta[0:10]
    fallouts, recalls, precisions = getFalloutsRecallsAndPrecisions(lgf, thresholds, xTest, yTest, theta)
    auc = getAUC(fallouts, recalls)
    print 'AUC for Training set integer' , trInt, ' is ',auc
    return auc, fallouts, recalls,precisions    
def getAverageCrossValidationFalloutsRecallsAndPrecisions(falloutai, recallai, precisionai):
    falloutai=np.array(falloutai)
    recallai=np.array(recallai)
    precisionai = np.array(precisionai)
    fallouts=np.mean(falloutai, axis=0)
    recalls=np.mean(recallai, axis=0)
    precisions=np.mean(precisionai, axis=0)
    return fallouts, recalls, precisions
def getAUCAndMakeROCFromCrossValidation(lgf, stoppingLimit, increaseFactor, thresholds):
    aucai=[]
    recallai=[]
    falloutai=[]
    precisionai=[]
    trainingIntegers = range(10)
    #trainingIntegers = [None]    
    for trInt in trainingIntegers:
        xTest, yTest = getTestXandY(trInt)
        tempAns=getAucFalloutsRecallsAndPrecisionsForTrainingSet(trInt,lgf, stoppingLimit, increaseFactor, thresholds, xTest, yTest)
        aucai.append(tempAns[0])
        falloutai.append(tempAns[1])
        recallai.append(tempAns[2])
        precisionai.append(tempAns[3])
    fallouts, recalls, precisions = getAverageCrossValidationFalloutsRecallsAndPrecisions(falloutai, recallai, precisionai)
    auc=np.mean(np.array(aucai))
    nameString= ','.join(['<-AVG','regCof',str(lgf.regularisationCoefficient),'regType',str(lgf.regularisationType),'AUC',str(auc)])
    
    plotRecallVSPrecision(recalls, precisions, shouldShow = False , aucstr = nameString)
    plotROC(fallouts,recalls, False, nameString )
    saveFalloutsRecallsAndPrecisionsToFile(fallouts, recalls, precisions, aucstr = nameString)

    return auc

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
                     





