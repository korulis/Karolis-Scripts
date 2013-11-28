from __future__ import division
import csv
import re
import sys
from collections import Counter
from collections import OrderedDict
from pylab import * ## should replace this with sth smarter
from numpy import genfromtxt
from time import clock
from numpy import linalg



class LogisticRegressionCostFunction:

    def __init__(self, regularisationCoefficient = 0, regularisationType = 2):
        self.regularisationCoefficient = regularisationCoefficient
        self.regularisationType = regularisationType

    def logisticFunction(self, z):
        return 1 / (1 + exp(-z))
    def cummulativeCost(self,z,y): # takes y and phi times x
        return ((np.dot(1-y,log(1+exp(z))**(1-y))+np.dot(y,log(1+exp(-z))**y))) / len(y)

    def regularisationCost(self, positionVector, sampleDataSize):
        return self.regularisationCoefficient * sum(abs(positionVector[1:])**self.regularisationType) / sampleDataSize
        
    def calculate(self, featureValues, positionVector, groundTruthValues):
        summand1 = self.cummulativeCost(np.dot(featureValues.values, positionVector), groundTruthValues)
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
        logHypothesis = self.logisticFunction(np.dot(featureValues.values, positionVector))
        loss = logHypothesis - groundTruthValues 
        return np.dot(featureValues.transposed(), loss) / len(groundTruthValues)

    def calculateGradient(self, featureValues, positionVector, groundTruthValues):
        summand1 = self.calculateCummulativeCostGradient(featureValues, positionVector, groundTruthValues) 
        summand2 = self.calculateRegularisationGradient(positionVector, len(groundTruthValues))
        return summand1 + summand2

class LogisticRegressionCostFunctionBad:

    def logisticFunction(self, z):
        return 1 / (1 + exp(-z))
    
    def cummulativeCost(self, z,y):
        return np.dot((1 - y),(-log(1 - z))**(1 - y))+np.dot(y,(-log(z)**y))
    
    def calculate(self, featureValues, positionVector, groundTruthValues):
        logisticHypothesis = self.logisticFunction(np.dot(featureValues.values, positionVector))
        return self.cummulativeCost(logisticHypothesis, groundTruthValues)

    def calculateGradient(self, featureValues, positionVector, groundTruthValues):
        logHypothesis = self.logisticFunction(np.dot(featureValues.values, positionVector))
        loss = logHypothesis - groundTruthValues 
        return np.dot(featureValues.transposed(), loss) / len(groundTruthValues)
        


def calculateDescentStepUltimate(oldStep, wasSuccessful, increaseFactor, decreaseFactor):
    return oldStep * increaseFactor if wasSuccessful else oldStep / decreaseFactor

def descentStepChangingBy(increaseFactor, decreaseFactor):
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
    

def gR(positionVector, featureValues, groundTruthValues, costFunction, calculateDescentStepFunction, processStopper):
    descentStep=1
    iteration = 0
    point = CostFunctionPoint(costFunction, None, positionVector)
    
    while True:
        iteration += 1
        nextPoint = point.descend(featureValues, groundTruthValues, descentStep)
        descentSuccessful = nextPoint.isLowerThan(point)
        descentStep = calculateDescentStepFunction(descentStep, descentSuccessful)
        if descentStep == 0 or processStopper.shouldStop(iteration, point, nextPoint):
            #if descentStep == 0: print 'Alpha was zero'
            print 'Stopped at:'
            print 'Iteration:', iteration
            
            return nextPoint if descentSuccessful else point
        if descentSuccessful:
            point = nextPoint
##        print point.toString()
##        print iteration
##        print descentSuccessful
##        print descentStep
##        raw_input()

def getPrediction(x, theta):
    return (np.dot(x,theta)>0.0)*1

def getPredictionAccuracy(prediction, y):
    return sum((prediction == y) * 1) / len(y)

def getTP(prediction, y):
    return sum(((prediction > 0.5) == y*2-1) * 1)

def getFP(prediction, y):
    return sum(((prediction > 0.5) == y+1) * 1)

def getFN(prediction, y):
    return sum(((prediction > 0.5) == y-1) * 1)

def getTN(prediction, y):
    return sum(((prediction > 0.5) == y*2) * 1)
    

class DataMunger:
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


def makeTestAndTrainSets(infilename = 'we love drugs.csv'):
    inf = open(infilename, 'rb')
    reader = csv.reader(inf, delimiter = ',')
    outftest = open('test' + infilename,'wb')
    outftrain = open('train' + infilename,'wb')
    writerTest = csv.writer(outftest)
    writerTrain = csv.writer(outftrain)
    cnt = 0
    for i in reader:
        if cnt == 0:
            writerTest.writerow(i)
            writerTrain.writerow(i)
        elif cnt % 10 == 1:
            writerTest.writerow(i)
        else:
            writerTrain.writerow(i)
        cnt+=1
    inf.close()
    outftest.close()
    outftrain.close()


        
minimum=np.array([-0.0316899,-0.1911447,-0.4516988,6.55483368,-1.5423271,1.20252151,
                   0.81213118,-0.5612584,-0.2435837,0.88195461,-1.2403244,0.25063274,
                  -0.3382761,1.20216841,-0.1771912,-0.7312406 ])

makeTestAndTrainSets()
rescaleData=True
munger=DataMunger(rescaleData)
x, y = munger.preprocessData('we love drugs.csv')
xTest, yTest = munger.preprocessData('we love drugs.csv')
m,n =shape(x)
theta = ones(n)
lgf=LogisticRegressionCostFunction(0.05,2)
#pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(2,2), AbsCauchyStopper(0.00001,10))
pt= gR(theta, TransposableNPArray(x), y, lgf, descentStepChangingBy(2,2), GradientNormStopper(0.01,10))
#temp1=munger.postprocessData(pt.positionVector)
temp1=pt.positionVector
temp2=lgf.calculate(TransposableNPArray(xTest),temp1,yTest)
mincost=lgf.calculate(TransposableNPArray(xTest),minimum,yTest)
print temp2,'implemented classifier cost'
print mincost, ' given answer cost'
pred1=getPredictionAccuracy(getPrediction(xTest,temp1),yTest)
pred2=getPredictionAccuracy(getPrediction(xTest,minimum),yTest)
distributionofdreams=getPredictionAccuracy(getPrediction(x,minimum),ones(m))
print pred1, ' implemented classifier accuracy' 
print pred2, ' given answer accuracy'            
print temp1
print minimum
print linalg.norm(temp1-minimum)

sys.exit()
