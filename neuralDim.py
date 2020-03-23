
import warnings
warnings.filterwarnings("ignore")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Do stuff here

import numpy as np  
import util
from sklearn.preprocessing import StandardScaler
  
from numpy import array
import pandas as pd
from time import time 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support

from sklearn.neural_network import MLPClassifier
# from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
   
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
 
 
import data
import plotter
import searcher
import sklearn
import dimRedu
 
from sklearn.model_selection import cross_val_score

 
def adult(package, iterations, title): 
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest 
  
    params = {'activation': 'relu', 'learning_rate': 'invscaling', 'solver': 'lbfgs'} 
 
    input = package.features.shape[1]
    input = int(.7 * input)  
    (input,7,2) 
    clf = MLPClassifier(hidden_layer_sizes = (input,7,2))
    clf.set_params(**params)
    plotNetwork(clf, iterations, 'Adult {0}'.format(title), package)   
    return 
 
   

def heart(package, iterations, title): 
    xTrain = package.xTrain
    xTest = package.xTest 
    yTrain = package.yTrain
    yTest = package.yTest
  
    params = {'activation': 'tanh', 'learning_rate': 'constant', 'solver': 'adam'}
 
 
    input = package.features.shape[1]
    input = int(.7 * input) 
 
  
    clf = MLPClassifier(hidden_layer_sizes = (input,5,2))
    clf.set_params(**params)
    plotNetwork(clf, iterations, 'Heart {0}'.format(title), package)   

    return
 

def plotNetwork(clf, iterations, title, data):
    xTrain = data.xTrain
    xTest = data.xTest 
    yTrain = data.yTrain.ravel()
    yTest = data.yTest.ravel()

    x = []
    y = []  
    yT = []
    timeData = []
    devSingle = []
    print(title)
    for i in iterations: 
        print(i)
        s = time()
        clf.max_iters=i
        clf.max_iter=i

        allscores = []
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(xTrain, yTrain):
            crossXTrain, crossXTest = xTrain[train_index], xTrain[test_index]
            crossYTrain, crossYTest = yTrain[train_index], yTrain[test_index]


            clf.fit(crossXTrain, crossYTrain)
            cross_preds = clf.predict(crossXTest)
            prec, rec, f1, sup = precision_recall_fscore_support(crossYTest, cross_preds, average='macro')
            accScore = accuracy_score(crossYTest, cross_preds, True)
            
            
            REAL_TEST_PREDICTIONS = clf.predict(xTest)
            testprec, testrec, testf1, testsup = precision_recall_fscore_support(yTest, REAL_TEST_PREDICTIONS, average='macro')
            testaccScore = accuracy_score(yTest, REAL_TEST_PREDICTIONS, True)
            allscores.append([prec, rec, f1, accScore, testprec, testrec, testf1, testaccScore])
        e = time()
        timeData.append(int(e - s))

        allscores = np.array(allscores)
        f1 = allscores[:,2]
        testf1 = allscores[:,6]
        # accScore = cross_val_score(clf, X_train, y_train, n_jobs=1)
        mean = np.mean(f1)
        dev = np.std(f1)
        score = np.mean(testf1)

        devSingle.append(dev)
        x.append(i)
        y.append(mean)   
        yT.append(score)
        

    
    dev = np.std(y, axis=0)  
    dev = np.array(devSingle)
    fig, ax = plt.subplots()
    plt.title(title) 
    ax.plot(x, y, label='Train Score')  
    ax.plot(x, yT, label='Test Score')  
    ax.fill_between(x, y - dev,  y + dev, alpha=0.1)

    plt.legend(loc='best')
    color = 'tab:blue'
    ax.set_ylabel('Score', color=color)  # we already handled the x-label with ax1
    ax.set_xlabel('Iterations')
        
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis 
    color = 'tab:green'
    ax2.set_ylabel('Time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, timeData, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

 
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('{0}\{1}.png'.format('C:\\Users\\mwest\\Desktop\\ML\\source\\A3\\graphs\\neural', title))
    # plt.show()
    return

def unprocess(package):
    package.xTrain = package.Unprocessed_xTrain
    package.yTrain = package.Unprocessed_yTrain
    package.xTest = package.Unprocessed_xTest
    package.yTest = package.Unprocessed_yTest
    return package

def run(): 
    dataType = 'heart'
    package = data.createData(dataType) 

    iterations = range(599, 699)
    
    heart(package, iterations, 'Baseline') 

    package.xTrain = dimRedu.getPCAData(package.xTrain, 'Heart')
    package.xTest = dimRedu.getPCAData(package.xTest, 'Heart')
    heart(package, iterations, 'PCA') 

    package = data.createData(dataType) 
    package.xTrain = dimRedu.getICAData(package.xTrain, 'Heart')
    package.xTest = dimRedu.getICAData(package.xTest, 'Heart')
    heart(package, iterations, 'ICA') 

    package = data.createData(dataType) 
    package.xTrain = dimRedu.getRCAData(package.xTrain, 'Heart')
    package.xTest = dimRedu.getRCAData(package.xTest, 'Heart')
    heart(package, iterations, 'RCA') 

    package = data.createData(dataType) 
    package = unprocess(package)
    package.xTrain = dimRedu.getFAMDData(package.Unprocessed_xTrain, 'Heart')
    package.xTest = dimRedu.getFAMDData(package.Unprocessed_xTest, 'Heart')
    heart(package, iterations, 'FAMD') 
 

    dataType = 'adult'
    package = data.createData(dataType) 
    iterations = range(799, 899)

    adult(package, iterations, 'Baseline') 

    package.xTrain = dimRedu.getPCAData(package.xTrain, 'Adult')
    package.xTest = dimRedu.getPCAData(package.xTest, 'Adult')
    adult(package, iterations, 'PCA') 

    package = data.createData(dataType) 
    package.xTrain = dimRedu.getICAData(package.xTrain, 'Adult')
    package.xTest = dimRedu.getICAData(package.xTest, 'Adult')
    adult(package, iterations, 'ICA') 

    package = data.createData(dataType) 
    package.xTrain = dimRedu.getRCAData(package.xTrain, 'Adult')
    package.xTest = dimRedu.getRCAData(package.xTest, 'Adult')
    adult(package, iterations, 'RCA') 

    package = data.createData(dataType) 
    package = unprocess(package)
    package.xTrain = dimRedu.getFAMDData(package.Unprocessed_xTrain, 'Adult')
    package.xTest = dimRedu.getFAMDData(package.Unprocessed_xTest, 'Adult')
    adult(package, iterations, 'FAMD') 

    #     # iterations = [599, 6000]
    #     heart(package, iterations) 
    # else:    
    #     iterations = range(1, 800)
    #     iterations = [799, 800]
    #     adult(package, iterations) 
    return


    

if __name__ == '__main__':
    run()