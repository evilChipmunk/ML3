 
import numpy as np
from numpy import array
import pandas as pd
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.metrics import make_scorer, accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

randState = 42

class Score:
     def __init__(self):
         self.Accuracy = 0
         self.Precision = 0
         self.Recall = 0
         self.F1 = 0
         self.TrainAccuracy = 0
         self.TrainPrecision = 0
         self.TrainRecall = 0
         self.TrainF1 = 0
         self.Tree = None
         self.HyperParamLabel = ''
         self.HyperParam = None
         self.AverageAccuracy = 0

class ScoreList:
    def __init__(self, hyperParamLabel):
        self.scores = []
        self.hyperParamLabel = hyperParamLabel

    def Add(self, yTest, yPred, xTest, trainPred, hyperParam):
        acc = accuracy_score(yTest, yPred) 
        precision, recall, fscore, support = precision_recall_fscore_support(yTest, yPred, average='macro')  

        false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest, yPred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        
        tracc = accuracy_score(xTest, trainPred) 
        trprecision, trrecall, trfscore, trsupport = precision_recall_fscore_support(xTest, trainPred, average='macro') 
        
        false_positive_rate, true_positive_rate, thresholds = roc_curve(xTest, trainPred)
        train_roc_auc = auc(false_positive_rate, true_positive_rate)


        # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='micro') 
        # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        # labels = ['short', 'med', 'long'] 
        # precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, labels=labels)
        score = [acc, precision, recall, fscore, tracc, trprecision, trrecall, trfscore,  hyperParam]
        
        customScore = Score()
        customScore.Accuracy = score[0]
        customScore.Precision = score[1]
        customScore.Recall = score[2]
        customScore.F1 = score[3] 
        customScore.AUC = roc_auc
        
        customScore.TrainAccuracy = score[4]
        customScore.TrainPrecision = score[5]
        customScore.TrainRecall = score[6]
        customScore.TrainF1 = score[7] 
        customScore.TrainAUC = train_roc_auc

        customScore.HyperParamLabel = self.hyperParamLabel
        customScore.HyperParam = score[-1]

        self.scores.append(customScore)
  
        return customScore 

    def GetScores(self): 
        return self.scores

    def GetSortedScores():
        return sorted(self.scores, key= lambda x: x.Accuracy, reverse=True)