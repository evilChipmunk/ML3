
import numpy as np
from numpy import array
import pandas as pd 
 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold 
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.decomposition import PCA
import util


# rows = 1000
rows = 1000
# rows = 100000
# rows = 1000000

class DataPackage:
    def __init__(self):
        self.features = None
        self.predictions = None 
        self.allData = None 
        self.xTrain = None 
        self.xTrain = None 
        self.yTrain = None
        self.yTest = None 
        self.Unprocessed = None
        self.Unprocessed_features = None
        self.Unprocessed_predictions = None  
        self.Unprocessed_xTrain = None 
        self.Unprocessed_xTest = None 
        self.Unprocessed_yTrain = None
        self.Unprocessed_yTest = None 
 

baseData = 'Data\\'
baseData = 'C:\\Users\\mwest\\Desktop\\ML\\source\\Data\\'

def readData(dataType, columns=None):
    import os
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if columns:
       df = pd.read_csv(baseData + dataType + '.csv' , usecols=columns) 
    df = pd.read_csv(baseData + dataType + '.csv') 
    sampleRows = rows
    if sampleRows > df.shape[0]:
        sampleRows = df.shape[0]
    return df.sample(sampleRows, replace=False, random_state=util.randState)

def createData(dataType='Adult', resampleData=False, stratify=True, describe=False):

    data = None
    if str.lower(dataType) == 'heart':
        data = createHeartData(dataType, describe) 
    else:
        data = createAdultData(dataType, describe)
  

    if resampleData:
        dat = data[2]
        label = dat._values[:,-1]
        classCount = []
        classes = np.unique(label)
        for c in classes:
            classCount.append([c, label[label == c].shape[0]])
        classCount = sorted(classCount, key= lambda x: x[0])
        maxLabel = classCount[-1]
        maxClass = maxLabel[0]
        maxCount = int(maxLabel[1])
        # maxData = dat.loc[dat['LOS'] == maxClass]
        maxData = dat[dat.iloc[:, -1] == maxClass] 

        samples = []
        for c in classCount:
            minClass = c[0]
            if minClass != maxClass:
                # minData = dat.loc[dat['LOS'] == minClass] 
                minData = dat[dat.iloc[:, -1] == minClass] 
                sampled = resample(minData, replace=True, n_samples= maxCount, random_state=util.randState) 
                samples.append(sampled)
 
        for sample in samples:
            maxData = pd.concat([maxData, sample]) 
 
 
        label = maxData._values[:,-1]
        classCount = []
        classes = np.unique(label)
        for c in classes:
            classCount.append([c, label[label == c].shape[0]])
        data = (data[0], data[1], maxData)


    package = DataPackage()
    package.features = data[0]
    package.predictions = data[1]
    package.allData = data[2]
    package.Unprocessed = data[3]
    package.Unprocessed_features = data[4]
    package.Unprocessed_predictions = data[5]
  
    xTrain, xTest, yTrain, yTest = split(package.features, package.predictions, stratify)
    package.xTrain = xTrain
    package.xTest = xTest
    package.yTrain = yTrain
    package.yTest = yTest
    package.yTrain.shape = (package.yTrain.shape[0])
    package.yTest.shape = (package.yTest.shape[0])

    
    xTrain, xTest, yTrain, yTest = split(package.Unprocessed_features, package.Unprocessed_predictions, stratify)
    package.Unprocessed_xTrain = xTrain
    package.Unprocessed_xTest = xTest
    package.Unprocessed_yTrain = yTrain
    package.Unprocessed_yTest = yTest
    package.Unprocessed_yTrain.shape = (package.Unprocessed_yTrain.shape[0])
    package.Unprocessed_yTest.shape = (package.Unprocessed_yTest.shape[0])


    (unique, counts) = np.unique(yTrain, return_counts=True)
    total = counts.sum()

    print('Train features {0} -\t count: {1}'.format(xTrain.shape[1], xTrain.shape[0]))
    print('Train classes count: {0}'.format(unique.shape))
    for i in range(counts.shape[0]):
        count = counts[i]
        print('Train class {0}: -\t {1}  %{2}'.format(unique[i], count, count/total))


    # print('{0} shape: {1}'.format(dataType, data[2].shape))
    # print(data[2].head()) 
    return package 

def describeData(df: pd.DataFrame, dataType):

    # https://github.com/pandas-profiling/pandas-profiling
    # https://stackoverflow.com/questions/22235245/calculate-summary-statistics-of-columns-in-dataframe

    import matplotlib.pyplot as plt 
    from pandas_profiling import ProfileReport
    # profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
    profile = ProfileReport(df)
    profile.to_file(output_file="{0} Profile.html".format(dataType))


def createHeartData(dataType, describe):
    # 1. #3 (age) 
    # 2. #4 (sex) 
    # 3. #9 (cp) 
    # 4. #10 (trestbps) 
    # 5. #12 (chol) 
    # 6. #16 (fbs) 
    # 7. #19 (restecg) 
    # 8. #32 (thalach) 
    # 9. #38 (exang) 
    # 10. #40 (oldpeak) 
    # 11. #41 (slope) 
    # 12. #44 (ca) 
    # 13. #51 (thal) 
    # 14. #58 (num) (the predicted attribute) 
 
    # 3 age: age in years 
    # 4 sex: sex (1 = male; 0 = female) 
    # 9 cp: chest pain type 
    # -- Value 1: typical angina 
    # -- Value 2: atypical angina 
    # -- Value 3: non-anginal pain 
    # -- Value 4: asymptomatic 
    # 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
    # 12 chol: serum cholestoral in mg/dl 
    # 16 fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
    # 19 restecg: resting electrocardiographic results 
    # -- Value 0: normal 
    # -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) 
    # -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria  
    # 32 thalach: maximum heart rate achieved 
    # 38 exang: exercise induced angina (1 = yes; 0 = no) 
    # 40 oldpeak = ST depression induced by exercise relative to rest 
    # 41 slope: the slope of the peak exercise ST segment 
    # -- Value 1: upsloping 
    # -- Value 2: flat 
    # -- Value 3: downsloping 
    # 44 ca: number of major vessels (0-3) colored by flourosopy 
    # 51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect 
    # 58 num: diagnosis of heart disease (angiographic disease status) 
    # -- Value 0: < 50% diameter narrowing 
    # -- Value 1: > 50% diameter narrowing  
    columns = ['age', 'sex', 'cp','trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    columns = None
    df = readData(dataType, columns)  
    if describe:
        describeData(df, dataType)
    df = df.fillna(0)  

    
    columns_to_encode = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target' ]
    columns_to_scale  = ['age',  'trestbps', 'chol', 'thalach', 'oldpeak']
 
    return transform(dataType, df, columns_to_scale, columns_to_encode)
    # # Instantiate encoder/scaler
    # scaler = StandardScaler()
    # ohe    = OneHotEncoder(sparse=False) 
    # scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 
    # encoded_columns =    ohe.fit_transform(df[columns_to_encode]) 
    # processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1) 
    # features = pd.DataFrame(processed_data[:, 0:-2])
    # predictions = pd.DataFrame(processed_data[:, -1])    
    # allData = pd.DataFrame(processed_data) 
    
    # unprocessed = df.values
    # unProcessedfeatures = pd.DataFrame(unprocessed[:, 0:-2])
    # unProcessedpredictions = pd.DataFrame(unprocessed[:, -1]) 


    # pca = PCA(n_components=15) 
    # features = pca.fit(features).transform(features)
    # print('PCA explained for {0}: {1}'.format(dataType, pca.explained_variance_ratio_.sum()))
    # print(pca.explained_variance_ratio_)

    # allData = np.concatenate([features, predictions], axis=1)

    # return (features, predictions.values, allData, unprocessed, unProcessedfeatures, unProcessedpredictions)


def transform(dataType, df, columns_to_scale, columns_to_encode):    # Instantiate encoder/scaler
    scaler = StandardScaler()
    ohe    = OneHotEncoder(sparse=False) 
    scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 
    encoded_columns =    ohe.fit_transform(df[columns_to_encode]) 
    processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1) 
    # features = pd.DataFrame(processed_data[:, 0:-1])
    # predictions = pd.DataFrame(processed_data[:, -1])    
    features = processed_data[:, 0:-1]
    predictions = processed_data[:, -1] 
    predictions.shape = (predictions.shape[0], 1)

    allData = pd.DataFrame(processed_data) 
     
    unprocessed = df.values    
    unProcessedfeatures = unprocessed[:, 0:-1]
    unProcessedpredictions = unprocessed[:, -1]
    # unProcessedfeatures = features.values.copy()
    # unProcessedpredictions = predictions.values.copy()


    # pca = PCA(n_components=15) 
    # features = pca.fit(features).transform(features)
    # print('PCA explained for {0}: {1}'.format(dataType, pca.explained_variance_ratio_.sum()))
    # print(pca.explained_variance_ratio_)

    allData = np.concatenate([features, predictions], axis=1)

    return (features, predictions, allData, unprocessed, unProcessedfeatures, unProcessedpredictions)


    # features = df.iloc[:, 0:-1]
    # predictions = df.iloc[:, -1] 
  
    # features = pd.get_dummies(features, drop_first=True)
  
    # allData = features.copy()  
    
    # idx = allData.shape[1]
    # allData.insert(loc=idx, column='AtRisk', value=predictions.values)

    # return (features, predictions, allData)

def createAdultData(dataType, describe):

    # >50K, <=50K.

    # age: continuous.
    # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    # fnlwgt: continuous.
    # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    # education-num: continuous.
    # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    # sex: Female, Male.
    # capital-gain: continuous.
    # capital-loss: continuous.
    # hours-per-week: continuous.
    # native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


    columns_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    columns_to_scale  = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country 


    columns = ['age'
                ,'workclass'
                ,'fnlwgt'
                ,'education'
                ,'education-num'
                ,'marital-status'
                ,'occupation'
                ,'relationship'
                ,'race'
                ,'sex'
                ,'capital-gain'
                ,'capital-loss'
                ,'hours-per-week'
                ,'native-country'
                , 'IncomeType']
  
    columns.remove('education')
    # columns.remove('native-country')
    columns_to_encode.remove('education')
    # columns_to_encode.remove('native-country')
    df = readData(dataType, columns = columns) 
    if describe:
        describeData(df, dataType)
    return transform(dataType, df, columns_to_scale, columns_to_encode)
    # # df = df.fillna(0)  
    # # features = pd.get_dummies(features, drop_first=True)
  
    # # Instantiate encoder/scaler
    # scaler = StandardScaler()
    # ohe    = OneHotEncoder(sparse=False)

    # # Scale and Encode Separate Columns
    # scaled_columns  = scaler.fit_transform(df[columns_to_scale]) 
    # encoded_columns =    ohe.fit_transform(df[columns_to_encode])

    # # Concatenate (Column-Bind) Processed Columns Back Together
    # processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)
 
    # # features = processed_data.iloc[:, 0:-2]
    # # predictions = processed_data.iloc[:, -1] 
     
    # features = pd.DataFrame(processed_data[:, 0:-2])
    # predictions = pd.DataFrame(processed_data[:, -1]) 
    
    # unprocessed = df.values
    # unProcessedfeatures = pd.DataFrame(unprocessed[:, 0:-2])
    # unProcessedpredictions = pd.DataFrame(unprocessed[:, -1]) 

    # # allData = features.copy()
    # # allData['IncomeType'] = predictions   
    # allData = pd.DataFrame(processed_data)
    # # print(features.cov())
    # # print(features.corr())
 

    
    # # b = np.var(features)
    # # print(b)
    # pca = PCA(n_components=15)
    # pca = PCA(n_components=20)
    # features = pca.fit(features).transform(features)
    # print('PCA explained for {0}: {1}'.format(dataType, pca.explained_variance_ratio_.sum()))
    # print(pca.explained_variance_ratio_)

    # allData = np.concatenate([features, predictions], axis=1)

    # return (features, predictions.values, allData, unprocessed, unProcessedfeatures, unProcessedpredictions)
 

def split(features, predictions, stratify):
    testSize = .20 
    if (stratify):  
        xTrain, xTest, yTrain, yTest = train_test_split(
            features, predictions, test_size=testSize, random_state=util.randState, stratify=predictions)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(
            features, predictions, test_size=testSize, random_state=util.randState)
        
    
    return (xTrain, xTest, yTrain, yTest)
 