from sklearn.decomposition import PCA, FastICA   
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import prince
from sklearn.random_projection import SparseRandomProjection 
from scipy.stats import norm, kurtosis

path = 'C:\\Users\\mwest\\Desktop\\ML\\source\\A3\\graphs\\dim\\'

 
def chartRCA(package, dataType):  
      
    title = '{0} RCA'.format(dataType)
    componentRange = range(1, package.xTrain.shape[1] - 1) 
    values = []
    for i in componentRange:
        transformer = SparseRandomProjection(n_components=i)
        transformed = transformer.fit_transform(package.xTrain)  
        val = np.corrcoef(pairwise_distances(transformed).ravel(), pairwise_distances(package.xTrain).ravel()) 
        error = np.mean(val)
        values.append(error)
     
    plt.plot(componentRange, values) 
    plt.xlabel("Components")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.title(title)
    # plt.show() 
    plt.savefig('{0}{1}.png'.format(path, title))
    plt.clf()


def getkurtosis(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3
    return kurt, skew, var, mean

def chartICA(package, dataType):
        
 
    title = '{0} ICA'.format(dataType)
    componentRange = range(1, package.xTrain.shape[1] - 1) 
    componentRange = range(1, package.xTrain.shape[1] - 1)
    icaValues = []
    values = []
    for i in componentRange:
        transformer = FastICA(n_components=i)
        transformed = transformer.fit_transform(package.xTrain) 
        icaValues.append(kurtosis(transformed).mean())
        values.append(kurtosis(package.xTrain[:, 0:i]).mean()) 
 
    plt.plot(componentRange, icaValues, label='Mixed Features')
    plt.plot(componentRange, values, label='Unmixed Features')
    plt.xlabel("Components")
    plt.ylabel("Kurtosis Mean")
    plt.legend()
    plt.title(title)
    # plt.show() 
    plt.savefig('{0}{1}.png'.format(path, title))
    plt.clf()

 

def chartFAMD(X, y, dataType):

    
    title = '{0} FAMD'.format(dataType) 
    componentRange = range(1, X.shape[1] - 1)
    icaValues = []
    values = []

    components = len(componentRange) 
    transformer = prince.FAMD(n_components=components, n_iter=3,copy=True,check_input=True,engine='auto', random_state=42) 
    fitted = transformer.fit(X)  
    print(transformer.row_coordinates(X))

    transformed = transformer.transform(X)

    cum = np.cumsum(transformer.explained_inertia_)  


    fig, ax = plt.subplots()
    plt.title('FAMD - {0} data set'.format(dataType)) 
    plt.legend(loc='best') 

    ax.plot(componentRange, transformer.explained_inertia_, label='Variance')    
    ax.set_ylabel('Eigenvalue')  
    ax.set_xlabel('Components')
        
    ax2 = ax.twinx() 
    ax2.plot(componentRange, cum, linestyle='--', label='Cumulative Variance', color='orange')   
    ax2.set_ylabel('Explained Variance')   

  
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    fig.tight_layout()   

    # plt.show() 
    plt.savefig('{0}{1}.png'.format(path, title))
    plt.clf()
 


def chartPCA(package, dataType): 
    
    title = '{0} PCA'.format(dataType)
    transformer = PCA() 
    transformed = transformer.fit_transform(package.xTrain)  

    cum = np.cumsum(transformer.explained_variance_ratio_)
    componentRange = range(len(transformer.explained_variance_ratio_)) 
 
    fig, ax = plt.subplots()
    plt.title('PCA - {0} data set'.format(dataType)) 
    plt.legend(loc='best') 

    ax.plot(componentRange, transformer.explained_variance_ratio_, label='Variance')    
    ax.set_ylabel('Eigenvalue')  
    ax.set_xlabel('Components')
        
    ax2 = ax.twinx() 
    ax2.plot(componentRange, cum, linestyle='--', label='Cumulative Variance', color='orange')   
    ax2.set_ylabel('Explained Variance')   

  
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    fig.tight_layout()   

    # plt.show() 
    plt.savefig('{0}{1}.png'.format(path, title))
    plt.clf()
 
def getPCAData(X, dataType):
    if dataType == 'Adult':
        pca = PCA(n_components=20)  
    else:
        pca = PCA(n_components=20) 

    return pca.fit_transform(X) 

def getICAData(X, dataType):
    if dataType == 'Adult':
        components = 65
    else:
        components = 23

    transformer = FastICA(n_components=components)
    transformed = transformer.fit_transform(X) 
    return transformed

def getRCAData(X, dataType):
    if dataType == 'Adult':
        components = 40
    else:
        components = 25

 
    transformer = SparseRandomProjection(n_components=components)
    transformed = transformer.fit_transform(X)
    return transformed

def getFAMDData(X, dataType):
    
    # vals = package.Unprocessed  
    X = createFAMDDataSets(X, dataType)
    # chartFAMD(X, vals[:, -1], dataType)  

    if dataType == 'Adult': 
        components = 6
    else:
        components = 8
 
    transformer = prince.FAMD(n_components=components, n_iter=3,copy=True,check_input=True,engine='auto', random_state=42) 
    fitted = transformer.fit(X)   
    transformed = transformer.transform(X)
    return transformed.values


def createFAMDDataSets(X, dataType):
    
    vals = X 
    if dataType == 'Adult' or dataType == 'adult':  
        age = vals[:, 0].astype(np.uint8)
        workClass = vals[:, 1]
        fnlwgt = vals[:, 2].astype(np.float)
        education = vals[:, 4].astype(np.uint8)
        marital = vals[:, 5]
        occupation = vals[:, 6]
        relationship = vals[:, 7]
        race = vals[:, 8]
        sex = vals[:, 9]
        gain = vals[:, 10].astype(np.float)
        loss = vals[:, 11].astype(np.float)
        hours = vals[:, 12].astype(np.float)
        country = vals[:, 13]

        x = pd.DataFrame(age, columns=['age'])
        x['workclass'] = workClass
        x['fnlwgt'] = fnlwgt
        x['education'] = education
        x['marital'] = marital
        x['occupation'] = occupation
        x['relationship'] = relationship
        x['race'] = race
        x['sex'] = sex
        x['gain'] = gain
        x['loss'] = loss
        x['hours'] = hours
        x['country'] = country
        
    else: 
         
        age = vals[:, 0].astype(np.uint8) 
        sex = vals[:, 1].astype(np.str) 
        sex[sex == '1.0'] = 'male'
        sex[sex == '0.0'] = 'female'
        cp = vals[:, 2]
        cpString = cp.astype(np.str)
        cpString[cp == 0] = 'missing'
        cpString[cp == 1] = 'typical angina'
        cpString[cp == 2] = 'atypical angina'
        cpString[cp == 3] = 'non-anginal pain'
        cpString[cp == 4] = 'asymptomatic'
        cp = cpString 

        trestbps = vals[:, 3].astype(np.float)
        chol = vals[:, 4].astype(np.float)
        fbs = vals[:, 5].astype(np.str)

        restecg = vals[:, 6]
        restecgString = restecg.astype(np.str)
        restecgString[restecg == 0] = 'normal'
        restecgString[restecg == 1] = 'ST-T wave abnormality'
        restecgString[restecg == 2] = 'left ventricular hypertrophy' 
        restecg = restecgString 


        thalach = vals[:, 7]
        exang = vals[:, 8].astype(np.str)
        oldpeak = vals[:, 9].astype(np.str) 

        slope = vals[:, 10] 
        slopeString = slope.astype(np.str)
        slopeString[slope == 0] = 'missing'
        slopeString[slope == 1] = 'upsloping'
        slopeString[slope == 2] = 'flat' 
        slopeString[slope == 3] = 'downsloping' 
        slopeString[slope == 4] = 'unknown' 
        slope = slopeString 

        ca = vals[:, 11].astype(np.str)

        thal = vals[:, 12]
        thalString = thal.astype(np.str)
        thalString[thal == 3] = 'normal'
        thalString[thal == 6] = 'fixed defect'
        thalString[thal == 7] = 'reversable defect '  
        thal = thalString 

        x = pd.DataFrame(age, columns=['age'])
        x['sex'] = sex
        x['cp'] = cp
        x['trestbps'] = trestbps
        x['chol'] = chol
        x['fbs'] = fbs
        x['restecg'] = restecg
        x['thalach'] = thalach
        x['exang'] = exang
        x['oldpeak'] = oldpeak
        x['slope'] = slope
        x['ca'] = ca
        x['thal'] = thal
    return x

def run(package, dataType):
    xTrain = package.xTrain
    yTrain = package.yTrain

    vals = package.Unprocessed 
    
    chartPCA(package, dataType)
    chartICA(package, dataType) 
    chartRCA(package, dataType)
 

    x = createFAMDDataSets(package.Unprocessed_xTrain, dataType)
    chartFAMD(x, vals[:, -1], dataType)  
           
 