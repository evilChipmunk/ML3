
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import timeit
from sklearn.metrics import plot_confusion_matrix, confusion_matrix


import data 
import matplotlib.cm as cm 
import matplotlib.pyplot as plt

from sklearn.metrics import homogeneity_score,   silhouette_score,  silhouette_samples

import dimRedu

np.random.seed(0)
baseGraphPath = 'C:\\Users\\mwest\\Desktop\\ML\\source\\A3\\graphs\\'
showPlot = False

 
def my_plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, info=""):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.suptitle((title),  fontsize=14, fontweight='bold')

    plt.title(info)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
 
def CreateClusterScoreHeader():
    return  ['DataType', 'K', 'IsProcessed', 'WCSS', 'Homogeneity', 'Silhouette', 'Iterations']


def CreateEMScoreHeader():
    return  ['DataType', 'K', 'IsProcessed', 'Homogeneity', 'Silhouette', 'AIC', 'BIC', 'Iterations']


def SaveScores(header, name, processedScores): 
    # df = pd.DataFrame(processedScores, columns=header) 
    # df.to_csv(baseGraphPath + 'scores\\{0} scores.csv'.format(name))
    a = 4
 


def CreateKMeanScores(km, labels, isProcessed, title, k, xTrain, yTrain, xTest, yTest, predictions):  
    avScore = 'macro'
    scores = [title, k, isProcessed 
    , km.inertia_
    , '{0:.3}'.format(homogeneity_score(yTrain, labels)) 
    , '{0:.3}'.format(silhouette_score(xTrain, labels, metric='euclidean')) 
    ,  km.n_iter_
              ]
    return scores  


def CreateEMScores(em, labels, isProcessed, title, k, xTrain, yTrain, xTest, yTest, predictions):  
    avScore = 'macro'
    scores = [title, k, isProcessed 
    , '{0:.3}'.format(homogeneity_score(yTrain, labels)) 
    , '{0:.3}'.format(silhouette_score(xTrain, labels, metric='euclidean')) 
    , '{0:.3}'.format(em.aic(xTrain))
    , '{0:.3}'.format(em.bic(xTrain)) 
    ,  em.n_iter_
              ]  
    return scores  

 
def Silhouette(km, cluster_labels, silhouette_avg, X, y, clusterCount, title):

    n_clusters = clusterCount
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
  
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = km.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
 
    plt.suptitle(("Silhouette analysis for KMeans - Avg Score = " + str(silhouette_avg) + " clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    if showPlot:
        plt.show()
    else:
        name = baseGraphPath + title + '.png'
        plt.savefig(name)


def Silhouette_With_Multiple_Clusters(package):

    X = package.xTrain
    y = package.yTrain

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns

        fig = plt.figure()
        ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=4)  # topleft

        ax2 = plt.subplot2grid((4, 4), (0, 1))                # right
        ax3 = plt.subplot2grid((4, 4), (0, 2))                # right
        ax4 = plt.subplot2grid((4, 4), (0, 3))                # right

        ax5 = plt.subplot2grid((4, 4), (1, 1))                # right
        ax6 = plt.subplot2grid((4, 4), (1, 2))                # right
        ax7 = plt.subplot2grid((4, 4), (1, 3))                # right

        ax8 = plt.subplot2grid((4, 4), (2, 1))                # right
        ax9 = plt.subplot2grid((4, 4), (2, 2))                # right
        ax10 = plt.subplot2grid((4, 4), (2, 3))                # right

        ax11 = plt.subplot2grid((4, 4), (3, 1))                # right
        ax12 = plt.subplot2grid((4, 4), (3, 2))                # right
        ax13 = plt.subplot2grid((4, 4), (3, 3))                # right

        axes = [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13]

        # ax1 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2) # topleft
        # ax3 = plt.subplot2grid((3,3), (0,2), rowspan=3)            # right
        # ax4 = plt.subplot2grid((3,3), (2,0))                       # bottom left
        # ax5 = plt.subplot2grid((3,3), (2,1))                       # bottom right
        fig.tight_layout()

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        featureCount = 12
        for ii in range(featureCount):
            plotAx(clusterer, colors, axes, X, ii)

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


def plotAx(clusterer, colors, axes, X, i):
    ax = axes[i]
    ax.scatter(X[:, 0], X[:, i], marker='.', s=30, lw=0, alpha=0.7,
               c=colors, edgecolor='k')
 
    centers = clusterer.cluster_centers_ 
    ax.scatter(centers[:, 0], centers[:, i], marker='o',
               c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                   s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature {0}".format(i))

  
def runClusterPackage(k, package, title, classNames):

    xTrain = package.xTrain
    yTrain = package.yTrain
    xTest = package.xTest
    yTest = package.yTest

    processedScores = []
    km = KMeans(n_clusters=k, random_state=10)
    cluster_labels = km.fit_predict(xTrain) 
    predictions = km.predict(xTrain)
 
    scores = CreateKMeanScores(km, cluster_labels, True, title, k, xTrain, yTrain, xTest, yTest, predictions)
    # Silhouette(km, cluster_labels, scores[7], xTrain, yTrain, k, '{0} Processed'.format(title))
    processedScores.extend(scores)  
  
    return processedScores
 
def runEMPackage(k, package, title, classNames):

    xTrain = package.xTrain
    yTrain = package.yTrain
    xTest = package.xTest
    yTest = package.yTest

    processedScores = [] 
    em = GaussianMixture(n_components=k, covariance_type='diag', n_init=1, warm_start=True, random_state=10) 
    em_labels = em.fit_predict(xTrain)  
    predictions = em.predict(xTrain)

    scores = CreateEMScores(em, em_labels, True, title, k, xTrain, yTrain, xTest, yTest, predictions)
    # Silhouette(em, em_labels, scores[7], xTrain, yTrain, k, '{0} Processed'.format(title))
    processedScores.extend(scores)  

    return processedScores

def runAll(package, classes, topRange, dataType, dim):

    title = 'KM {0} - {1}'.format(dataType, dim)
    print(title)
    header = CreateClusterScoreHeader() 
    processedScores = [] 
    for k in range(2, topRange): 
        print(k)
        p = runClusterPackage(k, package, dataType, classes) 
        processedScores.append(p) 
    SaveScores(header, 'KM {0} - {1}'.format(dataType, dim), processedScores)

    
    title = 'EM {0} - {1} '.format(dataType, dim)
    print(title)
    header = CreateEMScoreHeader()
    processedScores = [] 
    for k in range(2, topRange): 
        print(k) 
        p = runEMPackage(k, package, dataType, classes) 
        processedScores.append(p) 
    SaveScores(header, 'EM {0} - {1}'.format(dataType, dim), processedScores)


def unprocess(package):
    package.xTrain = package.Unprocessed_xTrain
    package.yTrain = package.Unprocessed_yTrain
    package.xTest = package.Unprocessed_xTest
    package.yTest = package.Unprocessed_yTest
    return package
 

def run():
    # adultPackage = unprocess( data.createData('adult'))
    # heartPackage = unprocess(data.createData('heart') )
 
    adultPackage = data.createData('adult')
    heartPackage = data.createData('heart') 

    topRange = 5
    adultClasses =  ['>50K', '<=50K'] 
    heartClasses = ['Diameter narrowing ', 'Diameter not narrowing']

    runAll(adultPackage, adultClasses, topRange, 'Adult', 'None')
    runAll(heartPackage, heartClasses, topRange, 'Heart', 'None')

    
    # adultPackage.xTrain = dimRedu.getPCAData(adultPackage.xTrain, 'Adult')
    # heartPackage.xTrain = dimRedu.getPCAData(heartPackage.xTrain, 'Heart')
    # runAll(adultPackage, adultClasses, topRange, 'Adult', 'PCA')
    # runAll(heartPackage, heartClasses, topRange, 'Heart', 'PCA') 

    # adultPackage = data.createData('adult')
    # heartPackage = data.createData('heart')  
    # adultPackage.xTrain = dimRedu.getICAData(adultPackage.xTrain, 'Adult')
    # heartPackage.xTrain = dimRedu.getICAData(heartPackage.xTrain, 'Heart')
    # runAll(adultPackage, adultClasses, topRange, 'Adult', 'ICA')
    # runAll(heartPackage, heartClasses, topRange, 'Heart', 'ICA')
 
    # adultPackage = data.createData('adult')
    # heartPackage = data.createData('heart')  
    # adultPackage.xTrain = dimRedu.getRCAData(adultPackage.xTrain, 'Adult')
    # heartPackage.xTrain = dimRedu.getRCAData(heartPackage.xTrain, 'Heart')
    # runAll(adultPackage, adultClasses, topRange, 'Adult', 'RCA')
    # runAll(heartPackage, heartClasses, topRange, 'Heart', 'RCA')
    
    # adultPackage = unprocess(adultPackage)
    # heartPackage = unprocess(heartPackage)
    # adultPackage.xTrain = dimRedu.getFAMDData(adultPackage.xTrain, 'Adult')
    # heartPackage.xTrain = dimRedu.getFAMDData(heartPackage.xTrain, 'Heart')
    # runAll(adultPackage, adultClasses, topRange, 'Adult', 'FAMD')
    # runAll(heartPackage, heartClasses, topRange, 'Heart', 'FAMD')

    # dimRedu.run(heartPackage, 'Heart')
    # dimRedu.run(adultPackage, 'Adult')
 

    print('done')
 

if __name__ == '__main__':
    run()