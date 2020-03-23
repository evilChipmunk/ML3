
import warnings
warnings.filterwarnings("ignore")
 
import data 
 
if __name__ == '__main__':
 

    import plotclusterscores
    plotclusterscores.run()
    import dimRedu
    datatypes = ['adult', 'heart']
    for d in datatypes:
        package = data.createData(d) 
        dimRedu.run(package, d)

    import neuralDim
    neuralDim.run()

    import cluster 
    cluster.run()
 
 