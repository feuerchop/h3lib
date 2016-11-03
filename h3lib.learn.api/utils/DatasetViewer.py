"""
This utility helper file is used for visualizing a dataset from .csv file.
The main usage of this file is for easy check how the training dataset looks like,
and to give developers a quick overview of the data they are dealing with.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
import getopt,sys,os.path
import plot_utils as pltool

class DatasetViewer(object):
    '''
    A class for dataset viewer used to analyse features in data
    set straightforwardly
    '''
    def __init__(self):
        pass

    def view_features_distribution(self, X):
        n,d = X.shape
        indices = []
        for i in np.arange(d):
            indices.extend((i+1)*np.ones(n))
        plt.gca().scatter(np.asarray(indices), X.T.flatten(),
                          s=5, marker='.', alpha=0.5,facecolor='None',edgecolor='k')
        plt.gca().plot([1,d],[0,0],'k-')
        plt.gca().set(xlabel='feature id',
                      ylabel='feature value',
                      title='Feature distribution',
                      xlim=[0,d-1])

    def project2d(self, X, y=None, method='pca', g=0.5):
        '''
        Project high-dimensiona data to 2D for visulaization
        using methods e.g., pca, kernel-pca, mds
        :param X: N*d dataset
        :param y: labels/None if not given
        :param method: string in ['pca','kpca','mds']
        :return: projected dataset X_project
        '''
        if y is not None and y.size != X.shape[0]:
            exit("Data dims are not matched!")
        else:
            n_comp = 2
            if method == 'pca':
                projector = PCA(n_components=n_comp)
                X_proj = projector.fit_transform(X)
            elif method == 'kpca':
                projector = KernelPCA(n_components=n_comp,kernel='rbf',gamma=g)
                X_proj = projector.fit_transform(X)
            elif method == 'mds':
                projector = MDS(n_components=n_comp)
                X_proj = projector.fit_transform(X)
            else:
                print 'No projector found!'
                X_proj = X

        if y is not None:
            plt.gca().scatter(X_proj[:,0], X_proj[:,1], c=y, alpha=0.4)
        else:
            plt.gca().scatter(X_proj[:,0], X_proj[:,1])
        plt.gca().set(xlabel='1st PC.',
                      ylabel='2nd PC.',
                      title=method)

    def plot_corr(self, X, names=None):
        n,d = X.shape
        xcorr = np.corrcoef(X.T)
        XX,YY = np.meshgrid(np.arange(d), np.arange(d))
        a1 = plt.scatter(XX.ravel(), YY.ravel(),
                          s=15,
                          c=xcorr.ravel(), cmap='jet', alpha=0.7)
        plt.gca().set(title='Correlations of features',
                      xlim=[0,d-1],
                      ylim=[0,d-1])
        plt.colorbar(ax=plt.gca())
        if names is not None and len(names) == d:
            a1.set_xticklabels(names, rotation=90)
            a1.set_yticklabels(names)

        pltool.setAxSquare(plt.gca())

    def visualize(self, X, y, names= None):
        plt.subplot(2,2,1)
        self.view_features_distribution(X)
        pltool.setAxSquare(plt.gca())
        plt.subplot(2,2,2)
        self.project2d(X,y,method='kpca')
        pltool.setAxSquare(plt.gca())
        plt.subplot(2,2,3)
        self.project2d(X,y,method='pca')
        pltool.setAxSquare(plt.gca())
        plt.subplot(2,2,4)
        self.plot_corr(X, names)
        pltool.setAxSquare(plt.gca())
        plt.show()

if __name__ == '__main__':
    def main(argv):
        skip_rows = 0
        skip_cols = 0
        l_col = 0
        try:
            opts, args = getopt.getopt(argv, ['h:'], ["help",
                                                      "skip_rows=",
                                                      "skip_cols=",
                                                      "label_col="])
            for opt, arg in opts:
                if opt in ['-h', "--help"]:
                    usage()
                elif opt == "skip_rows":
                    skip_rows = int(arg)
                elif opt == "skip_cols":
                    skip_cols = arg
                elif opt == "label_col":
                    l_col = int(arg)
            filename = args[-1]
            if os.path.isfile(filename) is False:
                print 'No such file!'
                usage()
                exit(1)
            else:
                X = np.loadtxt(filename, dtype=float, delimiter=',', skiprows=skip_rows)
                y = None
                n,d = X.shape
                if l_col > d or l_col < -1:
                    print 'Label column is wrongly set, default to no labels col'
                    viewer = DatasetViewer()
                    viewer.visualize(X[:-1], y=None)
                elif l_col == -1:
                    y = X[-1]
                    viewer = DatasetViewer()
                    viewer.visualize(X[:-1],y)
                else:
                    y = X[l_col-1]
                    viewer = DatasetViewer()
                    viewer.visualize(np.c_[X[:l_col-1], X[l_col:]],y)
        except:
            usage()
            exit(2)

    def usage():
        print '''
        python DatasetViewer.py [options] *.csv
        [options] -s or --skip integer : skip # headlines
                  --label_col integer : which column is labels, -1 for the last row, default no labels
                  -h or --help : print usage
        '''
    main(sys.argv[1:])



