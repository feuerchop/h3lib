import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import MDS
from sklearn.svm import SVC
from utils import plot_utils
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits
import random

# preconfigure dataset
# ----- read MyriadHub data from JSON file
#mydata = pickle.load(open('../feed_dataset.p', 'rb'))
#Xtr = mydata['vectors'].todense()
#ytr = np.array(mydata['labels'])
## multiple labels
#original_labels = ytr.copy()
## binary labels
#ytr[ytr!=0] = 1
#ytr[ytr==0] = -1
#print "here1"

 #test digits dataset
digits = load_digits()
Xtr = digits.data
original_labels = digits.target

# initialize classifiers
#clf_svm_part = SVC(kernel='rbf', C=20, gamma=20, class_weight='auto') #n_iter = 1
#clf_svm = SVC(kernel='rbf', C=20, gamma=20, class_weight='auto')
#clf_gaussnb = GaussianNB()
#clf_randforest = RFClassifier(n_estimators=10)
#clf_logreg = SGDClassifier(loss="log", warm_start=True, alpha=0.0001)
print "here2"

# model selection for all classifiers 
#model_selector_svm_part =  ?
#model_selector_svm = GridSearchCV(estimator=clf_svm, param_grid=dict(C=np.logspace(0.1,10,10),
#                                                      gamma=np.logspace(0.1,10,10)),
#                      verbose=1)
#model_selector_gaussnb = GridSearchCV(estimator=clf_gaussnb, param_grid=dict(),
#                      verbose=1)
#model_selector_randforest = GridSearchCV(estimator=clf_randforest, param_grid=dict(n_estimators=range(2,21)),
#                      verbose=1)
#model_selector_logreg = GridSearchCV(estimator=clf_logreg, param_grid=dict(alpha=np.logspace(0.0001,50,10)),
#                      verbose=1)
print "here3"

# start fitting classifiers
#model_selector_svm_part.partial_fit(Xtr, original_labels)
#model_selector_svm.fit(Xtr, original_labels)
#model_selector_gaussnb.fit(Xtr, original_labels)
#model_selector_randforest.fit(Xtr, original_labels)
#model_selector_logreg.fit(Xtr, original_labels)
print "here4"

# get best classifiers
#clf_svm_part = model_selector_svm_part.best_estimator_
#clf_svm = model_selector_svm.best_estimator_
#clf_gaussnb = model_selector_gaussnb.best_estimator_
#clf_randforest = model_selector_randforest.best_estimator_
#clf_logreg = model_selector_logreg.best_estimator_
print "here5"

# output best classifiers' scores
#print '{:25s}| Best model CV score = {:.3f}'.format(type(clf_svm_part).__name__, model_selector_svm_part.best_score_)
#print '{:25s}| Best model CV score = {:.3f}'.format(type(clf_svm).__name__, model_selector_svm.best_score_)
#print '{:25s}| Best model CV score = {:.3f}'.format(type(clf_gaussnb).__name__, model_selector_gaussnb.best_score_)
#print '{:25s}| Best model CV score = {:.3f}'.format(type(clf_randforest).__name__, model_selector_randforest.best_score_)
#print '{:25s}| Best model CV score = {:.3f}'.format(type(clf_logreg).__name__, model_selector_logreg.best_score_)

# fit on all data
# clf.fit(Xtr, original_labels)
# clf.partial_fit( ?? )
