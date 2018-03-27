# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:21:24 2018

@author: ning

This script is to do two things, 
    1. converting epochs to power spectrograms
    2. fit and test a non-linear model to the data
    3. optimize the non-linear model's hyperparameters
"""

from mne.decoding import Vectorizer
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import numpy as np
from sklearn import metrics
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import pandas as pd
from scipy import stats as stats
from scipy.stats import randint as sp_randint

os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
# define data working directory and the result saving directory
working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\SpindleClassification_DeepConvolutionalNeuralNets\\Baseline models\Results\\'
"""
# this part of the code is commented out because i don't want to run it again to optimize the hyperparameters of the randomforest classifier
#
def make_clf():
    clf = []
    clf.append(('vectorizer',Vectorizer()))
    clf.append(('estimator',RandomForestClassifier(n_estimators=100,
                                                   max_depth=50,
                                                   random_state=12345,
                                                  class_weight='balanced')))
    clf = Pipeline(clf)
    return clf
def get_best_(ii,random_=True,grid_=True,rgrid_=False):
    # load the train dataset
    X_train_,y_train_ = pickle.load(open('data/train/train%d.p'%(ii),'rb'))
    if random_:# add some random noise as part of the training data
        random_inputs = np.random.rand(X_train_.shape[0],32,16,192)
        random_labels = [0]*X_train_.shape[0] # obviously they are not spindles
        random_labels = np_utils.to_categorical(random_labels,2) # just to make sure it is similar to deep CNN
        X_train_ = np.concatenate([X_train_,random_inputs],axis=0)
        y_train_ = np.concatenate([y_train_,random_labels],axis=0)
    X_train_,y_train_ = shuffle(X_train_,y_train_)
    vectorizer = Vectorizer()
    X_train_ = vectorizer.fit_transform(X_train_)
    clf = RandomForestClassifier()
    cv = StratifiedKFold(n_splits=4,random_state=12345)
    if grid_: # finite grid search
        param_grid = {"n_estimators":[152,179,190],
                          "max_depth":[None],
                          "class_weight":['balanced'],
              "max_features": [10],
              "min_samples_split": [4, 10],
              "min_samples_leaf": [4],}

        
        grid = GridSearchCV(clf,param_grid=param_grid,cv=cv,scoring='roc_auc',)
        grid.fit(X_train_,y_train_[:,-1])
        return grid
    if rgrid_:# random grid search
        param_dist = {"n_estimators":sp_randint(50,200),
                      "class_weight":['balanced'],
            "max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),}
        rgrid = RandomizedSearchCV(clf,param_distributions=param_dist,
                                  n_iter=100,cv=cv,scoring='roc_auc',)
        rgrid.fit(X_train_,y_train_[:,-1])
        return rgrid
# take the functions above into a loop
dfs = {}
for ii in range(5): # do this 5 times
    best_params_ = []
    for jj in np.random.choice(range(10),10,replace=False): # within each time, we randomized the feeding order of the training data
        cc = get_best_(jj,grid_=False,rgrid_=True,random_=False)
        cc=pd.Series(cc.best_params_).to_frame().T
        best_params_.append(cc)
    dfs[ii] = pd.concat(best_params_)
pickle.dump(dfs,open('selected parameters (no random input).p','wb'))
temp_ = []
for a,b in dfs.items():
    temp_.append(b)
temp_ = pd.concat(temp_)
C = temp_[temp_.columns[2:]].apply(pd.to_numeric)
C.mode()
"""
def make_clf():
    clf = []
    clf.append(('vectorizer',Vectorizer()))
    # hyper parameters were optimized and here we just directly use them in the random forest model
    clf.append(('estimator',RandomForestClassifier(n_estimators=190,# number of trees
                                                   max_depth=None, # no need to specifiy the depth, in order words, feature depth
                                                   random_state=12345,
                                                  class_weight='balanced',
                                                  max_features=10, # dimension reduction
                                                  min_samples_leaf=4,# minimum feature span
                                                  min_samples_split=4)))# minimum feature split
    clf = Pipeline(clf)
    return clf
X_validation,y_validation = pickle.load(open('data/validation/validation.p','rb'))
X_test,y_test = pickle.load(open('data/test//test.p','rb'))
for ii in range(5):# more or less a 5-fold cross validation
    clf = make_clf()
    for ii in np.random.choice(range(10),10,replace=False):# within each fold, the order of feeding the training data is randomized
        X_train_,y_train_ = pickle.load(open('data/train/train%d.p'%(ii),'rb'))
        random_inputs = np.random.rand(X_train_.shape[0],32,16,192)
        random_labels = [0]*X_train_.shape[0]
        random_labels = np_utils.to_categorical(random_labels,2)
        X_train_ = np.concatenate([X_train_,random_inputs],axis=0)
        y_train_ = np.concatenate([y_train_,random_labels],axis=0)
        clf.fit(X_train_,y_train_)
        pred_ = clf.predict(X_validation)
    print(metrics.classification_report(y_validation,pred_))
X_predict_prob_ = clf.predict_proba(X_test)[1][:,-1]
X_predict_ = X_predict_prob_ > 0.5 # just for thresholding
print(metrics.classification_report(y_test[:,-1],X_predict_))# classification measurement on the test data
AUC = metrics.roc_auc_score(y_test[:,-1], X_predict_prob_)
# get the step function of false positive rate as a function of true positive rate
fpr,tpr,th = metrics.roc_curve(y_test[:,-1], X_predict_prob_,pos_label=1)
# average sensitivity and selectivity
sensitivity = metrics.precision_score(y_test[:,-1],X_predict_,average='weighted')
selectivity = metrics.recall_score(y_test[:,-1],X_predict_,average='weighted')
plt.close('all') # plot the roc curve
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(fpr,tpr,label='AUC = %.3f\nSensitivity = %.3f\nSelectivity = %.3f'%(AUC,sensitivity,selectivity))
ax.set(xlabel='false postive rate',ylabel='true positive rate',title='test data\nrandom forest',
       xlim=(0,1),ylim=(0,1))
ax.legend(loc='best')