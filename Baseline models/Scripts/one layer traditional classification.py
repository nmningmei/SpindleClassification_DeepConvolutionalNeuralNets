# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:36:41 2017

@author: ning
"""

import mne
import os
import numpy as np
from mne.decoding import Vectorizer,SlidingEstimator,cross_val_multiscore
from tqdm import tqdm
from matplotlib import pyplot as plt
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated\\'
data = mne.time_frequency.read_tfrs(saving_dir+'sub30_day2-eventsRelated-tfr.h5')
data = data[0]
events = data.info['event']
labels = events[:,-1]

from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_clf():
    clf = []
#    clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    Cs = np.logspace(-3,3,7)
    estimator = LogisticRegressionCV(Cs,cv=4,scoring='roc_auc',max_iter=int(3e3),random_state=12345,class_weight='balanced')
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf
cv = StratifiedShuffleSplit(n_splits=4,random_state=12345)

#scores = []
#for ii in tqdm(range(data.data.shape[2])):
#    temp = data.data[:,:,ii,::50]
#    clf = make_clf()
#    td = SlidingEstimator(clf,scoring='roc_auc')
#    scores_=cross_val_multiscore(estimator=td,X=temp,y=labels,cv=cv)
#    scores.append(scores_)
#
#scores = np.array(scores)
#
#fig, ax = plt.subplots()
#im = ax.imshow(scores.mean(1),origin='lower',extent=[0,2001,8,16],aspect='auto')
#plt.colorbar(im,ax=ax)
#
#fig, ax = plt.subplots()
#im = ax.imshow(scores.std(1),origin='lower',extent=[0,2001,8,16],aspect='auto')
#plt.colorbar(im,ax=ax)
#
#X = data.data.reshape(127,-1)
#clf = make_clf()
#ss = cross_val_score(clf,X,labels,cv=cv)
#for train,test in cv.split(X,labels):
#    clf.fit(X[train],labels[train])
    


