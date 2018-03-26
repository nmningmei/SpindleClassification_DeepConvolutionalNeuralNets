# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:28:52 2017

@author: ning
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from scipy import stats
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential,Model
from keras.layers import Dense,Lambda,Layer
from keras.layers import Dropout
from keras.layers import Flatten,Reshape
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import losses,metrics
from keras.layers.convolutional import Conv2D,Conv1D,Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D,MaxPooling1D
from keras.layers import UpSampling2D,UpSampling1D
from keras.utils import np_utils
from keras import backend as K
import os
from sklearn.model_selection import train_test_split
import mne

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\CNN vae\\'

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
    
labels = []
for e in os.listdir(working_dir):
    if '-epo.fif' in e:
        temp_epochs = mne.read_epochs(working_dir + e,preload=True)
#        temp_data = temp_epochs.get_data()
#        data.append(temp_data)
        labels.append(temp_epochs.events[:,-1])
        
        del temp_epochs

#data = np.concatenate(data,0)
labels = np.concatenate(labels)
#
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn import metrics
from mne.decoding import Vectorizer,SlidingEstimator,cross_val_multiscore,LinearModel,get_coef
def make_clf(pattern=False,vectorized=False):
    clf = []
    if vectorized:
        clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',MinMaxScaler()))
#    Cs = np.logspace(-3,3,7)
#    estimator = LogisticRegressionCV(Cs,cv=4,scoring='roc_auc',max_iter=int(3e3),random_state=12345,class_weight='balanced')
    estimator = SGDClassifier(max_iter=int(1e3),random_state=12345,class_weight='balanced')
    if pattern:
        estimator = LinearModel(estimator)
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf
cv = StratifiedShuffleSplit(n_splits=10,random_state=12345)
#
#
#
td = SlidingEstimator(make_clf(),scoring='roc_auc',)
#
#scores = cross_val_multiscore(td,data,labels,cv=cv,n_jobs=4)
#
#plt.plot(scores.mean(0))
data = []
#labels = []
for tf in [f for f in os.listdir(working_dir) if ('-tfr.h5' in f)]:
    tfcs = mne.time_frequency.read_tfrs(working_dir+tf)[0]
    data_ = tfcs.data
    
    scaler = MinMaxScaler(feature_range=(0,1))
    vectorizer = Vectorizer()
    data_vec = vectorizer.fit_transform(data_)
    data_scaled = scaler.fit_transform(data_vec)
    data_scaled = vectorizer.inverse_transform(data_scaled)
#    labels_ = tfcs.info['event'].iloc[:,-1].values
    del tfcs
    del data_, data_vec
    data.append(data_scaled)
#    labels.append(labels_)
data = np.concatenate(data,axis=0)


#labels = np.concatenate(labels)

#scores = []
#for freq in range(data.shape[2]):
#    print('start')
#    data_ = data[:,:,freq,:]
#    td = SlidingEstimator(make_clf(),scoring='roc_auc',)
#    scores_ = cross_val_multiscore(td,data_,labels,cv=cv,)#n_jobs=4)
#    print('done')
#    scores.append(scores_)


coefs = []
scores = []
for time_ in range(data.shape[-1]):
    coef = []
    scores_=[]
    for train,test in cv.split(data,labels):
        data_ = data[train,:,:,time_]
        clf = make_clf(True,True)
        clf.fit(data_,labels[train])
        print(time_,metrics.classification_report(clf.predict(data[test,:,:,time_]),labels[test]))
        coef_ = get_coef(clf,'patterns_',True)
        coef.append(coef_)
        temp = metrics.roc_auc_score(clf.predict(data[test,:,:,time_]),labels[test])
        scores_.append(temp)
    coefs.append(np.array(coef))
    scores.append(scores_)
coefs = np.array(coefs)
scores = np.array(scores)  


fig,ax=plt.subplots(figsize=(12,6))  
times = np.linspace(0,3000,192)
ax.plot(times,scores.mean(1),color='red',label='mean decoding scores')
ax.fill_between(times,scores.mean(1)-scores.std(1),scores.mean(1)+scores.std(1),color='red',alpha=0.4)
ax.set(xlabel='Time (ms)',ylabel='AUC ROC',title='Decoding results',xlim=(0,3000),ylim=(0.5,1.))
fig.savefig(saving_dir+'decoding results.png',dpi=300)
temp_epochs = mne.read_epochs(working_dir+'sub5_d2-eventsRelated-epo.fif')
info = temp_epochs.info

coefs = np.swapaxes(coefs,0,-1)
coefs = np.swapaxes(coefs,0,2)
coefs = np.swapaxes(coefs,0,1)

fig,axes = plt.subplots(nrows=4,ncols=int(32/4),figsize=(20,8))
for ii,(ax,title) in enumerate(zip(axes.flatten(),info['ch_names'])):
    im = ax.imshow(coefs.mean(0)[ii,:,:],origin='lower',aspect='auto',extent=[0,3000,6,22],
                   vmin=0,)
    ax.set(title=title)
fig.tight_layout()
fig.savefig(saving_dir+'decoding patterns.png',dpi=300)

























