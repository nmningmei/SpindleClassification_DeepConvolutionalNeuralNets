# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:28:52 2017

@author: ning

This script is to do two things, 
    1. converting epochs to power spectrograms
    2. fit and test a linear model to the data
    

"""

import numpy as np
from matplotlib import pyplot as plt
import os
import mne
from glob import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn import metrics
from mne.decoding import Vectorizer,SlidingEstimator,cross_val_multiscore,LinearModel,get_coef
from sklearn import utils
from tqdm import tqdm

def make_clf(pattern=False,vectorized=False):
    clf = []
    if vectorized:
        clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',MinMaxScaler()))
    # use linear SVM as the estimator
    estimator = SVC(max_iter=-1,kernel='linear',random_state=12345,class_weight='balanced',probability=True)
    if pattern:
        estimator = LinearModel(estimator)
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf

os.chdir('D:/Ning - spindle/training set')
# define data working directory and the result saving directory
working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\SpindleClassification_DeepConvolutionalNeuralNets\\Baseline models\Results\\'

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
# we don't get the data and labels simultaneously
# get the labels first because we need to do some other preprocessing before 
# we put all the data together    
labels = []
for e in glob(os.path.join(working_dir,'*-epo.fif')):
    temp_epochs = mne.read_epochs(e,preload=False)
    labels.append(temp_epochs.events[:,-1])
    # save memory
    del temp_epochs
labels = np.concatenate(labels)

# get the data
# sacale the data to (0,1)
data = []
for tf in glob(os.path.join(working_dir,'*-tfr.h5')):
    tfcs = mne.time_frequency.read_tfrs(tf)[0]
    data_ = tfcs.data
    # define a (0,1) scaler
    scaler = MinMaxScaler(feature_range=(0,1))
    # define a vectorizer so we can transform the data from 3D to 2D
    vectorizer = Vectorizer()
    data_vec = vectorizer.fit_transform(data_)
    data_scaled = scaler.fit_transform(data_vec)
    # after we scale the data to (0,1), we transform the data from 2D back to 3D
    data_scaled = vectorizer.inverse_transform(data_scaled)
    del tfcs
    del data_, data_vec
    data.append(data_scaled)
    del data_scaled
data = np.concatenate(data,axis=0)

# shuffle the order of the feature matrix and the labels

for _ in range(10):
    data, labels = utils.shuffle(data,labels)

# customized the temporal decoding process
# define 10-fold cross validation
cv = StratifiedShuffleSplit(n_splits=10,random_state=12345)
coefs = []
scores = []
for time_ in tqdm(range(data.shape[-1]),desc='temporal decoding'):
    coef = []
    scores_=[]
    # at each time point, we use the frequency information in each channel as the features
    for train,test in cv.split(data,labels):
        data_ = data[train,:,:,time_]
        clf = make_clf(True,True)
        clf.fit(data_,labels[train])
#        print(time_,metrics.classification_report(clf.predict(data[test,:,:,time_]),labels[test]))
        # get the patterns decoded by the classifier
        coef_ = get_coef(clf,'patterns_',True)
        coef.append(coef_)
        temp = metrics.roc_auc_score(labels[test],clf.predict_proba(data[test,:,:,time_])[:,-1])
        scores_.append(temp)
    print('\n','%d'%time_,'auc = ',np.mean(scores_),'\n')
    coefs.append(np.array(coef))
    scores.append(scores_)
coefs = np.array(coefs)
scores = np.array(scores)  

# plot the temporal decoding
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
# plot the linear decoding patterns
fig,axes = plt.subplots(nrows=4,ncols=int(32/4),figsize=(20,8))
for ii,(ax,title) in enumerate(zip(axes.flatten(),info['ch_names'])):
    im = ax.imshow(coefs.mean(0)[ii,:,:],origin='lower',aspect='auto',extent=[0,3000,6,22],
                   vmin=0,)
    ax.set(title=title)
fig.tight_layout()
fig.savefig(saving_dir+'decoding patterns.png',dpi=300)

























