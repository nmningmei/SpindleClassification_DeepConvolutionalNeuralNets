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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import metrics,utils
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from tqdm import tqdm
#from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#import pandas as pd
#from scipy import stats as stats
#from scipy.stats import randint as sp_randint

os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
# define data working directory and the result saving directory
working_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_12_20_2017\\'
saving_dir = 'D:\\NING - spindle\\SpindleClassification_DeepConvolutionalNeuralNets\\Baseline models\Results\\Non-Linear model\\'
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
font = {
        'weight' : 'bold',
        'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)
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
    
# non-linear temporal decoding
cv = StratifiedShuffleSplit(n_splits=10,random_state=12345)
scores = []
for time_ in tqdm(range(data.shape[-1]),desc='temporal decoding'):
    scores_=[]
    # at each time point, we use the frequency information in each channel as the features
    for train,test in cv.split(data,labels):
        data_ = data[train,:,:,time_]
        clf = make_clf()
        clf.fit(data_,labels[train])
        temp = metrics.roc_auc_score(labels[test],clf.predict_proba(data[test,:,:,time_])[:,-1])
        scores_.append(temp)
    print('\n','%d'%time_,'auc = ',np.mean(scores_),'\n')    
    scores.append(scores_)
scores = np.array(scores,dtype=np.float32)      
pickle.dump(scores,open(saving_dir+'RF scores.p','wb'))
# plot the temporal decoding
fig,ax=plt.subplots(figsize=(12,6))  
times = np.linspace(0,3000,192)
ax.plot(times,scores.mean(1),color='black',alpha=1.,label='Decoding Scores (Mean ROC AUC)')
ax.fill_between(times,scores.mean(1)-scores.std(1)/np.sqrt(10),scores.mean(1)+scores.std(1)/np.sqrt(10),
                color='red',alpha=0.4,label='Decoding Scores (SE ROC AUC)')
ax.axvline(500,linestyle='--',color='black',label='Sleep Spindle Marked Onset')
ax.set(xlabel='Time (ms)',ylabel='AUC ROC',title='Decoding Results\nRandom Forest, 10-fold\nSleep Spindle (N=3372) vs Non-Spindle (N=3368)',
       xlim=(0,3000),ylim=(0.5,1.))
ax.legend()
fig.savefig(saving_dir+'decoding results.png',dpi=400,bbox_inches='tight')

# compare linear and non linear
scores_RF = scores
linear_dir = 'D:\\NING - spindle\\SpindleClassification_DeepConvolutionalNeuralNets\\Baseline models\Results\\Linear model\\'
scores_SVM,_,_ = pickle.load(open(linear_dir+'score_info_coefs.p','rb'))

fig,ax = plt.subplots(figsize=(12,8))
times = np.linspace(0,3000,scores.shape[0])
ax.plot(times,scores_RF.mean(1),color='black',alpha=1.,label='Mean Decoding Scores (RandomForest)')
ax.plot(times,scores_SVM.mean(1),color='black',linestyle='--',alpha=1.,label='Mean Decoding Scores (SVM)')
ax.fill_between(times,
               scores_RF.mean(1)-scores_RF.std(1)/np.sqrt(10),
               scores_RF.mean(1)+scores_RF.std(1)/np.sqrt(10),
               color='red',alpha=.4,label='Decoding Scores (SE)')
ax.fill_between(times,
               scores_SVM.mean(1)-scores_SVM.std(1)/np.sqrt(10),
               scores_SVM.mean(1)+scores_SVM.std(1)/np.sqrt(10),
               color='red',alpha=.4)
ax.axvline(500,linestyle='--',color='blue',label='Sleep Spindle Marked Onset')
ax.legend()
ax.set(xlim=(0,3000),ylim=(0.5,1.),xlabel='Time (ms)',ylabel='AUC ROC',
       title='Decoding Results\nRandom Forest vs Linear SVM, 10-fold\nSleep Spindle (N=3372) vs Non-Spindle (N=3368)')
fig.savefig(saving_dir+'linear vs non linear.png',dpi=600,bbox_inches='tight')

cv = StratifiedShuffleSplit(n_splits=10,random_state=12345)
# non linear whole window classification
AUC,fpr,tpr,sensitivity,selectivity,cm = [],[],[],[],[],[]
for train,test in cv.split(data,labels):
    X = data[train]
    y = labels[train]
    X_ = data[test]
    y_ = labels[test]
    clf = make_clf()
    clf.fit(X,y)
    pred = clf.predict(X_)
    print(metrics.classification_report(y_,pred))
    X_predict_prob_ = clf.predict_proba(X_)[:,-1]
    # metics
    AUC_temp = metrics.roc_auc_score(y_, X_predict_prob_)
    AUC.append(AUC_temp)
    # get the step function of false positive rate as a function of true positive rate
    fpr_temp,tpr_temp,th = metrics.roc_curve(y_, X_predict_prob_,pos_label=1)
    fpr.append(fpr_temp);tpr.append(tpr_temp)
    # average sensitivity and selectivity
    sensitivity_temp = metrics.precision_score(y_,pred,average='weighted')
    selectivity_temp = metrics.recall_score(y_,pred,average='weighted')
    sensitivity.append(sensitivity_temp);selectivity.append(selectivity_temp)
    cm_temp = metrics.confusion_matrix(y_,pred)
    cm_temp = cm_temp.astype('float') / cm_temp.sum(axis=1)[:,np.newaxis]
    cm.append(cm_temp)
import pandas as pd
nonLinear = {'AUC':AUC,'fpr':fpr,'tpr':tpr,'sen':sensitivity,'sel':selectivity,'cm':cm }
nonLinear = pd.DataFrame(nonLinear)
nonLinear['model'] = 'Random Forest'

# linear whole window classification
def make_clf(pattern=False,vectorized=False):
    clf = []
    from sklearn.svm import SVC
    clf.append(('vectorizer',Vectorizer()))
    # use linear SVM as the estimator
    estimator = SVC(max_iter=-1,kernel='linear',random_state=12345,class_weight='balanced',probability=True)
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf
AUC,fpr,tpr,sensitivity,selectivity,cm = [],[],[],[],[],[]
for train,test in cv.split(data,labels):
    X = data[train]
    y = labels[train]
    X_ = data[test]
    y_ = labels[test]
    clf = make_clf()
    clf.fit(X,y)
    pred = clf.predict(X_)
    print(metrics.classification_report(y_,pred))
    X_predict_prob_ = clf.predict_proba(X_)[:,-1]
    # metics
    AUC_temp = metrics.roc_auc_score(y_, X_predict_prob_)
    AUC.append(AUC_temp)
    # get the step function of false positive rate as a function of true positive rate
    fpr_temp,tpr_temp,th = metrics.roc_curve(y_, X_predict_prob_,pos_label=1)
    fpr.append(fpr_temp);tpr.append(tpr_temp)
    # average sensitivity and selectivity
    sensitivity_temp = metrics.precision_score(y_,pred,average='weighted')
    selectivity_temp = metrics.recall_score(y_,pred,average='weighted')
    sensitivity.append(sensitivity_temp);selectivity.append(selectivity_temp)
    cm_temp = metrics.confusion_matrix(y_,pred)
    cm_temp = cm_temp.astype('float') / cm_temp.sum(axis=1)[:,np.newaxis]
    cm.append(cm_temp)
Linear = {'AUC':AUC,'fpr':fpr,'tpr':tpr,'sen':sensitivity,'sel':selectivity,'cm':cm }
Linear = pd.DataFrame(Linear)
Linear['model'] = 'Linear SVM'
df_results = pd.concat([nonLinear,Linear])
df_results.to_csv(saving_dir+'comparison_linear_vs_nonlinear.csv',index=False)