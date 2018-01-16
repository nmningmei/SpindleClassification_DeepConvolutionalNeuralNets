# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:31:27 2018

@author: ning
"""

import os
import mne
os.chdir('D:/Ning - spindle/')
#import eegPinelineDesign
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
import re

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\'
saving_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\sample images\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
title = {0:'non spindle',1:'spindle'}
for e in os.listdir(working_dir):
    if ('-tfr.h5' in e):
        sub,day,_ = re.findall('\d+',e)
        tfcs = mne.time_frequency.read_tfrs(working_dir+e)
        tfcs = tfcs[0]
        print(e,tfcs.info['event'].values[:,-1].mean(),len(tfcs.info['event'].values[:,-1]),)
        data = tfcs.data
        
        # scale the data to between 0 and 1
        data_s = (data - data.min(0)) / (data.max(0) - data.min(0))
        
        for ii in tqdm(range(data_s.shape[0],desc='within subject%s,day%s'%(sub,day))):
            plt.close('all')
            fig,axes = plt.subplots(figsize=(16,8),nrows=4,ncols=8)
            for jj,ax in enu
        
        
        
        
        
        
        
        
        scaler = MinMaxScaler(feature_range=(0,1))
        vectorizer = Vectorizer()
        data_vec = vectorizer.fit_transform(data)
        data_scaled = scaler.fit_transform(data_vec)
        data_scaled = vectorizer.inverse_transform(data_scaled)
        for ii in tqdm(range(data.shape[0]),desc='within'):
            plt.close('all')
            fig,ax = plt.subplots(figsize=(12,5))
            
            
            im = ax.imshow(data_scaled[ii,:,:,:].mean(0),origin='lower',aspect='auto',extent=[0,3000,6,22],
                           vmin=0,vmax=0.3)
            ax.set(xlabel='time (ms)',ylabel='freqency (Hz)',title=title[tfcs.info['event'].values[:,-1][ii]])
            plt.colorbar(im)
            fig.savefig(saving_dir + '%s_%s_%d.png'%(title[tfcs.info['event'].values[:,-1][ii]],e.split('-')[0],ii))
            plt.close('all')
            
        del tfcs,data,data_vec,data_scaled