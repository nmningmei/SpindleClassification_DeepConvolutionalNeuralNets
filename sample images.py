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
#from sklearn.preprocessing import MinMaxScaler
#from mne.decoding import Vectorizer
import re

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\'
saving_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\sample images\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
title = {9:'non spindle',0:'spindle',1:'spindle',2:'spindle',3:'spindle',4:'spindle'}
count = 0
for e in os.listdir(working_dir)[5:]:
    if ('-tfr.h5' in e):
        sub,day,_ = re.findall('\d+',e)
        tfcs = mne.time_frequency.read_tfrs(working_dir+e)
        tfcs = tfcs[0]
        print(e,tfcs.info['event'].values[:,-2].mean(),len(tfcs.info['event'].values[:,-2]),)
        data = tfcs.data
        ch_names = tfcs.info['ch_names']
        # scale the data to between 0 and 1
        data_s = (data - data.min(0)) / (data.max(0) - data.min(0))
#        if count == 0:
#            break
        for ii in tqdm(range(data_s.shape[0]),desc='within subject%s,day%s'%(sub,day)):
            plt.close('all')
            fig,axes = plt.subplots(figsize=(16,8),nrows=4,ncols=8)
            for jj,(ax,ch_) in enumerate(zip(axes.flatten(),ch_names)):
                im = ax.imshow(data_s[ii,jj,:,:],origin='lower',aspect='auto',extent=[0,3000,6,22],vmin=0,vmax=.5)
                ax.set(xlabel='',ylabel='',title=ch_)
            instance = title[tfcs.info['event'].values[:,-2][ii]]
            jitter_ = np.abs(tfcs.info['event'].iloc[ii][-1] - 0.5)
            title_ = 'sub%s,day%s,marker is %.2f from 0,%s'%(sub,day,jitter_,instance)
            fig.suptitle(title_)
            fig.tight_layout(pad=2.)
            fig.savefig(saving_dir + '%s_%s_%s_%d.png'%(sub,day,instance,ii))
#            plt.show()
            plt.close('all')
        del tfcs,data,data_s
        
        
        
        
        
        
        
       