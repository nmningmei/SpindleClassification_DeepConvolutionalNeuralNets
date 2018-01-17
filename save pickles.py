# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:00:59 2018

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
import pickle
#from sklearn.preprocessing import MinMaxScaler
#from mne.decoding import Vectorizer
import re

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\'
saving_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\total data\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
title = {9:'non spindle',0:'spindle',1:'spindle',2:'spindle',3:'spindle',4:'spindle'}
count_index = 0
for e in os.listdir(working_dir):
    if ('-tfr.h5' in e):
        sub,day,_ = re.findall('\d+',e)
        tfcs = mne.time_frequency.read_tfrs(working_dir+e)
        tfcs = tfcs[0]
        #print(e,tfcs.info['event'].values[:,-2].mean(),len(tfcs.info['event'].values[:,-2]),)
        data = tfcs.data
        ch_names = tfcs.info['ch_names']
        events = tfcs.info['event']
        labels = np.array(events.c.values != 9).astype(int)
        for instance,label in tqdm(zip(data,labels)):
            for_save = [instance, label]
            pickle.dump(for_save,open(saving_dir+'instance_%d.p'%count_index,'wb'))
            count_index += 1