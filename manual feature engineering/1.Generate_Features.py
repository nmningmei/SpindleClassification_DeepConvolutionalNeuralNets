# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:35:28 2017

@author: ning

This script is to generate features that are chosen manually based on the literature. 
The features were generated according to a given set of epoch length, phase lock value, phase lack index, and coherence values.
"""

#import mne
import numpy as np
import pandas as pd
import os
from time import time
#import networkx as nx
from collections import Counter
from glob import glob
import re
os.chdir('D:\\NING - spindle\\Spindle_by_Graphical_Features')
channelList = ['F3','F4','C3','C4','O1','O2']
import eegPipelineFunctions # customized helper functions

raw_dir = 'D:\\NING - spindle\\training set\\'
# get EEG files that have corresponding annotations
raw_files = []
txt_files = glob(os.path.join(raw_dir,'*annotations.txt'))
for file in txt_files:
    sub = int(re.findall('\d+',file)[0])
    if sub < 11:
        day = re.findall('\d+',file)[1]
        day_for_load = file.split('_')[1][:2]
    else:
        day = file.split('_')[2][-1]
        day_for_load = file.split('_')[2]
    raw_file = [f for f in os.listdir(raw_dir) if ('suj%d'%sub in f) and (day_for_load in f) and ('fif' in f)]
    if len(raw_file) != 0:
        raw_files.append([raw_dir + raw_file[0],raw_dir + file])
        
# directory for storing all the feature files
raw_dir = 'D:\\NING - spindle\\training set\\road_trip\\'
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)
# initialize the range of the parameters we want to compute based on
epoch_lengths  = np.arange(1.,5.,0.2) # 1. to 5 seconds with 0.5 stepsize
plv_thresholds = np.arange(0.6, 0.85, 0.05) # 0.6 to 0.8 with .05
pli_thresholds = np.arange(0.05,0.30, 0.05) # 0.05 to 0.25 with 0.05
cc_thresholds  = np.arange(0.7, 0.95,0.05) # 0.7 to 0.9 with 0.05
# make sub-directories based on epoch length
first_level_directory = []
for epoch_length in epoch_lengths:
    directory_1 = raw_dir + 'epoch_length '+str(epoch_length)+'\\'
    if not os.path.exists(directory_1):
        os.makedirs(directory_1)
        
    first_level_directory.append(directory_1)   
    os.chdir(directory_1)
    #print(os.getcwd())
    for files in raw_files:
        raw_file, annotation_file = files
        temp_anno = annotation_file.split('\\')[-1]
        sub = int(temp_anno.split('_')[0][3:])
        if sub < 11:
            day = temp_anno.split('_')[1][1]
            day_for_load = temp_anno.split('_')[1][:2]
        else:
            day = temp_anno.split('_')[2][-1]
            day_for_load = temp_anno.split('_')[2]
        directory_2 = directory_1 + 'sub' + str(sub) + 'day' + day + '\\'
        if not os.path.exists(directory_2):
            #print(directory_2)
            os.makedirs(directory_2)
        os.chdir(directory_2)
        # epoch the data 
        ssssss = time()
        # the following function `get_data_ready' will segment the data with a given length of hanning window and label the segments according
        # to the manual annotations
        # features will be extracted from these epochds
        # the "my_features_" is the same as decribed in https://osf.io/aqgxe/ paper
        # in principle, the features are basic features like amplitude, frequency peaks, and root-mean-squares
        epochs,label,my_features,_ = eegPipelineFunctions.get_data_ready(raw_file,channelList,
                                                             annotation_file,
                                                             epoch_length=epoch_length)
        print('epoch_length '+str(epoch_length),Counter(label))
        # extract signal features
        print('extracting signal features ......')
        epochFeature = eegPipelineFunctions.featureExtraction(epochs,)
        epochFeature = pd.DataFrame(epochFeature)
        epochFeature['label']=label
        epochFeature.to_csv('sub'+str(sub)+'day'+day+'_'+str(epoch_length)+'_'+'epoch_features.csv',index=False)
        my_features = pd.DataFrame(my_features)
        my_features['label']=label
        my_features.to_csv('sub'+str(sub)+'day'+day+'_'+str(epoch_length)+'_'+'my_features.csv',index=False)
        # compute adjasency matrices based on epochs
        connectivity = eegPipelineFunctions.connectivity(epochs)
        connectivity = np.array(connectivity)
        plv, pli, cc = connectivity[0,:,:,:],connectivity[1,:,:,:],connectivity[2,:,:,:]
        # pre-thresholding graph features
        print('extracting graph features of plv ........')
        plv_pre_threshold = eegPipelineFunctions.extractGraphFeatures(plv)
        plv_pre_threshold['label']=label
        print('extracting graph features of pli ........')
        pli_pre_threshold = eegPipelineFunctions.extractGraphFeatures(pli)
        pli_pre_threshold['label']=label
        print('extracting graph features of cc .........')
        cc_pre_threshold  = eegPipelineFunctions.extractGraphFeatures(cc )
        cc_pre_threshold['label']=label
        plv_pre_threshold.to_csv('sub'+str(sub)+'day'+day+'plv_features.csv',index=False)
        pli_pre_threshold.to_csv('sub'+str(sub)+'day'+day+'pli_features.csv',index=False)
        cc_pre_threshold.to_csv('sub'+str(sub)+'day'+day+'cc_features.csv',index=False)
        eeeeee = time()
        print('done signal, plv, pli, and cc, cost time: %d s'%(eeeeee - ssssss))
#        print('start thresholding')
#        # extract graph features
#        for t_plv,t_pli,t_cc in zip(plv_thresholds,pli_thresholds,cc_thresholds):
#            # convert adjasency matrices to binary adjasency matrices
#            adj_plv = eegPipelineFunctions.thresholding(t_plv,plv)
#            adj_pli = eegPipelineFunctions.thresholding(t_pli,pli)
#            adj_cc  = eegPipelineFunctions.thresholding(t_cc, cc )
#            # this is how we extract graph features
#            graphFeature_plv = eegPipelineFunctions.extractGraphFeatures(adj_plv)
#            graphFeature_pli = eegPipelineFunctions.extractGraphFeatures(adj_pli)
#            graphFeature_cc  = eegPipelineFunctions.extractGraphFeatures(adj_cc )
#            # prepare the sub-directories for storing feature files
#            plv_dir = directory_2 + 'plv_' + str(t_plv) + '\\'
#            pli_dir = directory_2 + 'pli_' + str(t_pli) + '\\'
#            cc_dir  = directory_2 + 'cc_'  + str(t_cc ) + '\\'
#            if not os.path.exists(plv_dir):
#                os.makedirs(plv_dir)
#            if not os.path.exists(pli_dir):
#                os.makedirs(pli_dir)
#            if not os.path.exists(cc_dir):
#                os.makedirs(cc_dir)
#            # saving csvs
#            pd.concat([epochFeature,graphFeature_plv],axis=1).to_csv(plv_dir + 'plv_' + str(t_plv) + '.csv',index=False)
#            pd.concat([epochFeature,graphFeature_pli],axis=1).to_csv(pli_dir + 'pli_' + str(t_pli) + '.csv',index=False)
#            pd.concat([epochFeature,graphFeature_cc ],axis=1).to_csv(cc_dir  + 'cc_'  + str(t_cc ) + '.csv',index=False)


 