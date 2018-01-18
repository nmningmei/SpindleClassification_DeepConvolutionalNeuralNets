# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:26:20 2018

@author: ning
"""

import os
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm

working_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\total data\\'
training_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\training\\'
validation_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\validation\\'
testing_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\test\\'

for name in [training_dir,validation_dir,testing_dir]:
    if not os.path.exists(name):
        os.mkdir(name)
        
total_files = os.listdir(working_dir)

test_size = int(len(total_files) / 5)
validation_size = int((len(total_files) - test_size) / 10)

"""make test set"""
np.random.seed(12345)
#testing_data = []
testing_data_name = []
testing_labels=[]
sampling_type = 0
for iiii in tqdm(range(int(1e15))):
    if len(testing_data_name) >= test_size:
        break
    else:
        sample = np.random.choice(total_files,size=1,replace=False)[0]
        instance,label = pickle.load(open(working_dir + sample,'rb'))
        if (sample not in testing_data_name) and (label == sampling_type):
#            print(sample,label)
#            testing_data.append(instance)
            testing_labels.append(label)
            testing_data_name.append(sample)
            try:
                Counts_dict = Counter(testing_labels)
#                print(Counts_dict)
                if Counts_dict[0] > Counts_dict[1]:
                    sampling_type = 1
                else:
                    sampling_type = 0
            except:
                print('too early to determine')
                
"""make validation set"""
validation_data_name = []
validation_labels = []
sampling_type = 0
for iiii in tqdm(range(int(1e15))):
    if len(validation_data_name) >= validation_size:
        break
    else:
        sample = np.random.choice(total_files,size=1,replace=False)[0]
        instance,label = pickle.load(open(working_dir + sample,'rb'))
        if (sample not in testing_data_name) and (label == sampling_type) and (sample not in validation_data_name):
            validation_labels.append(label)
            validation_data_name.append(sample)
            try:
                Counts_dict = Counter(validation_labels)
#                print(Counts_dict)
                if Counts_dict[0] > Counts_dict[1]:
                    sampling_type = 1
                else:
                    sampling_type = 0
            except:
                print('too early to determine')

"""check if overlap"""
for a in validation_data_name:
    if a in testing_data_name:
        print(a)
        
"""training set"""
training_data_name = []
for sample in total_files:
    if (sample not in testing_data_name) and (sample not in validation_data_name):
        training_data_name.append(sample)

"""clean the directories"""
directories = [training_dir,testing_dir,validation_dir]
import glob

for directory in directories:
    file_list = glob.glob(directory+"*.p")
    for f in file_list:
        os.remove(f)

"""copy and paste"""
import shutil
cases = [training_data_name,testing_data_name,validation_data_name]
directories = [training_dir,testing_dir,validation_dir]


for case,directory in zip(cases,directories):
    for sample in tqdm(case,desc='%s'%directory):
        shutil.copy2(working_dir+sample,directory)





























         