# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:26:20 2018

@author: ning
"""

import os


working_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\total data\\'
training_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\training\\'
validation_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\validation\\'
testing_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\test\\'

for name in [training_dir,validation_dir,testing_dir]:
    if not os.path.exists(name):
        os.mkdir(name)
        
total_n = len(os.list)