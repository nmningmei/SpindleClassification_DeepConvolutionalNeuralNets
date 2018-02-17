# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:31:54 2017

@author: ning
"""

import os
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
from sklearn import metrics
import pandas as pd
import pickle
from keras.utils import np_utils
import matplotlib.pyplot as plt
from glob import glob
from random import shuffle
from tqdm import tqdm

# model related
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,Dropout,BatchNormalization
from keras.layers import Flatten,Dense
from keras.models import Model
import keras
from keras.callbacks import ModelCheckpoint


#os.chdir('D:/Ning - spindle/variational_autoencoder_spindles')
#from DataGenerator import DataGenerator

os.chdir('D:/Ning - spindle/training set')

working_dir='D:\\NING - spindle\\DCNN data\\eventRelated_2_14_2018'
saving_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_2_14_2018\\saving'

# directories for saving trained models, uncomment the one I want
#saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\classification 1 (add random inputs)\\'
#saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\shallow\\'
#saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\shallow_mixed\\'
#saving_dir_weight = 'D:\\NING - spindle\\Spindle_by_Graphical_Features\\inverse\\'
saving_dir_weight = 'D:\\NING - spindle\\DCNN data\\eventRelated_2_14_2018\\trained models\\'
# make the directories if not exist
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
if not os.path.exists(saving_dir_weight):
    os.mkdir(saving_dir_weight)

# load validation data
validation_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_2_14_2018\\validation\\'
group = glob(os.path.join(validation_dir,'*.p'))
temp = [pickle.load(open(f,'rb')) for f in group]
X_validation = [a for a,b in temp]
X_validation = np.array(X_validation)
#X_validation_max = X_validation.max(0)
#X_validation_min = X_validation.min(0)
#X_validation = (X_validation - X_validation_min) / (X_validation_max - X_validation_min)

y_validation = [b for a,b in temp]
y_validation = np.array(y_validation)
y_validation = np_utils.to_categorical(y_validation,2)
# training directory
training_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_2_14_2018\\training\\'

#########################################################
############## covn autoencoder model ###################
#########################################################
""" the 5 CNN layer one"""
inputs = Input(shape=(32,16,192),batch_shape=(None,32,16,192),name='input',dtype='float64',)# define input shape and batch input shape
conv1 = Conv2D(256,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(inputs)
print('conv1 shape:',conv1.shape)
drop1 = Dropout(0.5)(conv1)
norm1 = BatchNormalization()(drop1)
print('norm1 shape:',norm1.shape)
#down1 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm1)
#print('down1 shape:', down1.shape)

conv2 = Conv2D(128,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(norm1)
print('conv2 shape:',conv2.shape)
drop2 = Dropout(0.5)(conv2)
norm2 = BatchNormalization()(drop2)
print('norm2 shape:',norm2.shape)
#down2 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm2)
#print('down2 shape:',down2.shape)

conv3 = Conv2D(64,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(norm2)
print('conv3 shape:',conv3.shape)
drop3 = Dropout(0.5)(conv3)
norm3 = BatchNormalization()(drop3)
print('norm3 shape:',norm3.shape)
#down3 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm3)
#print('down3 shape:',down3.shape)

conv4 = Conv2D(32,(4,48),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(norm3)
print('conv4 shape:',conv4.shape)
drop4 = Dropout(0.5)(conv4)
norm4 = BatchNormalization()(drop4)
print('norm4 shape:',norm4.shape)
#down4 = MaxPooling2D((2,4),(1,2),padding='valid',data_format='channels_first',)(norm4)
#print('down4 shape:',down4.shape)

conv5 = Conv2D(16,(4,4),strides=(1,1),activation='relu',padding='valid',data_format='channels_first',
              kernel_initializer='he_normal')(norm4)
print('conv5 shape:',conv5.shape)
drop5 = Dropout(0.5)(conv5)
norm5 = BatchNormalization()(drop5)
print('norm5 shape:',norm5.shape)
#down5 = MaxPooling2D((2,2),(1,1),padding='valid',data_format='channels_first',)(norm5)
#print('down5 shape:',down5.shape)

flat6 = Flatten()(norm5)
drop6 = Dropout(0.5)(flat6)
print('flatten 6 shape:',drop6.shape)

dens7 = Dense(kernel_initializer='he_normal',units=2,activation='softmax')(flat6)
drop7 = Dropout(0.5)(dens7)
print('dense 7 shape:',drop7.shape)

model_auto = Model(inputs = inputs,outputs=dens7)
# put everything together and make a model graph
model_auto.compile(optimizer=keras.optimizers.SGD(),loss=keras.losses.binary_crossentropy,metrics=['categorical_accuracy'])


"""classification"""
breaks = 500 # in each break, we will look at the data and plot the validation results
batch_size = 100 # batch training size, memory reason
through = 5 # how many times we want to go through the training data
#conditions_ = '_random inputs_random_selected'
conditions_ = '_random inputs'
file_path = saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_) # define the path for saving the model
checkPoint = ModelCheckpoint(file_path,monitor='val_loss',save_best_only=True,mode='min',period=1,verbose=1)
callback_list = [checkPoint]
temp_results = [] # temporary results saving list, will convert to pandas dat frame
if os.path.exists(saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_)):# if the model is trained, load the trained model
    model_auto.load_weights(saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_))

#### training and validating ########
for ii in range(breaks):
    labels = []
    all_objects = glob(os.path.join(training_dir,'*.p'))
    for aaa in range(50):
        shuffle(all_objects)
    groups = np.array_split(all_objects,20)
    for jj in range(through):# going through the training data 5 times
#        step_idx = np.random.choice(np.arange(10),size=10,replace=False)
        for step_idx in np.random.choice(len(groups),size=len(groups),replace=False): # going through 15 splitted training data
            group = groups[step_idx]
            temp = [pickle.load(open(f,'rb')) for f in group]
            print('load training instances')
            X_train_ = [a for a,b in temp]
            X_train_ = np.array(X_train_,dtype=np.float32)
            X_train_max = X_train_.max(0)
            X_train_min = X_train_.min(0)
#            print('nomalizing')
#            X_train_ = (X_train_ - X_train_min) / (X_train_max - X_train_min)
            y_train_ = [b for a,b in temp]
            y_train_ = np.array(y_train_,dtype=np.float32)
            y_train_ = np_utils.to_categorical(y_train_,2)
            print('add random inputs')
        	  # add random inputs because the previous model score random inputs as spindles with super high confidence
        	  # however, we never test/validate the model with any random inputs
            random_inputs = np.random.rand(X_train_.shape[0],32,16,192)
            random_labels = [0]*int(X_train_.shape[0]/4)
            random_labels = np_utils.to_categorical(random_labels,2)
            X_train_ = np.concatenate([X_train_,random_inputs],axis=0)
            y_train_ = np.concatenate([y_train_,random_labels],axis=0)
            labels.append(y_train_) # a trick only used in training
            # train each small batch of the data 2 times, the order of the data is shuffled
            model_auto.fit(x=X_train_,y=y_train_,batch_size=batch_size,epochs=2,
                        validation_data=(X_validation,y_validation),shuffle=True,callbacks=callback_list)
            if os.path.exists(saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_)):# if the model is trained, load the trained model
                model_auto.load_weights(saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_))

    labels = np.concatenate(labels,axis=0)
    # load the best state to determine the hyperparameters: stopping point ---- testing data is left untouched
    model_auto.load_weights(saving_dir_weight+'weights.2D_classification%s.best.hdf5'%(conditions_))
    X_predict = model_auto.predict(X_validation)[:,-1] > np.mean(labels[:,-1]) # no at 0.5 level
    X_predict_prob = model_auto.predict(X_validation)[:,-1]
    print(metrics.classification_report(y_validation[:,-1],X_predict))
    # AUC scores, sensitivity, and selectivity (precision and recall)
    AUC = metrics.roc_auc_score(y_validation[:,-1], X_predict_prob)
    fpr,tpr,th = metrics.roc_curve(y_validation[:,-1], X_predict_prob,pos_label=1)
    sensitivity = metrics.precision_score(y_validation[:,-1],X_predict,average='weighted')
    selectivity = metrics.recall_score(y_validation[:,-1],X_predict,average='weighted')
    plt.close('all')
    fig,ax = plt.subplots(figsize=(8,8))
    ax.plot(fpr,tpr,label='AUC = %.3f'%(AUC))
    ax.set(xlabel='false postive rate',ylabel='true positive rate',title='%dth 5 epochs'%(ii+1),
           xlim=(0,1),ylim=(0,1))
    ax.legend(loc='best')
    fig.savefig(saving_dir_weight + 'AUC plot_%d.png'%(ii+1),dpi=400)
    plt.close('all')
#    validation_measure = [cos_similarity(a,b) for a,b in zip(X_validation, X_predict)]
#    print('mean similarity: %.4f +/- %.4f'%(np.mean(validation_measure),np.std(validation_measure)))
    temp_results.append([(ii+1)*50,AUC,sensitivity,selectivity])
    results_for_saving = pd.DataFrame(np.array(temp_results).reshape(-1,4),columns=['epochs','AUC','sensitivity','selectivity'])
    if os.path.exists(saving_dir_weight + 'scores_classification_%s.csv'%conditions_):
        temp_result_for_saving = pd.read_csv(saving_dir_weight + 'scores_classification_%s.csv'%conditions_)
        results_for_saving = pd.concat([temp_result_for_saving,results_for_saving])
    results_for_saving.to_csv(saving_dir_weight + 'scores_classification_%s.csv'%conditions_,index=False)



### testing #####
test_dir = 'D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\test'
group = glob(os.path.join(test_dir,'*.p'))
shuffle(group)
X_predict_prob_ = []
y_test = []
for ii,k in enumerate(np.array_split(group,7)):
    k
    X_test,y_test_ = [],[]
    for f in tqdm(k,desc='split group %d'%ii):
        a,b = pickle.load(open(f,'rb'))
        X_test.append(a)
        y_test_.append(b)
    #temp = [pickle.load(open(f,'rb')) for f in group]
    #X_test = [a for a,b in temp]
    X_test = np.array(X_test)
    X_test_max = X_test.max(0)
    X_test_min = X_test.min(0)
    X_test = (X_test - X_test_min) / (X_test_max - X_test_min)
    #y_test = [b for a,b in temp]
    y_test_ = np.array(y_test_)
    y_test_ = np_utils.to_categorical(y_test_,2)
    y_test.append(y_test_)
    
    #X_predict_ = model_auto.predict(X_test)[:,-1] > 0.5
    print('start a prediction')
    X_predict_prob__ = model_auto.predict(X_test)[:,-1]
    X_predict_prob_.append(X_predict_prob__)
    print('finish a prediction')
X_predict_prob_ = np.concatenate(X_predict_prob_)
y_test = np.concatenate(y_test)

X_predict_ = X_predict_prob_ > 0.5

pickle.dump([X_predict_prob_,X_predict_,y_test],open(saving_dir_weight+'predictions.p','wb'))
X_predict_prob_,X_predict_,y_test = pickle.load(open(saving_dir_weight+'predictions.p','rb'))
print(metrics.classification_report(y_test[:,-1],X_predict_))
AUC = metrics.roc_auc_score(y_test[:,-1], X_predict_prob_)
fpr,tpr,th = metrics.roc_curve(y_test[:,-1], X_predict_prob_,pos_label=1)
sensitivity = metrics.precision_score(y_test[:,-1],X_predict_,average='weighted')
selectivity = metrics.recall_score(y_test[:,-1],X_predict_,average='weighted')
plt.close('all')
fig,ax = plt.subplots(figsize=(8,8))
ax.plot(fpr,tpr,label='AUC = %.3f\nSensitivity = %.3f\nSelectivity = %.3f'%(AUC,sensitivity,selectivity))
ax.set(xlabel='false postive rate',ylabel='true positive rate',title='test data\nlarge to small',
       xlim=(0,1),ylim=(0,1))
ax.legend(loc='best')
fig.savefig(saving_dir_weight + 'test data AUC plot.png',dpi=400)
plt.close('all')

cf =metrics.confusion_matrix(y_test[:,-1],X_predict_)
cf = cf / cf.sum(1)[:, np.newaxis]
import seaborn as sns
plt.close('all')
fig,ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(cf,vmin=0.,vmax=1.,cmap=plt.cm.Blues,annot=False,ax=ax)
coors = np.array([[0,0],[1,0],[0,1],[1,1],])+ 0.5
for ii,(m,coor) in enumerate(zip(cf.flatten(),coors)):
    ax.annotate('%.2f'%(m),xy = coor,size=25,weight='bold',ha='center')
ax.set(xticks=(0.5,1.5),yticks=(0.25,1.25),
        xticklabels=['non spindle','spindle'],
        yticklabels=['non spindle','spindle'])
ax.set_title('Confusion matrix\nDCNN',fontweight='bold',fontsize=20)
ax.set_ylabel('True label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.savefig(saving_dir_weight+'confusion matrix.png',dpi=400)









































