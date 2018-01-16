# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:07:15 2017

@author: ning
"""
import os
os.chdir('D:/Ning - spindle/')
import eegPinelineDesign  # have to change the directory so that I can load these functions, might not use any
from Filter_based_and_thresholding import Filter_based_and_thresholding
import mne
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_multitaper,tfr_morlet
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\DCNN data\\eventRelated_1_15_2018\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
annotations = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
fif_data = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files
def spindle(x,KEY='spindle'):# iterating through each row of a data frame and matcing the string with "KEY"
    keyword = re.compile(KEY,re.IGNORECASE) # make the keyword
    return keyword.search(x) != None # return true if find a match
def get_events(fif,f,channelList = None):

    raw = mne.io.read_raw_fif(fif,preload=True)
    anno = pd.read_csv(f)
    raw.filter(6,22,fir_design='firwin') # make frequency axis 16 dimensions
    
    if channelList == None:
        channelList = ['F3','F4','C3','C4','O1','O2']
    else:
        channelList = raw.ch_names[:32]
    raw.pick_channels(channelList)
    raw.info.normalize_proj()
    spindles = anno[anno.Annotation.apply(spindle)]
    record_jitters = []
    current_time_pairs = []
    jitter_spindle_onset = []
    record_jitters_code = []
    for ii,row in spindles.iterrows():
        for jj,jitter_ in enumerate([-0.5,-0.25,0,0.25,0.5]):
            record_jitters.append(jitter_)
            record_jitters_code.append(jj)
            jitter_onset = row['Onset'] + jitter_
            jitter_spindle_onset.append(jitter_onset)
            start_,stop_ = jitter_onset - 0.5, jitter_onset + 2.5
            current_time_pairs.append([start_,stop_])
    spindle_intervals = np.array(current_time_pairs)
    jitter_spindle_onset = np.array(jitter_spindle_onset)
    nonspindle_time_pairs = []
    nonspindle_onset = []
    """psuedo implementation of coupon collector's problem"""
    for ii, onset in tqdm(enumerate(jitter_spindle_onset),desc='non spindle sampling'):
        if (ii > 0) and (jitter_spindle_onset[ii-1] - onset > 10) and (((onset - 5.5) > (300)) or ((onset + 8.5) < (raw.times[-1]-100))):
            """
            sample spindles and sample non spindle around a spindle
            if there no consective spindles occur within 10 seconds
            """
            nonspindle_time_pairs.append([onset - 8.5, onset - 5.5])
            current_time_pairs.append([onset - 8.5, onset - 5.5])
            nonspindle_onset.append(onset - 8.5)
            record_jitters.append(0)
            record_jitters_code.append(9)
        elif (ii == 0) and (((onset - 5.5) > (300)) or ((onset + 8.5) < (raw.times[-1]-100))):
            nonspindle_time_pairs.append([onset - 8.5, onset - 5.5])
            current_time_pairs.append([onset - 8.5, onset - 5.5])
            nonspindle_onset.append(onset - 8.5)
            record_jitters.append(0)
            record_jitters_code.append(9)
        else:
            #print('start a new non spindle sampling')
            for counter in range(int(1e5)):
                new_sample_ = np.random.choice(raw.times[300000:-100000],size=1)[0]
                new_sample  = np.array( [new_sample_, new_sample_+3])
                if sum([eegPinelineDesign.getOverlap(s_interval,new_sample) for s_interval in current_time_pairs]) != 0:
                    #print('find one',new_sample_,'sample %d times'%counter)
                    current_time_pairs.append(new_sample)
                    nonspindle_onset.append(new_sample_)
                    nonspindle_time_pairs.append([new_sample_, new_sample_+3])
                    record_jitters.append(0)
                    record_jitters_code.append(9)
                    break
            else: # for - else loop!!!!!!
                print('seriously?!, stop at 10,0000 times of samping')
                break
    spindle_events = np.zeros((len(spindle_intervals),3))
    spindle_events[:,0] = spindle_intervals[:,0] + 0.1 # move forward for baselining
    event_id = {'spindle jitter -0.5':0,
                'spindle jitter -0.25':1,
                'spindle jitter 0':2,
                'spindle jitter 0.25':3,
                'spindle jitter 0.5':4,
                'non spindle':9}
    spindle_events[:,0] = spindle_events[:,0] * raw.info['sfreq'] # convert s to ms
    spindle_events = spindle_events.astype(int)
    spindle_events[:,-1] = 1
    
    nonspindle_events = np.zeros((len(nonspindle_time_pairs),3))
    nonspindle_events[:,0] = np.array(nonspindle_time_pairs)[:,0] + 0.1 # move forward for baselining
    nonspindle_events[:,0] = nonspindle_events[:,0] * raw.info['sfreq'] # convert s to ms
    nonspindle_events = nonspindle_events.astype(int)
    nonspindle_events[:,-1] = 0
    
    events = np.concatenate([spindle_events,nonspindle_events],axis=0)
    events[:,-1]=record_jitters_code
    print('\nsampled events:',events.shape)
    # drop duplicates: oversamping
    print('drop duplicates')
    events_ = pd.DataFrame(events,columns=['onset','e','c']) # names of the columns don't matter
    events_['jitter']=record_jitters
    events_ = events_.drop_duplicates('onset')
    events = events_[['onset','e','c']].values.astype(int)
    
    epochs_ = mne.Epochs(raw,events,event_id=event_id,tmin=0,tmax=3.,preload=True,detrend=1,baseline=(None,None))
    print('down sampling')
    epochs_.resample(64)
    print('computing power spectrogram')
    freqs = np.arange(6,22,1)
    n_cycles = freqs / 2.
#    time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
    power = tfr_morlet(epochs_,freqs,n_cycles=n_cycles,return_itc=False,average=False,)
    
    power.info['event'] = events_
    del raw
    return power,epochs_
for f in tqdm(annotations,desc='annotation loop'):
    temp_ = re.findall('\d+',f) # get the numbers in the string
    
    sub = temp_[0] # the first one will always be subject number
    day = temp_[-1]# the last one will always be the day
    
    if int(sub) < 11: # change a little bit for matching between annotation and raw EEG files
        day = 'd%s' % day
    else:
        day = 'day%s' % day
    fif_file = [f for f in fif_data if ('suj%s_'%sub in f.lower()) and (day in f)][0]# the .lower() to make sure the consistence of file name cases
    print(sub,day,f,fif_file) # a checking print 

    power,epochs= get_events(fif_file,f,channelList=32)
    epochs.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day)+ '-epo.fif',)
    power.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day) + '-tfr.h5',overwrite=True)
    del power
#    except:
#        pass
#    
#    try:
#        epochs_spindle = mne.Epochs(raw_fif,events,tmin=0,tmax=2,event_id=event_id,baseline=None,detrend=0,preload=True) 
#        del raw_fif # save memory
#        
#        freqs = np.arange(8,21,1)
#        n_cycles = freqs / 2.
#        time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
#        power = tfr_multitaper(epochs_spindle,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
#        power.info['event'] = events
#        power.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day) + '-tfr.h5',overwrite=True)
#        del power
#    except:
#        del raw_fif
#        print(sub,day,'No matching events found for spindle (event id 1)')
