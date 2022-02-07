import os, time, librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils

import audio_data_loader as audioloader
import references as ref


def get_features(y, sr):
    out = []
    
    out.append(np.mean(librosa.feature.rms(y = y)))
    out.append(np.mean(librosa.feature.spectral_centroid(y = y, sr = sr)))
    out.append(np.mean(librosa.feature.spectral_bandwidth(y = y, sr = sr)))
    out.append(np.mean(librosa.feature.spectral_rolloff(y = y, sr = sr))) 
    out.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    for e in librosa.feature.mfcc(y = y, sr = sr):
        out.append(np.min(e))
        out.append(np.mean(e))
        out.append(np.max(e))
        
   
    for e in librosa.feature.chroma_stft(y = y, sr = sr):
        out.append(np.min(e))
        out.append(np.mean(e))
        out.append(np.max(e))

    return out

'''
create dataset fucntion

returns:
x_train
y_train

x_val
y_val

'''
def get_dataset(source):
    
    x_train = []
    y_train = []
    
    for i, item in tqdm(source.iterrows()):
        
        try:
            # read the audio file
            y, sr = audioloader.read_audio(item[0])
            
            # append x_train item
            x_train.append(get_features(y, sr))
            y_train.append(item[8])
            
        except:
            
            print(f'Invalid object {item[0]}')
            
    
    x_train, x_val, y_train, y_val = train_test_split(np.array(x_train), 
                                                      np.array(y_train), 
                                                      test_size = 0.1, 
                                                      random_state = 42, 
                                                      shuffle = True)        
        
    return x_train, x_val, y_train, y_val

'''
This function returns predictors collection
'''
def get_x_data(source):
    
    x_train = []
    
    for i, item in tqdm(source.iterrows()):        
        y, sr = audioloader.read_audio(item[0])        
        x_train.append(get_features(y, sr))
        
    return np.array(x_train)

'''
This function returns gender targets collection
'''
def get_y_gender_data(source):
    
    y_train = []
    
    for i, item in tqdm(source.iterrows()):        
        y_train.append(item[8])
        
    return np.array(y_train)

'''
This function returns emotional intencivity 
targets collection
'''
def get_y_intensivity_data(source):
    
    y_train = []
    
    for i, item in tqdm(source.iterrows()):        
        y_train.append(item[4])
        
    return np.array(y_train)
    
'''
This function returns emotion targets collection
'''
def get_y_emotion_data(source):
    
    y_train = []
    
    for i, item in tqdm(source.iterrows()):
        y_train.append(utils.to_categorical(item[3], len(ref.emotions_ref) + 1))
        
    return np.array(y_train)

'''
This function returns data splitted
'''
def get_data_split(X, GY, EY, split_size = 0.1):
    
    return train_test_split(X, GY, EY, test_size = 0.1, random_state = 42, shuffle = True)    
    