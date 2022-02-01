import os, time, librosa
import pandas as pd
import numpy as np

from tqdm import tqdm

def create_source_dataset(root_path): 
    
    current_time = time.time()
    
    ds = []
    
    # get absolute root folder path
    root_abspath = os.path.join(os.path.abspath(os.getcwd()), root_path)    
    root_dirs = [os.path.join(root_abspath, x) for x in os.listdir(root_path)] 
    
    for root_dir in root_dirs:
        
        for file_src in os.listdir(root_dir):
            # 0 element
            file_path = os.path.join(root_dir, file_src)
            # 1 modality 0..2
            modality = int(file_src[0:2])
            # 2 vocal channel 3..5
            vocal_channel = int(file_src[3:5])
            # 3 emotion 6..8
            emotion = int(file_src[6:8])
            # 4 emotional intensity 9..11
            emotional_intensity = int(file_src[9:11])
            # 5 statement 12..14
            statement = int(file_src[12:14])
            # 6 repetitions 15..17
            repetions = int(file_src[15:17])
            # 7 actor index 18..20
            actor_id = int(file_src[18:20])
            gender = 0
            if actor_id % 2 == 0:
                gender = 1
                
            ds.append([file_path, modality, vocal_channel, emotion, 
                       emotional_intensity, statement, repetions, actor_id, gender])
            
    assert len(ds) == 1440, 'it must be 1440 files in the source'
    
    df = pd.DataFrame(ds, columns = ['FileSource', 
                                     'Modality', 
                                     'VocalChannel', 
                                     'Emotion', 
                                     'EmotionalIntensity',
                                     'Statement',
                                     'Repetions',
                                     'ActorId',
                                     'Gender'])    
    
    print('Data source array is prepared - {} c'.format(time.time() - current_time))
    
    return df

'''
This function reads audio file and returns:


    y, sr = librosa.load(path, sr = 22050, mono = True, offset = 0.0, 
                 duration = None, dtype = <class 'numpy.float32'>, 
                 res_type = 'kaiser_best')
                 
         sr - target sampling rate. ‘None’ uses the native sampling rate
         
 Returns:
     y    - np.ndarray [shape=(n,) or (2, n)], audio time series
     sr   - number > 0 [scalar]. sampling rate of y 
 
'''
def read_audio(file_name):
    
    assert os.path.isfile(file_name), '{} file is not found'.format(file_name)
    
    return librosa.load(file_name, mono = True) 

'''
This function returns the group of samples array:
0 - audio array 
1 - rate
2 - emotion
3 - full file path
'''

def get_samples(df_src):
    
    samples = []
    
    for i, row in df_src.iterrows():        
        emotion = emotions_ref[row[3] - 1]       
        y, sr = read_audio(row[0])
        
        samples.append([y, sr, emotion, row[0]])
        
    return samples