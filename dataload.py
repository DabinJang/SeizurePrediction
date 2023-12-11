from operator import contains
import os
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pyedflib

global state_list
global total_channels
global used_channels

state_list = ['preictal_early', 'preictal_ontime', 'preictal_late', 'ictal', 'postictal',
              'interictal']


def get_batch_index(data_len, batch_num):
    batch_size = int(data_len / batch_num)
    idx_list = list(range(data_len))
    np.random.shuffle(idx_list)
    batch_idx_mask = []
    for i in range(batch_num):
        batch_idx_mask.append(idx_list[batch_size*i:batch_size*(i+1)])
    return batch_idx_mask


def load_dataset(filename)->pd.DataFrame:
    return pd.read_csv(filename)


def groupby_state(df)->dict:
    columns = ['name','start','end','state']
    interval_dict = {}
    for state in df["state"].unique():
        interval_dict[state] = df[columns].values.tolist()
    return interval_dict


def psm_train_test_split(df:pd.DataFrame, train_size=.8, test_size=.2):
    states = df['state'].unique()

    df_train = pd.DataFrame([], columns=df.columns)
    df_test = pd.DataFrame([], columns=df.columns)

    for state in states:
        this_state = df[df['state']==state]
        n_sample = this_state.index.size
        n_test = max(1, int(n_sample*test_size)) # number of test sample should be more than 1 
        test_index = np.random.choice(range(n_sample),n_test,replace=False)
        train_index = [x for x in range(n_sample) if x not in test_index]
        df_train = pd.concat([df_train, this_state.iloc[train_index, :]], axis=0)
        df_test = pd.concat([df_test, this_state.iloc[test_index, :]], axis=0)

    return df_train, df_test
    
def get_used_state(df:pd.DataFrame, state_label:dict):
    df_selected = list()
    for idx_, row in df.iterrows():
        if state_label[row['state']]!=None:
            df_selected.append(row)
    return pd.DataFrame(df_selected, columns=df.columns)

    
def interval_to_segment(interval_list, edf_dir, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        name, start, end, state = interval[0], int(interval[1]), int(interval[2]), interval[3]
        patient_dir = name.split('_')[0]
        len_segment = int(((end-start-window_size)/sliding_size))+1
        file_path = os.path.join(edf_dir, patient_dir, name+'.edf')
        for i in range(len_segment):
            segments_list.append([file_path, start+sliding_size*i, window_size, state]) 
    return segments_list


def segment_to_data(segments, state_label:dict, channels:list, sampling_frequency:int=128):
    used_state = [x for x in state_label.values() if x!=None]
    num_classes = len(np.unique(used_state))
    ONEHOT = True
    
    chb_channel_label_file_dir = './'
    chb_channel_label_file = "chb_channel_label.csv"
    
    x = [] # return value
    y = [] # return value
    
    # combine segments on same file to read once.
    segments_on_same_file_dict = dict()
    
    for segment in segments:
        file, start, duration, state = segment
        try:
            segments_on_same_file_dict[file].append([start, duration, state])
        except:
            segments_on_same_file_dict[file] = [[start, duration, state]]
            
    for file, segment_list in segments_on_same_file_dict.items():
        if "CHB" in file:
            used_channels = channels
        elif "SNU" in file:
            used_channels = channels
        else: continue
        
        with pyedflib.EdfReader(file) as edf_reader:
            edf_labels = edf_reader.getSignalLabels()
            edf_freq = list(map(int, edf_reader.getSampleFrequencies()))

            if not all([(channel in edf_labels) for channel in used_channels]): continue
                
            for segment_info in segment_list:
                start, duration, state = segment_info
                
                segment_data = []
                scale_factor = 10
                target_freq = sampling_frequency
                
                for channel in used_channels:
                    channel_index = edf_labels.index(channel)
                    channel_freq = edf_freq[channel_index]
                    start_index = int(start)*channel_freq
                    duration_length = int(duration)*channel_freq

                    signal = edf_reader.readSignal(channel_index,start_index, duration_length)
                    
                    resample_len = int(len(signal)*target_freq/channel_freq)
                    if channel_freq!=target_freq:
                        signal = scipy.signal.resample(signal, resample_len)
                        
                    scaled_signal = signal/scale_factor
                    segment_data.append(scaled_signal)

                x.append(segment_data)
                y.append(state_label[state])
    
    x=np.array(x)
    y=np.array(y)
    
    if ONEHOT:
        y = np.array(tf.keras.utils.to_categorical(y, num_classes=num_classes))
        
    return x, y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data:list, state_label, channels, batch_size, proportion, balance_data=True, shuffle=True, save_y=False):
        self.data = data
        self.state_label = state_label
        self.channels = channels
        self.batch_size = batch_size
        self.proportion = proportion
        self.balance_data = balance_data
        self.shuffle = shuffle
        self.save_y = save_y
        
        if self.balance_data:
            self.x = self.balanced_sampling(proportion=self.proportion)
        else:
            self.x = []
            self.x.extend(self.data)
        
        self.indices = np.arange(len(self.x))
        self.y = list()
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __len__(self):
	    return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        input_seg = [self.x[i] for i in indices]
        batch_x, batch_y = segment_to_data(input_seg, self.state_label, channels=self.channels, sampling_frequency=128)
        if self.save_y:
            self.y.extend(batch_y)
        #print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y
        
    def on_epoch_end(self):
        self.x = self.balanced_sampling(proportion=self.proportion)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def on_epoch_start(self):
        self.y = list()

    def balanced_sampling(self, proportion):
        data_list = self.data
        data_list_len = len(data_list)
        try:
            for prop_key in proportion.keys():                          
                if prop_key not in range(data_list_len):
                    raise IndexError
                if (proportion[prop_key]<0) or (1<proportion[prop_key]): 
                    raise ProportionValueError
            
            if 1<sum(proportion.values()):
                raise ProportionSumError
                
        except IndexError:
            print("proportion key not in data_list index")
            return []
        
        except ProportionValueError:
            print("proportion value should be in range(0, 1)")
            return []
        
        except ProportionSumError:
            print("Sum of proportion value should be in range(0, 1)")
            return []
            
        data_len = [len(data) for data in data_list]  
        total_data = sum(data_len)
        expect_proportion = dict()

        if not proportion.keys():
            for idx in range(data_list_len):
                expect_proportion[idx] = 1/data_list_len
        
        if proportion.keys():
            sum_rest_proportion = 1 - sum(proportion.values())
            n_rest_proportion = data_list_len - len(proportion.keys())
            for idx in range(data_list_len):
                if idx in proportion.keys():
                    expect_proportion[idx] = proportion[idx]
                else:
                    expect_proportion[idx] = sum_rest_proportion/n_rest_proportion

        needed_max_sample_data=dict()
        for idx in range(data_list_len):
            needed_max_sample_data[idx] = data_len[idx]/expect_proportion[idx]
        
        max_sample_data = min(needed_max_sample_data.values())
        
        expect_sample_len=dict()
        for idx in range(data_list_len):
            expect_sample_len[idx] = int(max_sample_data*expect_proportion[idx])
        
        balanced_sampling_data = list()
        for idx in range(data_list_len):
            sample_index = np.random.choice(data_len[idx], expect_sample_len[idx], replace=False)
            balanced_sampling_data.extend(data_list[idx][sample_index])
            
        return balanced_sampling_data     


class ProportionValueError(Exception):
    pass     

class ProportionSumError(Exception):
    pass

class AboutDatasetHeader:
    def __init__(self):
        self.datasets = ['snu', 'chb']
        
        self.patient_info = {'snu': "./patient_info_snu.csv",
                             'chb': "./patient_info_chb.csv",
                             }
        self.train_info = {'snu': "./patient_info_snu_train.csv",
                           'chb': "./patient_info_chb_train.csv",
                           }
        
        self.test_info = {'snu': "./patient_info_snu_test.csv",
                          'chb': "./patient_info_chb_train.csv",
                          }
        
        self.edf_dir = {'snu': "./data/SNU",
                        'chb': "./data/CHB",
                        }
        
    def set_patient_info(self, dataset:str, path:str):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
        self.patient_info[dataset] = path
    
    def set_train_info(self, dataset:str, path:str):
        if dataset not in self.datasets:
            self.datasets.append(dataset)        
        self.train_info[dataset] = path
    
    def set_test_info(self, dataset:str, path:str):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
        self.test_info[dataset] = path
    
    def set_edf_dir(self, dataset:str, path:str):
        if dataset not in self.datasets:
            self.datasets.append(dataset)        
        self.edf_dir[dataset] = path
        
    def get_info(self, dataset):
        pass

class DatasetHeader:
    def __init__(self,
                 name:str,
                 dir:str,
                 patient_info:str,
                 train_info:str='',
                 test_info:str='',
                 channels:dict=dict(),
                 ):
        self.__dataset_channels = {
            "snu": [
                'Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG',
                'F4-AVG', 'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG',
                'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG',
                'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG',
                'Pz-AVG'
                ],
            "chb": [
                'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                'P8-O2', 'FZ-CZ', 'CZ-PZ'
                ],
            }
        
        self.name = name
        self.dir = dir
        self.patient_info = patient_info
        self.train_info = train_info
        self.test_info = test_info
        self.channels = channels
        self.used_channels = channels
        
        if (not channels):
            for dataset_name in list(self.__dataset_channels.keys()):
                if dataset_name in name:
                    self.channels = self.__dataset_channels[dataset_name]
                    self.used_channels = self.__dataset_channels[dataset_name]
                    break

    def set_used_channels(self, used_channels):
        self.used_channels = used_channels