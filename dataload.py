import os
import numpy as np
import pandas as pd
import scipy

import tensorflow as tf
import pyedflib

global state_list
global total_channels
global used_channels

state_list = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']

def get_batch_index(data_len, batch_num):
    batch_size = int(data_len / batch_num)
    idx_list = list(range(data_len))
    np.random.shuffle(idx_list)
    batch_idx_mask = []
    for i in range(batch_num):
        batch_idx_mask.append(idx_list[batch_size*i:batch_size*(i+1)])
    return batch_idx_mask


def load_dataset(filename):
    global state_list
    df = pd.read_csv(filename)
    columns = ['name','start','end','state']
    interval_dict = {}
    for state in state_list:
        condition = df['state'] == state
        df_state = df[condition]
        interval_dict[state] = df_state[columns].values.tolist()
        
    return interval_dict


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


def segment_to_data(segments, channels:list, ):
    state_y = {'preictal_early': 0,
               'preictal_ontime': 1,
               'preictal_late': 0,
               'ictal': 0,
               'postictal': 0,
               'interictal': 0}
    
    num_classes = len(np.unique(list(state_y.values())))
    ONEHOT = True
    
    chb_channel_label_file_dir = './'
    chb_channel_label_file = "chb_channel_label.csv"
    if any([file=="chb_channel_label.csv" for file in os.listdir(chb_channel_label_file_dir)]):
        chb_channel = pd.read_csv(os.path.join(chb_channel_label_file_dir,chb_channel_label_file))
        channels = chb_channel['label'].to_list()
    
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
                target_freq = 128
                
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
                y.append(state_y[state])
    
    x=np.array(x)
    y=np.array(y)
    
    if ONEHOT:
        y = np.array(tf.keras.utils.to_categorical(y, num_classes=num_classes))
        
    return x, y


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data:list, batch_size, proportion={}, shuffle=True, save_y=False):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.save_y = save_y
        self.proportion = proportion if proportion else {0:.5}
        
        self.x = self.balanced_sampling(proportion=self.proportion)
        self.indices = np.arange(len(self.x))
        self.y = list()
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)
    
    def __len__(self):
	    return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        input_seg = [self.x[i] for i in indices]
        batch_x, batch_y = segment_to_data(input_seg, channels = )
        if self.save_y:
            self.y.extend(batch_y)
        #print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y
        
    def on_epoch_end(self):
        self.x = self.balanced_sampling(proportion={0:.5})
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def on_epoch_start(self):
        self.y = list()

    def balanced_sampling(self, proportion={0:.5}):
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

class DatasetHeader:
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

if __name__=="__main__":
    pass

