# %%
# public module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random
import operator
import pickle

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from datetime import datetime, timedelta
from pyedflib import EdfReader

# private module
from dilationmodel import dilationnet
from read_dataset import GetBatchIndexes, LoadDataset, Interval2Segments_v2, Segments2Data_v2

# to use gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%
class Dataloader(Sequence):
    def __init__(self,type_1_data, type_2_data, type_3_data, batch_size):
        
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data

        self.type_1_data_len = len(type_1_data)
        self.type_2_data_len = len(type_2_data)
        
        type_3_sampled_for_balance = type_3_data[np.random.choice(len(type_3_data), int((self.type_1_data_len + self.type_2_data_len)*1.5),replace=False)]
        self.type_3_data_len = len(type_3_sampled_for_balance)

        self.batch_num = int((self.type_1_data_len + self.type_2_data_len + self.type_3_data_len)/batch_size)

        self.type_1_batch_indexes = GetBatchIndexes(self.type_1_data_len, self.batch_num)
        self.type_2_batch_indexes = GetBatchIndexes(self.type_2_data_len, self.batch_num)
        self.type_3_batch_indexes = GetBatchIndexes(self.type_3_data_len, self.batch_num)

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], self.type_2_data[self.type_2_batch_indexes[idx]], self.type_3_data[self.type_3_batch_indexes[idx]]))

        batch_x, batch_y = Segments2Data_v2(input_seg)

        return batch_x, batch_y

# %%
'''        
    def __init__(self, type_1, type_2, type_3, batch_size, shuffle=True):
        self.type_1 = type_1
        self.type_2 = type_2
        self.type_3 = type_3
        self.batch_size = batch_size
        print(type(self.type_1))
        self.type_3_ratio = 1.5
        self.type_3_used_size = int((len(self.type_1)+len(self.type_2))*self.type_3_ratio)
    
        if self.type_3_used_size<len(self.type_3):
            type_3_index = np.random.choice(len(self.type_3), self.type_3_used_size, replace=False)
            self.type_3_used = [self.type_3[i] for i in type_3_index]
        else:
            self.type_3_used = self.type_3
    
        self.data = np.concatenate((self.type_1, self.type_2, self.type_3_used))
        np.random.shuffle(self.data)
        
    def __len__(self):
        return np.ceil(len(self.data)/self.batch_size)
    
    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))        
        input_seg = self.data[low:high]
        batch_x, batch_y = Segments2Data_v2(input_seg)

        print(f'data gen: {idx}')
        print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y
'''

# %%
window_size = 2
overlap_sliding_size = 1
normal_sliding_size = window_size
state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

# for WSL
train_info_file_path = "./patient_info_chb_train.csv"
test_info_file_path = "./patient_info_chb_test.csv"
edf_file_path = "./data/CHB"

# %%
train_interval_set = LoadDataset(train_info_file_path)
train_segments_set = {}

test_interval_set = LoadDataset(test_info_file_path)
test_segments_set = {}

# 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
    train_segments_set[state] = Interval2Segments_v2(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
    test_segments_set[state] = Interval2Segments_v2(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
    

for state in ['postictal', 'interictal']:
    train_segments_set[state] = Interval2Segments_v2(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)
    test_segments_set[state] = Interval2Segments_v2(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

# type 1은 True Label데이터 preictal_ontime
# type 2는 특별히 갯수 맞춰줘야 하는 데이터
# type 3는 나머지

# AutoEncoder 단계에서는 1:1:3

train_type_1 = np.array(train_segments_set['preictal_ontime']) # true
train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'] + train_segments_set['preictal_late']) # false
train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal']) # false

test_type_1 = np.array(test_segments_set['preictal_ontime']) # true
test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late']) # false
test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal']) # false

fold_n = 5
kf = KFold(n_splits=fold_n, shuffle=True)
epochs = 10
batch_size = 1000   # 한번의 gradient update시마다 들어가는 데이터의 사이즈

type_1_kfold_set = kf.split(train_type_1)
type_2_kfold_set = kf.split(train_type_2)
type_3_kfold_set = kf.split(train_type_3)

# %%
for _ in range(fold_n):
    (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
    (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
    (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
    
    checkpoint_path = f"Cov1d_training_{_}/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(f"./Cov1d_training_{_+1}"):
        continue
    else:
        if os.path.exists(f"./Cov1d_training_{_}"):
            print("Model Loaded!")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            inputs = Input(shape=(18,512))
            outputs = dilationnet(inputs)
            model = Model(inputs=inputs, outputs=outputs)
    
        model.compile(optimizer = 'Adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
                      metrics=['acc'])
        
    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    '''
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = (1, 400))
    '''
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_best_only=True,
                                                    verbose=1)
    
    train_generator = Dataloader(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes], batch_size)
    validation_generator = Dataloader(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes], batch_size)
    history = model.fit(
                train_generator,
                epochs = epochs,
                validation_data = validation_generator,
                #use_multiprocessing=True,
                #workers=6,
                callbacks= [#tboard_callback,
                            cp_callback]
                )
    with open(f'./Cov1d_training_{_}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# %%


# %%



