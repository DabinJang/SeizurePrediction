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

from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision

from datetime import datetime, timedelta
from pyedflib import EdfReader

# private module
from dilationmodel import dilationnet
from read_dataset2 import GetBatchIndexes, LoadDataset, Interval2Segments, Segments2Data

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
        
        self.ratio_type_1 = [5,4,3,2]
        self.ratio_type_2 = [3,3,3,2]
        self.ratio_type_3 = [2,3,4,6]
        self.batch_size = batch_size
        self.epoch = 0
        self.update_period = 20
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data


        self.update_data()

    def on_epoch_end(self):
        self.epoch += 1
        self.update_data()

    def update_data(self):
        # 데이터 밸런스를 위해 데이터 밸런스 조절 및 resampling
        if self.epoch/self.update_period < 4:
            # ratio에 따라 데이터 갯수 정함
            self.type_1_sampled_len = len(self.type_1_data)
            self.type_2_sampled_len = min(int((self.type_1_sampled_len/self.ratio_type_1[int(self.epoch/self.update_period)])*self.ratio_type_2[int(self.epoch/self.update_period)]),len(self.type_2_data))
            self.type_3_sampled_len = int((self.type_1_sampled_len/self.ratio_type_1[int(self.epoch/self.update_period)])*self.ratio_type_3[int(self.epoch/self.update_period)])
            # Sampling mask 생성
            self.type_2_sampling_mask = sorted(np.random.choice(len(self.type_2_data), self.type_2_sampled_len, replace=False))
            self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))

            self.type_2_sampled = self.type_2_data[self.type_2_sampling_mask]
            self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]

            self.batch_num = int((self.type_1_sampled_len + self.type_2_sampled_len + self.type_3_sampled_len)/self.batch_size)
            
            self.type_1_batch_indexes = GetBatchIndexes(self.type_1_sampled_len, self.batch_num)
            self.type_2_batch_indexes = GetBatchIndexes(self.type_2_sampled_len, self.batch_num)
            self.type_3_batch_indexes = GetBatchIndexes(self.type_3_sampled_len, self.batch_num)

    def __len__(self):
        return self.batch_num
    
    def __getitem__(self, idx):
        input_seg = np.concatenate((self.type_1_data[self.type_1_batch_indexes[idx]], 
                                    self.type_2_sampled[self.type_2_batch_indexes[idx]], 
                                    self.type_3_sampled[self.type_3_batch_indexes[idx]]))
        y_batch = np.concatenate( ( np.ones(len(self.type_1_batch_indexes[idx])), 
                                   (np.zeros(len(self.type_2_batch_indexes[idx]))), 
                                   (np.zeros(len(self.type_3_batch_indexes[idx]))) )  )
        
        y_batch = y_batch.tolist()
        y_batch = list(map(int,y_batch))
        y_batch = np.eye(2)[y_batch]
        
        data = Segments2Data(input_seg) # (batch, eeg_channel, data)
        x_batch = np.split(data, 10, axis=-1) # (10, batch, eeg_channel, data)
        x_batch = np.transpose(x_batch,(1,0,2,3))

        if (idx+1) % int(self.batch_num / 5) == 0:
            self.type_3_sampling_mask = sorted(np.random.choice(len(self.type_3_data), self.type_3_sampled_len, replace=False))
            self.type_3_sampled = self.type_3_data[self.type_3_sampling_mask]
            self.type_3_batch_indexes = GetBatchIndexes(self.type_3_sampled_len, self.batch_num)

        return x_batch, y_batch

def train_model():
    window_size = 2
    overlap_sliding_size = 1
    normal_sliding_size = window_size
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    train_info_file_path = "./patient_info_snu_train.csv"
    test_info_file_path = "./patient_info_snu_test.csv"
    edf_file_path = "./data/SNU"

    train_interval_set = LoadDataset(train_info_file_path)
    train_segments_set = {}

    test_interval_set = LoadDataset(test_info_file_path)
    test_segments_set = {}

    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, overlap_sliding_size)
        

    for state in ['postictal', 'interictal']:
        train_segments_set[state] = Interval2Segments(train_interval_set[state],edf_file_path, window_size, normal_sliding_size)
        test_segments_set[state] = Interval2Segments(test_interval_set[state],edf_file_path, window_size, normal_sliding_size)

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
    epochs = 100
    batch_size = 100   # 한번의 gradient update시마다 들어가는 데이터의 사이즈

    type_1_kfold_set = kf.split(train_type_1)
    type_2_kfold_set = kf.split(train_type_2)
    type_3_kfold_set = kf.split(train_type_3)

    checkpoint_name = "Cov1d_ver2_training_"
    # %%
    for _ in range(fold_n):
        (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
        (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
        (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
        
        checkpoint_path = f"{checkpoint_name}{_}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(f"./{checkpoint_name}{_+1}"):
            continue
        else:
            if os.path.exists(f"./{checkpoint_name}{_}"):
                print("Model Loaded!")
                model = tf.keras.models.load_model(checkpoint_path)
            else:
                inputs = Input(shape=(18,512))
                outputs = dilationnet(inputs)
                model = Model(inputs=inputs, outputs=outputs)
        
            model.compile(optimizer = 'Adam',
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
                        metrics=[BinaryAccuracy(threshold=0), Recall(thresholds=0), Precision(thresholds=0)])

            
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        '''
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                        histogram_freq = 1,
                                                        profile_batch = '100,200')
        
        '''
        # Create a callback that saves the model's weights
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_best_only=True,
                                                        verbose=1)
        val_loss_callback = tf.keras.callbacks.EarlyStopping('val_loss', patience=3)
        
        train_generator = Dataloader(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes], batch_size)
        validation_generator = Dataloader(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes], batch_size)
        history = model.fit(
                    train_generator,
                    epochs = epochs,
                    validation_data = validation_generator,
                    use_multiprocessing=True,
                    workers=8,
                    callbacks= [#tboard_callback,
                                checkpoint_callback,
                                val_loss_callback]
                    )
        with open(f'./{checkpoint_name}{_}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

if __name__=="__main__":
    train_model()