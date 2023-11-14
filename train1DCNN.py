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
import gc

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split, KFold

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision
from tensorflow.keras.backend import clear_session
from datetime import datetime, timedelta
from pyedflib import EdfReader

# private module
from dilationmodel import dilationnet
from read_dataset import GetBatchIndexes, LoadDataset, Interval2Segments_v2, Segments2Data_v2
from dataloader import DataLoader

# use gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# %%

class on_epoch_begin(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def one_epoch_begin(self, epoch, logs=None):
        gc.collect()
        clear_session()


def create_model():
    inputs = Input(shape=(18,256))
    outputs = dilationnet(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class BasicTensorFlowModel(tf.keras.Model):
    def __init__(self):
        super(BasicTensorFlowModel, self).__init__()

        # Define the model architecture
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Pass the inputs through the model layers
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        # Return the model output
        return x

def train_model():
    window_size = 2
    overlap_sliding_size = 1
    normal_sliding_size = window_size
    state = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    train_info_file_path = "./patient_info_chb_train.csv"
    test_info_file_path = "./patient_info_chb_test.csv"
    edf_file_path = "./data/CHB"

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

    train_type_1 = np.array(train_segments_set['preictal_ontime']) # true
    train_type_2 = np.array(train_segments_set['ictal'] + train_segments_set['preictal_early'] + train_segments_set['preictal_late']) # false
    train_type_3 = np.array(train_segments_set['postictal'] + train_segments_set['interictal']) # false

    test_type_1 = np.array(test_segments_set['preictal_ontime']) # true
    test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late']) # false
    test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal']) # false

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    epochs = 100
    batch_size = 128   # 한번의 gradient update시마다 들어가는 데이터의 사이즈

    type_1_kfold_set = kf.split(train_type_1)
    type_2_kfold_set = kf.split(train_type_2)
    type_3_kfold_set = kf.split(train_type_3)
    
    checkpoint_dir = "./training_checkpoint/"    
    # %%
    for n_fold in range(n_splits):
        (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
        (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
        (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
        
        checkpoint_path = os.path.join(checkpoint_dir,f"simple_training_{n_fold}/cp.ckpt")
        next_checkpoint_path = os.path.join(checkpoint_dir,f"simple_training_{n_fold+1}/cp.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # load|create model
        if os.path.exists(next_checkpoint_path): continue        
        elif os.path.exists(checkpoint_path):
            print(f"Load model from {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            print(f"Create new model at {checkpoint_path}")
            model = BasicTensorFlowModel()


        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        train_generator = DataLoader(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes], batch_size)
        model.fit(train_generator, epochs=10)
        
        """
        model.compile(optimizer = 'Adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
                      metrics=[BinaryAccuracy(threshold=.5),
                               Recall(thresholds=0),
                               Precision(thresholds=0),
                               ]
        )
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = (1, 400))
        
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_best_only=True,
                                                        verbose=1)
        
        val_loss_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=0,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False,
            start_from_epoch=0)
        
        train_generator = DataLoader(train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes], batch_size)
        validation_generator = DataLoader(train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes], batch_size)
        history = model.fit(
                    train_generator,
                    epochs = epochs,
                    validation_data = validation_generator,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks= [#tboard_callback,
                                cp_callback,
                                val_loss_callback,
                                on_epoch_begin()
                                ]
                    )
        
        with open(f'./{checkpoint_name}{_}/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        """
        
if __name__=="__main__":
    train_model()
