# %%
# public module
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import CategoricalAccuracy, Recall, Precision
from tensorflow.keras.backend import clear_session
from datetime import datetime, timedelta
from pytz import timezone
import matplotlib.pyplot as plt
# private module

from dilationmodel import dilationnet
from read_dataset import GetBatchIndexes, LoadDataset, Interval2Segments_v2, Segments2Data_v2
from dataload import DataGenerator
import envset

train_dataset = 'snu'

# get server datetime to make checkpoint
def get_current_datetime(format='%Y%m%d%H%M'):
    return datetime.now().astimezone(timezone('Asia/Seoul')).strftime(format)

# %%
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    outputs = dilationnet(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model():
    window_size = 2
    overlap_sliding_size = 1
    normal_sliding_size = window_size

    OVERLAP_SLIDING_SIZE = 1
    NORMAL_SLIDING_SIZE = window_size
    DEFAULT_SLIDING_SIZE = window_size
    
    state_list = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']

    train_info_file_path = "./patient_info_snu_train.csv"
    test_info_file_path = "./patient_info_snu_test.csv"
    edf_file_path = "./data/SNU"

    target_patient = None


    train_interval_set = LoadDataset(train_info_file_path)
    train_segments_set = {}

    test_interval_set = LoadDataset(test_info_file_path)
    test_segments_set = {}
    
    # 상대적으로 데이터 갯수가 적은 것들은 window_size 2초에 sliding_size 1초로 overlap 시켜 데이터 증강
    for state in state_list:
        if state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
            sliding_size = OVERLAP_SLIDING_SIZE
        elif state in ['postictal', 'interictal']:
            sliding_size = NORMAL_SLIDING_SIZE
        else:
            sliding_size = DEFAULT_SLIDING_SIZE
            print(f"MatchError: [{state}] is not an expected value")
            
        train_segments_set[state] = Interval2Segments_v2(train_interval_set[state],edf_file_path, window_size, sliding_size)
        test_segments_set[state] = Interval2Segments_v2(test_interval_set[state],edf_file_path, window_size, sliding_size)
        
        # sellect patient
        if target_patient:
            train_segments_set[state] = train_segments_set[state][train_segments_set[state]['name']==target_patient]
            test_segments_set[state]  = test_segments_set[state][train_segments_set[state]['name']==target_patient]

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
    now = get_current_datetime()
    checkpoint_now_dir = os.path.join(checkpoint_dir, f"training_{train_dataset}_{now}")
    
    # %%
    for n_fold in range(n_splits):
        (type_1_train_indexes, type_1_val_indexes) = next(type_1_kfold_set)
        (type_2_train_indexes, type_2_val_indexes) = next(type_2_kfold_set)
        (type_3_train_indexes, type_3_val_indexes) = next(type_3_kfold_set)
        
        checkpoint_fold_dir = os.path.join(checkpoint_now_dir,f"fold_{n_fold}")
        checkpoint_next_fold_dir = os.path.join(checkpoint_now_dir,f"fold_{n_fold+1}")        
        checkpoint_path = os.path.join(checkpoint_fold_dir,"cp.ckpt")
        next_checkpoint_path = os.path.join(checkpoint_next_fold_dir,"cp.ckpt")
        
        
        # load|create model
        if os.path.exists(next_checkpoint_path): continue        
        elif os.path.exists(checkpoint_path):
            print(f"Load model from {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            print(f"Create new model at {checkpoint_path}")
            model = create_model((21,256))

        print(checkpoint_path)
        
        #model.summary()
        
        model.compile(optimizer = 'Adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[CategoricalAccuracy(),
                               Recall(class_id=0),
                               Precision(class_id=0),
                               ]
        )
        logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        """
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                    histogram_freq = 1,
                                                    profile_batch = (1, 400))
        
        """
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_best_only=True,
                                                        verbose=1)
        
        val_loss_callback = envset.earlystop_callback(monitor="val_loss")
        
        train_generator = DataGenerator([train_type_1[type_1_train_indexes], train_type_2[type_2_train_indexes], train_type_3[type_3_train_indexes]], batch_size=batch_size)
        validation_generator = DataGenerator([train_type_1[type_1_val_indexes], train_type_2[type_2_val_indexes], train_type_3[type_3_val_indexes]], batch_size=batch_size)
        history = model.fit(
                    train_generator,
                    epochs = epochs,
                    validation_data = validation_generator,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks= [cp_callback,
                                val_loss_callback,
                                #envset.garbage_collection_callback()
                                ]
                    )
        print(f'\n\n\ncurrent fold{n_fold} done.\n\n\n')
        try:            
            with open(f"{checkpoint_fold_dir}/trainHistoryDict", 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
                print(f"save trainHistoryDict at {checkpoint_fold_dir}")
        except:
            print('SaveError: Cannot save history')
    #test_generator = DataGenerator([test_type_1,test_type_2,test_type_3],batch_size=batch_size)
    #model.predict(test_generator)
        
if __name__=="__main__":
    try:
        envset.config_gpu()
        train_model()
        
    except Exception as e:
        print(e)
        