#%%
from re import L
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import seaborn as sns
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
import dataload
import envset
#import test_val

# use gpu
try:
    envset.config_gpu()
except:
    pass

window_size = 2
overlap_sliding_size = 1
normal_sliding_size = window_size

OVERLAP_SLIDING_SIZE = 1
NORMAL_SLIDING_SIZE = window_size
DEFAULT_SLIDING_SIZE = window_size

state_list = ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early', 'postictal','interictal']


train_info = {"chb": "./patient_info_chb_train.csv",
              "snu": "./patient_info_snu_train.csv"}
test_info = {"chb": "./patient_info_chb_test.csv",
             "snu": "./patient_info_snu_test.csv"}
edf_dir = {"chb": "./data/CHB",
           "snu": "./data/SNU"}
model_list = {"chb": ["/home/Code/training_checkpoint/training_202311222007/fold_0/cp.ckpt",
                      "/home/Code/training_checkpoint/training_202311222007/fold_1/cp.ckpt",
                      "/home/Code/training_checkpoint/training_202311222007/fold_2/cp.ckpt",
                      "/home/Code/training_checkpoint/training_202311222007/fold_3/cp.ckpt",
                      "/home/Code/training_checkpoint/training_202311222007/fold_4/cp.ckpt"],
              "snu": ["/home/Code/training_checkpoint/training_snu_202311241656/fold_0/cp.ckpt",
                      "/home/Code/training_checkpoint/training_snu_202311241656/fold_1/cp.ckpt",
                      "/home/Code/training_checkpoint/training_snu_202311241656/fold_2/cp.ckpt",
                      "/home/Code/training_checkpoint/training_snu_202311241656/fold_3/cp.ckpt",
                      "/home/Code/training_checkpoint/training_snu_202311241656/fold_4/cp.ckpt"]
              }

target='snu'


test_interval_set = LoadDataset(test_info[target])
test_segments_set = {}

for state in ['preictal_ontime', 'ictal', 'preictal_late', 'preictal_early']:
    test_segments_set[state] = Interval2Segments_v2(test_interval_set[state],edf_dir[target], window_size, overlap_sliding_size)
        
for state in ['postictal', 'interictal']:
    test_segments_set[state] = Interval2Segments_v2(test_interval_set[state],edf_dir[target], window_size, normal_sliding_size)

test_type_1 = np.array(test_segments_set['preictal_ontime']) # true
test_type_2 = np.array(test_segments_set['ictal'] + test_segments_set['preictal_early'] + test_segments_set['preictal_late']) # false
test_type_3 = np.array(test_segments_set['postictal'] + test_segments_set['interictal']) # false

test_generator = DataGenerator([test_type_1,test_type_2,test_type_3], batch_size=256,shuffle=False,save_y=True)

def get_evaluate(model_list):
    try:
        eval = list()
        for model_path in model_list:
            print(model_path)   
            model = tf.keras.models.load_model(model_path)
            evaluations = model.evaluate(test_generator)
            eval.append(evaluations)
        return eval
    except Exception as e:
        print(e)
            

def get_predict(model_list):
    try:
        pred = list()
        for model_path in model_list:            
            model = tf.keras.models.load_model(model_path)
            print(model_path)
            predictions = model.predict(test_generator)
            pred.append(np.argmax(predictions, axis=1))
        return pred
    except Exception as e:
        print(e)


            
def get_confusion_matrix(model_list):
    try:
        cm = list()
        for model_path in model_list:    
            model = tf.keras.models.load_model(model_path)
            print(model_path)
            predictions = model.predict(test_generator)
            print(np.shape(test_generator.y))
            y_predict = np.argmax(predictions, axis=1)
            y_true = np.argmax(test_generator.y, axis=1)
            cm_temp = confusion_matrix(y_true, y_predict)
            print(cm_temp)
            cm.append(cm_temp)
        return cm
    except Exception as e:
        print(e)

#%%


def calculate_confusion_matrix_from_metrics(accuracy, precision, recall):
    # 정확도에 대한 혼동 행렬 비율 계산
    fn_fp = 1 - accuracy
    
    # 정밀도에 대한 혼동 행렬 비율 계산
    fp_over_tp = (1 / precision) - 1
    
    # 리콜에 대한 혼동 행렬 비율 계산
    fn_over_tp = (1 / recall) - 1
    
    # 혼동 행렬의 나머지 부분 계산
    tp = fn_fp/(fp_over_tp+fn_over_tp)
    fn = tp * fn_over_tp
    fp = tp * fp_over_tp
    tn = 1 - (tp + fn + fp)

    return [tp, fp, tn, fn]


#%%  
if __name__=="__main__":
    eval = get_evaluate(model_list[target])
    #get_confusion_matrix(model_list[target])
    cm = list()
    for ev in eval:
        matrics = calculate_confusion_matrix_from_metrics(ev[1], ev[3], ev[2])
        print(matrics)
        cm.append(matrics)
        
    #matrix, tf_matrix, sens, fpr, seg_result = test_val.validation(model_list["snu"][0], test_interval_set, 'snu',5,3)


#%%
m1 = np.array([[cm[0][0],cm[0][3]],
                [cm[0][1],cm[0][2]]])
#%%
sns.heatmap(m1,fmt='.4f',annot=True, cmap='Blues')
plt.xlabel('Predict')
plt.ylabel('True')
# %%
for x in cm:
    print(sum(x))
# %%
history_path = "/home/Code/training_checkpoint/training_snu_202311241656/fold_0/trainHistoryDict"

with open(history_path, 'rb') as file:
    train_history = pickle.load(file)

# %%
snu_train=dict()
for x in ['val_categorical_accuracy', 'val_recall', 'val_precision']:
    snu_train[x] = train_history[x][-1]
# %%

cma = calculate_confusion_matrix_from_metrics(snu_train['val_categorical_accuracy'],snu_train['val_precision'],snu_train['val_recall'] )
# %%
m1 = np.array([[cma[0],cma[3]],
                [cma[1],cma[2]]])
sns.heatmap(m1,fmt='.4f',annot=True, cmap='Blues')
plt.xlabel('Predict')
plt.ylabel('True')

# %%
print(snu_train)
# %%
tp = 16303