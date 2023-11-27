# public module
import numpy as np
import pandas as pd
import os
import gc
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.metrics import BinaryAccuracy, Recall, Precision
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pyedflib import EdfReader

#os.environ['TF_GPU_THREAD_MODE']='gpu_private'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def __init__(self, mode=None):
        super().__init__()
        self.mode=mode
        
    def one_epoch_begin(self, epoch, logs=None):
        if self.mode=="one_epoch_begin":
            gc.collect()
            tf.keras.backend.clear_session()


def garbage_collection_callback(mode='one_epoch_begin'):
    return GarbageCollectionCallback(mode)
    

def earlystop_callback(monitor='val_loss'):
    return tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                            min_delta=0,
                                            patience=10,
                                            verbose=0,
                                            mode='auto',
                                            baseline=None,
                                            restore_best_weights=False,
                                            start_from_epoch=0)              
    

def config_gpu():
    os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


