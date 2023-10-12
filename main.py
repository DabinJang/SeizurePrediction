import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from dilationmodel import dilationnet, dilation1ch, dilation_layer
#from test import dilationnet, dilation1ch

CHB_FS = 256
SNU_FS = 250 

batch = 50
n_signal = 23
time_sec = 2
data_length = SNU_FS*time_sec

input_shape = (batch, n_signal, data_length, 1)


inputs = Input(shape=(n_signal, data_length, 1))
outputs = dilationnet(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.summary()