import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from dilationmodel import dilationnet, dilation1ch, dilation_layer
#from test import dilationnet, dilation1ch
CHB_FS = 256
SNU_FS = 256
batch = 50
n_signal = 23
window_size = 2

inputs = Input(shape=(n_signal, , 1))
outputs = dilationnet(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.summary()