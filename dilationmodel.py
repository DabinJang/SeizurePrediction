from typing import Union
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model


def dilation_layer(inputs: Input,
                   dilation_rate=[1, 2, 4, 8, 16, 32],
                   filters=2,
                   kernel_size=5):

    def conv1d_with_dilation_rate(dr):
        x = layers.Conv1D(dilation_rate=dr, filters=filters,
                          kernel_size=kernel_size, activation='relu',
                          padding='same', strides=1)
        return x

    dilation_layers = [conv1d_with_dilation_rate(dr) for dr in dilation_rate]

    outputs_dilation_layers = [layer(inputs) for layer in dilation_layers]

    concat_layers = layers.Concatenate(axis=-1)(outputs_dilation_layers)

    return concat_layers


def dilation1ch(inputs):

    x = dilation_layer(inputs,
                       dilation_rate=[1, 2, 4, 8, 16, 32],
                       filters=2,
                       kernel_size=5)
    # Depth-wise 1D
    x = layers.SeparableConv1D(filters=12, kernel_size=5, activation='relu',
                               padding='same', strides=2, data_format='channels_last')(x)
    # Standard 1D
    x = layers.Conv1D(filters=24, kernel_size=5,
                      activation='relu', padding='same', strides=2)(x)
    # Maxpooling
    x = layers.MaxPool1D(3, strides=1, padding='same')(x)
    # Depth-wise 1D
    x = layers.SeparableConv1D(filters=24, kernel_size=5, activation='relu',
                               padding='same', strides=2, data_format='channels_last')(x)
    # Standard 1D
    x = layers.Conv1D(filters=32, kernel_size=5,
                      activation='relu', padding='same', strides=1)(x)
    # Maxpooling
    x = layers.MaxPool1D(3, strides=1, padding='same')(x)
    # Depth-wise 1D
    x = layers.SeparableConv1D(filters=32, kernel_size=5, activation='relu',
                               padding='same', strides=2, data_format='channels_last')(x)
    # Standard 1D
    x = layers.Conv1D(filters=48, kernel_size=5,
                      activation='relu', padding='same', strides=1)(x)
    # Maxpooling
    x = layers.MaxPool1D(3, strides=1, padding='same')(x)
    # average pooling
    x = layers.GlobalAvgPool1D()(x)
    return x


# n channel dilation model
# input_shape = (batch_size, signal_ch, data_size)
def dilationnet(inputs: Input):

    channels = tf.split(inputs, inputs[0].shape[-3], axis=-3)
    channels = [tf.reshape(x, shape=(-1, inputs[0].shape[-2], 1))
                for x in channels]

    after_dilation = list()
    for i in range(len(channels)):
        after_dilation.append(dilation1ch(channels[i]))

    # output
    concat_layers = tf.concat(after_dilation, axis=-1)
    x = layers.Dense(2, activation='softmax')(concat_layers)
    return x
