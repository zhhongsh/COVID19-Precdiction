import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Permute, Multiply, Lambda, RepeatVector,Input
from keras.layers.core import Flatten,Reshape
from keras.layers import LSTM, Activation, Dropout,Bidirectional,Embedding,MaxPooling1D
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.utils import plot_model

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import keras.backend as K
import datetime
import pandas as pd
import numpy as np
import math

import scipy.integrate as spi
import time

from Keras_IndRNN.ind_rnn import IndRNN
from Keras_IndRNN.ind_rnn import IndRNNCell, RNN

import random


#plt.rcParams['font.sans-serif'] = ['SimHei'] #Specify the default font
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.family'] = 'sans-serif'

look_back = 45#23
predict_time = 32

TIME_STEPS = 1
INPUT_DIM = 45

SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = True

test_noadjust_time = 0
adjust_time = 0


def mape(true,pred):
    return np.mean(np.abs((pred-true)/true))*100

##########  Sequence data generation ##########
def create_dataset(dataset, look_back=1):
    #look_back: the size of the sliding window, i.e., the length of the sequence data
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):   
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

'''
##########  Attention mechanism module ##########
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    #input_dim = int(inputs.shape[2])
    input_dim = INPUT_DIM
    a = Permute((2, 1))(inputs)
    #a = np.Reshape((input_dim, TIME_STEPS,))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

##########  Attention + LSTM ##########
def build_before_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    attention_mul = attention_3d_block(inputs)
    lstm_out1 = LSTM(25, return_sequences=True)(attention_mul)
    lstm_out2 = LSTM(50, return_sequences=True)(lstm_out1)
    lstm_out3 = LSTM(50, return_sequences=False)(lstm_out2)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))
    dense_out = Dense(1)(lstm_out3)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

##########  LSTM + Attention ##########
def build_after_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm_out1 = LSTM(25, return_sequences=True)(inputs)
    lstm_out2 = LSTM(50, return_sequences=True)(lstm_out1)
    lstm_out3 = LSTM(50, return_sequences=True)(lstm_out2)
    attention_mul = attention_3d_block(lstm_out3)
    attention_mul = Flatten()(attention_mul)
    
    dense_out = Dense(1,activation = 'sigmoid')(attention_mul)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

##########  3*LSTM network structure ##########
def build_model(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))

    #LSTM lyars:Number of output nodes are 25
    model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    #model.add(Dropout(0.6))
    model.add(Dense(1))#Output dimension:1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    #plot_model(model,to_file = '../Model_3lstm.jpg',show_shapes = True)
    model.summary()
    return model
'''

##########  Fine-tuning of 3*LSTM model ##########
def adjust_model(look_back):
    model = Sequential()
    '''
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm_out1 = LSTM(25, return_sequences=True)(inputs)
    lstm_out2 = LSTM(50, return_sequences=True)(lstm_out1)
    lstm_out3 = LSTM(50, return_sequences=True)(lstm_out2)
    dense_out = Dense(1,activation = 'sigmoid')(lstm_out3)
    output = Activation('linear')(dense_out)

    model = Model(input=[inputs], output=output)
    '''
    model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    #model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights

    #x = base_model.output
    #x = Dense(1,activation = 'sigmoid')(x)
    #output = Activation('linear')(x)
    #model = Model(input=[inputs], output=output)

    for layer in model.layers[:3]:     #Freeze the parameters before the dense layer
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
   

    #plot_model(model,to_file = '../Model_3lstm.jpg',show_shapes = True)
    model.summary()
    return model

##########  LSTM model ##########
def LSTM_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm= LSTM(128,return_sequences=True)(inputs)
    lstm=Flatten()(lstm)
    dense_out = Dense(1)(lstm)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  Fine-tuning of LSTM model ##########
def adjust_LSTM_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm= LSTM(128,return_sequences=True)(inputs)
    lstm=Flatten()(lstm)
    dense_out = Dense(1)(lstm)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights

    for layer in model.layers[:2]:     #Freeze the parameters before the dense layer
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########   GRU model ##########
def GRU_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    gru= GRU(128,return_sequences=True)(inputs)
    gru=Flatten()(gru)
    dense_out = Dense(1)(gru)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  Fine-tuning of GRU model ##########
def adjust_GRU_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    gru= GRU(128,return_sequences=True)(inputs)
    gru=Flatten()(gru)
    dense_out = Dense(1)(gru)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights

    for layer in model.layers[:2]:     #Freeze the parameters before the dense layer
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  Bi-LSTM model ##########
def Bi_LSTM(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(128, input_shape=(1, look_back),return_sequences=True)))
    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    #model.add(Bidirectional(LSTM(32, return_sequences=True)))
    #model.add(Bidirectional(LSTM(16, return_sequences=True)))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))##Add a fully connected layer, the output dimension is 1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  Fine-tuning of LSTM model ##########
def adjust_Bi_LSTM_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(32,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(16,return_sequences=True))(bi_lstm)
    bi_lstm=Flatten()(bi_lstm)
    dense_out = Dense(1)(bi_lstm)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)

    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights

    for layer in model.layers[:3]:     ##Freeze the parameters before the dense layer
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model
'''
##########  Bi_LSTM + LSTM model ##########
def Bi_LSTM_LSTM(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(25, input_shape=(1, look_back),return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(LSTM(50, return_sequences=True))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  2*Bi_LSTM model ##########
def Bi_LSTM_2(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(25, input_shape=(1, look_back),return_sequences=True)))
    model.add(Bidirectional(LSTM(50,return_sequences=True)))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model


##########  attention + Bi_LSTM model ##########
def Bi_LSTM_before_attention(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    bi_lstm  = Bidirectional(LSTM(25,return_sequences=True))(attention_mul)
    fla = Flatten()(bi_lstm)
    dense_out = Dense(1)(fla)
    output = Activation('linear')(dense_out)

    model = Model(input=[inputs], output=output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))

    model.summary()
    return model
'''

##########  IndRNN model ##########
def build_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    indrnn = IndRNN(128)(inputs)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))
    dense_out = Dense(1)(indrnn)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

##########  Fine-tuning of IndRNN model ##########
def adjust_build_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    indrnn = IndRNN(128)(inputs)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))
    dense_out = Dense(1)(indrnn)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights

    for layer in model.layers[:2]:     #Freeze the parameters before the dense layer
        layer.trainable = False  

    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))

    model.summary()
    return model

'''
##########  Bi_LSTM + IndRNN model ##########
def Bi_LSTM_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(128)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))
    dense_out = Dense(64)(indrnn)
    dense_out = Dense(32)(dense_out)
    dense_out = Dense(16)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    dense_out = Dense(1)(dense_out)

    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

##########  Fine-tuning of Bi_LSTM+IndRNN model ##########
def adjust_Bi_LSTM_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(128)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))
    dense_out = Dense(64)(indrnn)
    dense_out = Dense(32)(dense_out)
    dense_out = Dense(16)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    dense_out = Dense(1)(dense_out)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)

    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    for layer in model.layers[:2]:     #冻结dense层之前的参数
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.build((None,1, look_back))
    model.summary()
    return model

##########  4*Bi_LSTM+IndRNN model ##########
def Bi_LSTM4_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(bi_lstm)
    bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(32)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(16)(indrnn)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    dense_out = Dense(1)(dense_out)

    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model

##########  Fine-tuning of 4*Bi_LSTM+IndRNN model ##########
def adjust_Bi_LSTM4_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(bi_lstm)
    bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(32)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(16)(indrnn)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    dense_out = Dense(1)(dense_out)

    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重
    for layer in model.layers[:3]:     #冻结dense层之前的参数
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model
'''

##########   Stacked_Bi_GRU model ##########
def Stacked_Bi_GRU(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    bi_gru= Bidirectional(GRU(256,return_sequences=True,activation='sigmoid'))(inputs)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(128,return_sequences=True,activation='sigmoid'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(64,return_sequences=True,activation='sigmoid'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(32,return_sequences=True,activation='tanh'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    fla = Flatten()(bi_gru)
    dense_out = Dense(8)(fla)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########   Fine-tuning of Stacked_Bi_GRU model ##########
def adjust_Stacked_Bi_GRU(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    bi_gru= Bidirectional(GRU(256,return_sequences=True,activation='sigmoid'))(inputs)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(128,return_sequences=True,activation='sigmoid'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(64,return_sequences=True,activation='sigmoid'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    bi_gru= Bidirectional(GRU(32,return_sequences=True,activation='tanh'))(bi_gru)
    bi_gru = Dropout(0.2)(bi_gru)
    fla = Flatten()(bi_gru)
    dense_out = Dense(8)(fla)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  ##Loading of model weights
    for layer in model.layers[:10]:     ##Freeze the parameters before the dense layer
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########   CNN_LSTM model ##########
def CNN_LSTM(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(256,3,border_mode='same')(inputs)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)

    lstm= LSTM(128,return_sequences=True)(conv1d)
    lstm = Dropout(0.2)(lstm)
    lstm= LSTM(64,return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    lstm= LSTM(32,return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)

    fla = Flatten()(lstm)

    dense_out = Dense(16)(fla)
    dense_out = Dropout(0.2)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dropout(0.2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Fine-tuning of CNN_LSTM model ##########
def adjust_CNN_LSTM(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(256,3,border_mode='same')(inputs)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)

    lstm= LSTM(128,return_sequences=True)(conv1d)
    lstm = Dropout(0.2)(lstm)
    lstm= LSTM(64,return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)
    lstm= LSTM(32,return_sequences=True)(lstm)
    lstm = Dropout(0.2)(lstm)

    fla = Flatten()(lstm)

    dense_out = Dense(16)(fla)
    dense_out = Dropout(0.2)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dropout(0.2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #Loading of model weights
    for layer in model.layers[:14]:     ##Freeze the parameters before the dense layer
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########   deep_CNN model ##########
def deep_CNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(16,3,border_mode='same')(inputs)
    conv1d = Convolution1D(32,3,border_mode='same')(conv1d)
    conv1d = Convolution1D(32,3,border_mode='same')(conv1d)
    conv1d = Convolution1D(64,3,border_mode='same')(conv1d)

    fla = Flatten()(conv1d)
    drop_out = Dropout(0.2)(fla)

    output = Dense(1,activation='sigmoid')(drop_out)
    model = Model(input=[inputs], output=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Fine-tuning of deep_CNN model ##########
def adjust_deep_CNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(16,3,border_mode='same')(inputs)
    conv1d = Convolution1D(32,3,border_mode='same')(conv1d)
    conv1d = Convolution1D(32,3,border_mode='same')(conv1d)
    conv1d = Convolution1D(64,3,border_mode='same')(conv1d)
    
    fla = Flatten()(conv1d)
    drop_out = Dropout(0.2)(fla)

    output = Dense(1,activation='sigmoid')(drop_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  ##Loading of model weights
    for layer in model.layers[:7]:     ##Freeze the parameters before the dense layer
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  CNN_IndRNN model ##########
def CNN_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(256,3,border_mode='same')(inputs)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)

    indrnn= IndRNN(128)(conv1d)
    dense_out = Dense(64)(indrnn)
    dense_out = Dense(32)(indrnn)
    dense_out = Dense(16)(indrnn)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(indrnn)
    dense_out = Dense(2)(indrnn)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Fine-tuning of CNN_IndRNN model ##########
def adjust_CNN_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    conv1d = Convolution1D(256,3,border_mode='same')(inputs)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)
    conv1d = Convolution1D(256,3,border_mode='same')(conv1d)
    conv1d = Dropout(0.2)(conv1d)

    indrnn= IndRNN(128)(conv1d)
  
    dense_out = Dense(64)(indrnn)
    dense_out = Dense(32)(indrnn)
    dense_out = Dense(16)(indrnn)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(indrnn)
    dense_out = Dense(2)(indrnn)
    output = Dense(1,activation='sigmoid')(dense_out)

    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  ##Loading of model weights
    for layer in model.layers[:22]:     ##Freeze the parameters before the dense layer
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Stacked_IndRNN model ##########
def Stacked_IndRNN(look_back):

    cells = [IndRNNCell(256)]

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    indrnn = RNN(cells)(inputs)
    dense_out = Dense(128)(indrnn)
    dense_out = Dense(64)(dense_out)
    dense_out = Dense(32)(dense_out)
    dense_out = Dense(16)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Fine-tuning of Stacked_IndRNN model ##########
def adjust_Stacked_IndRNN(look_back):

    cells = [IndRNNCell(256)]

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    indrnn = RNN(cells)(inputs)
    dense_out = Dense(128)(indrnn)
    dense_out = Dense(64)(dense_out)
    dense_out = Dense(32)(dense_out)
    dense_out = Dense(16)(dense_out)
    dense_out = Dense(8)(dense_out)
    dense_out = Dense(4)(dense_out)
    dense_out = Dense(2)(dense_out)
    output = Dense(1,activation='sigmoid')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  ##Loading of model weights
    for layer in model.layers[:2]:     ##Freeze the parameters before the dense layer
        layer.trainable = False
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

########## CIRNN model ##########
def CIRNN(look_back):
    cells = [IndRNNCell(256),IndRNNCell(256)]

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #embed = Embedding(2,2)(inputs)
    conv1d = Convolution1D(256,3,border_mode='same',activation='relu')(inputs)
    pool = MaxPooling1D(pool_size = 2,strides = 1,border_mode='same')(conv1d)
    pool = Dropout(0.2)(pool)
    indrnn = RNN(cells)(pool)
    output = Dense(1,activation='sigmoid')(indrnn)
    model = Model(input=[inputs], output=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########  Fine-tuning of CIRNN model ##########
def adjust_CIRNN(look_back):
    cells = [IndRNNCell(256),IndRNNCell(256)]

    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #embed = Embedding(2,2)(inputs)
    conv1d = Convolution1D(256,3,border_mode='same',activation='relu')(inputs)
    pool = MaxPooling1D(pool_size = 2,strides = 1,border_mode='same')(conv1d)
    pool = Dropout(0.2)(pool)
    indrnn = RNN(cells)(pool)
    output = Dense(1,activation='sigmoid')(indrnn)
    model = Model(input=[inputs], output=output)
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

'''
############ Model training and prediction ###########
def train_model(trainX, trainY, testX,testY,train, predict_time):
    ###########Model training###########
    #model = build_model(look_back)
    #model = adjust_model(look_back)
    #model = build_before_model(look_back)
    #model = build_after_model(look_back)
    #model = Bi_LSTM(look_back)
    #model = Bi_LSTM_before_attention(look_back)
    #model = Bi_LSTM_LSTM(look_back)
    #model = Bi_LSTM_2(look_back)
    model = build_IndRNN(look_back)
    #model = Bi_LSTM_IndRNN(look_back)
    #model = build_IndRNN_6(look_back)
    #model = Bi_LSTM4_IndRNN(look_back)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    model.fit(trainX, trainY, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr]) #Iterative training using training data with 3000 epchs

    ###########  Model loading  ###########
    #model = load_model("model.h5")
    
        
    #Prediction
    trainPredict = model.predict(trainX) #type(trainPredict)= <class 'numpy.ndarray'>
    testPredict = model.predict(testX)

    #testx = [0.]*(predict_time+look_back) #testx= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if(predict_time<=look_back):
        testx = [0.]*(2*look_back)
    else:
        testx = [0.]*((int(predict_time/look_back)+2)*look_back)

    #print("testx.shape",testx.shape)
    testx[0:look_back] = train[-look_back:]  #Read the last look_back data in the train dataset
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    Predict = [0]*predict_time  #Predict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #Use testxx (the last look_back data of testx) to predict the data
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #Add a predicted data to testx
        Predict[i] = testy

    
    Predict = numpy.array(Predict)  
    Predict = numpy.reshape(Predict,(predict_time,1))
    

    #print("Save the model")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return trainPredict, testPredict,Predict

########### training, testing, prediction ###########
def train_test_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    ###########Model training###########
    #model = build_model(look_back)
    #model = adjust_model(look_back)
    #model = build_before_model(look_back)
    #model = build_after_model(look_back)
    
    model = Bi_LSTM_IndRNN(look_back)
    #model = Bi_LSTM4_IndRNN(look_back)
    
    #model = Bi_LSTM_before_attention(look_back)
    #model = Bi_LSTM_LSTM(look_back)
    #model = Bi_LSTM_2(look_back)
    #model = build_IndRNN(look_back)
    #model = Bi_LSTM(look_back)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    #Iterative training using training data with 3000 epchs
    model.fit(train_X, train_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')

    ###########  Model loading  ###########
    #model = load_model("model.h5")

    train_Y_Predict = model.predict(train_X)
    
    #Fine-tune training after loading the model on the testing set
    #model = adjust_Bi_LSTM_model(look_back)
    model = adjust_Bi_LSTM_IndRNN(look_back)
    #model = adjust_build_IndRNN(look_back)
    #model = adjust_Bi_LSTM4_IndRNN(look_back)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    model.fit(adjust_X, adjust_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    adjust_Y_Predict = model.predict(adjust_X)
    test_Y_Predict = model.predict(test_X)

    #testx = [0.]*(predict_time+look_back) #testx= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if(predict_time<=look_back):
        testx = [0.]*(2*look_back)
    else:
        testx = [0.]*((int(predict_time/look_back)+2)*look_back)

    #print("testx.shape",testx.shape)
    testx[0:look_back] = train[-look_back:]  #Read the last look_back data in the train dataset
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    testPredict = [0]*predict_time  #testPredict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #Use testxx (the last look_back data of testx) to predict the data
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #Add a predicted data to testx
        testPredict[i] = testy

    
    testPredict = numpy.array(testPredict)  #testPredict.shape= (14, 1, 1)
    testPredict = numpy.reshape(testPredict,(predict_time,1))
    

    #print("Save the model")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return train_Y_Predict,adjust_Y_Predict,test_Y_Predict, testPredict
'''

###########  Model training  ###########
def only_train(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    ##########  Model loading  ###########
    #model = build_model(look_back)
    #model = adjust_model(look_back)
    #model = build_before_model(look_back)
    #model = build_after_model(look_back)
    
    #model = Bi_LSTM_IndRNN(look_back)
    #model = Bi_LSTM4_IndRNN(look_back)
    
    #model = Bi_LSTM_before_attention(look_back)
    #model = Bi_LSTM_LSTM(look_back)
    #model = Bi_LSTM_2(look_back)
    #model = build_IndRNN(look_back)
    #model = Bi_LSTM(look_back)
    #model = LSTM_model(look_back)
    #model = GRU_model(look_back)

    #model = Stacked_Bi_GRU(look_back)
    #model = CNN_LSTM(look_back)
    #model = deep_CNN(look_back)
    #model = CNN_IndRNN(look_back)
    #model = Stacked_IndRNN(look_back)
    model = CIRNN(look_back)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    #Iterative training using training data with 3000 epchs
    model.fit(train_X, train_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')

###########  Fine-tuning  ###########
def adjust_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    #Fine-tune training after loading the model on the testing set
    #model = adjust_Bi_LSTM_model(look_back)
    #model = adjust_Bi_LSTM_IndRNN(look_back)
    #model = adjust_build_IndRNN(look_back)
    #model = adjust_LSTM_model(look_back)
    #model = adjust_GRU_model(look_back)
    #model = adjust_Bi_LSTM4_IndRNN(look_back)

    #model = adjust_Stacked_Bi_GRU(look_back)
    #model = adjust_CNN_LSTM(look_back)
    #model = adjust_deep_CNN(look_back)
    #model = adjust_CNN_IndRNN(look_back)
    #model = adjust_Stacked_IndRNN(look_back)
    model = adjust_CIRNN(look_back)
    
    train_Y_Predict = model.predict(train_X)

    test_noadjust_start = time.clock()
    test_Y_noadjust = model.predict(test_X)
    test_noadjust_end = time.clock()
    print("test_Y_noadjust Time:",test_noadjust_end-test_noadjust_start)
    test_noadjust_time = test_noadjust_end-test_noadjust_start

    adjust_start = time.clock()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    model.fit(adjust_X, adjust_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    adjust_Y_Predict = model.predict(adjust_X)
    test_Y_Predict = model.predict(test_X)
    

    #testx = [0.]*(predict_time+look_back) #testx= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if(predict_time<=look_back):
        testx = [0.]*(2*look_back)
    else:
        testx = [0.]*((int(predict_time/look_back)+2)*look_back)

    #print("testx.shape",testx.shape)
    testx[0:look_back] = train[-look_back:]  #Read the last look_back data in the train dataset
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    testPredict = [0]*predict_time  #testPredict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #Use testxx (the last look_back data of testx) to predict the data
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #Add a predicted data to testx
        testPredict[i] = testy

    
    testPredict = numpy.array(testPredict)  #testPredict.shape= (14, 1, 1)
    testPredict = numpy.reshape(testPredict,(predict_time,1))
    
    adjust_end = time.clock()
    print("adjust Time:",adjust_end-adjust_start)
    adjust_time = adjust_end-adjust_start

    #print("Save the model")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict, testPredict

#Add the date to the data
def add_date(length,date_begin,add_number):
    Date = []
    day = datetime.datetime.strptime(date_begin, '%Y-%m-%d')
    delta=datetime.timedelta(days = add_number)
    for i in range(length):
        day = day+delta
        Date.append(day)
    return Date

#having training set
def output_havetest(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict):
    train_Y_Predict = scaler.inverse_transform(train_Y_Predict) ##Convert the standardized data to the original data
    train_Y = scaler.inverse_transform([train_Y])

    adjust_Y_Predict = scaler.inverse_transform(adjust_Y_Predict) ##Convert the standardized data to the original data
    adjust_Y = scaler.inverse_transform([adjust_Y])

    test_Y_Predict = scaler.inverse_transform(test_Y_Predict) ##Convert the standardized data to the original data
    test_Y = scaler.inverse_transform([test_Y])


    testPredict = scaler.inverse_transform(testPredict)

    '''
    data1 = pd.DataFrame(testPredict)
    data1.to_csv('E:\data_top10.csv')
    data2 = pd.DataFrame(trainPredict)
    data2.to_csv('E:\data_trainpredict.csv')
    '''

    print("==============Output accuracy==============")
    #true = test_Y[0,:] 
    #predict = test_Y_Predict[:,0]

    true = np.array([None]*(len(adjust_X)+len(test_X)))
    predict = np.array([None]*(len(adjust_X)+len(test_X)))
    print(true.shape)
    true[0:len(adjust_X)] = adjust_Y[0,:]
    true[len(adjust_X):(len(adjust_X)+len(test_X))] = test_Y[0,:]
    predict[0:len(adjust_X)] = adjust_Y_Predict[:,0]
    predict[len(adjust_X):(len(adjust_X)+len(test_X))]  = test_Y_Predict[:,0]

    print(adjust_Y)
    print(test_Y)
    print(true)
    print(adjust_Y_Predict)
    print(test_Y_Predict)
    print(predict)


    '''
    error = []
    fabs_error = []
    for m in range(0,len(true)):
        error.append((predict[m]-true[m]) / true[m])
        fabs_error.append(math.fabs((predict[m]-true[m]) / true[m]))
    #print('error:',error)
    #print('fabs_error:',fabs_error)
    #print('mean_error:',np.mean(error))
    print('mean_absolute_error:%.5f' % (np.mean(fabs_error)))
    '''
    print("==============fine-tuning==============")
    #print(mean_absolute_error(true,predict))
    print('MAPE:%.5f' % (mape(true,predict)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict))
    print('Test Score: %.2f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true, predict)))

    print("==============Predict future trends==============")
    print("testPredict=",testPredict)


    ##Show picture and view model prediction results
    train_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    train_Y_PredictPlot[look_back:len(train_Y_Predict)+look_back, :] = train_Y_Predict

    adjust_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    adjust_Y_PredictPlot[look_back+len(train_Y_Predict):look_back+len(train_Y_Predict)+len(adjust_Y_Predict), :] = adjust_Y_Predict

    test_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    test_Y_PredictPlot[look_back+len(train_Y_Predict)+len(adjust_Y_Predict):look_back+len(train_Y_Predict)+len(adjust_Y_Predict)+len(test_Y_Predict), :] = test_Y_Predict

    testPredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(dataset):(len(dataset)+predict_time), :] = testPredict

    Date = add_date(len(dataset)+predict_time,'2020-2-15',7)

    fig = plt.figure(dpi=128, figsize=(8, 5))
    plt.plot(Date[0:len(dataset)],dataset_or,linewidth = 1,label='Original data',c='red')
    plt.plot(Date, train_Y_PredictPlot,linewidth = 1, label='Training result',c='dodgerblue')
    plt.plot(Date, adjust_Y_PredictPlot,linewidth = 1, label='Adjusting result',c='yellow')
    plt.plot(Date, test_Y_PredictPlot,linewidth = 1, label='Testing result',c='green',marker='.')
    plt.plot(Date, testPredictPlot,linewidth = 1, label='Trend forecast',c='darkorange')

    fig.autofmt_xdate()
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Number of cases',fontsize=10)
    plt.legend()
    plt.show()

#No testing set
def output_notest(dataset,dataset_or,trainPredict,train_Y,testPredict,test_Y,Predict):
    trainPredict = scaler.inverse_transform(trainPredict) #Convert the standardized data to the original data
    train_Y = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    test_Y = scaler.inverse_transform([test_Y])
    Predict = scaler.inverse_transform(Predict)

    data1 = pd.DataFrame(testPredict)
    data1.to_csv('E:\data_top10.csv')
    data2 = pd.DataFrame(trainPredict)
    data2.to_csv('E:\data_trainpredict.csv')

    print("==============Output accuracy==============")

    print("Predict=",Predict)
    
    true = test_Y[0,:] 
    predict = testPredict[:,0]
    print('true:',true)
    print('predict:',predict)
    error = []
    fabs_error = []
    for m in range(0,len(true)):
        error.append((predict[m]-true[m]) / true[m])
        fabs_error.append(math.fabs((predict[m]-true[m]) / true[m]))
    #print('error:',error)
    #print('fabs_error:',fabs_error)
    #print('mean_error:',np.mean(error))
    print('mean_absolute_error:%.5f' % (np.mean(fabs_error)))

    #print(mean_absolute_error(true,predict))

    #RMSE
    print('MAPE:%.5f' % (mape(true,predict)))
    testScore = math.sqrt(mean_squared_error(test_Y[0,:], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    print('R2: %.2f' % (r2_score(test_Y[0,:], testPredict[:,0])))

    ##Show picture and view model prediction results
    trainPredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    testPredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    testPredictPlot[look_back+len(trainPredict):look_back+len(trainPredict)+len(testPredict), :] = testPredict

    PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    PredictPlot[:, :] = numpy.nan
    PredictPlot[len(dataset):(len(dataset)+predict_time), :] = Predict
    '''
    train_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    train_Y_PredictPlot[look_back:len(train_Y_Predict)+look_back, :] = train_Y_Predict

    test_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    test_Y_PredictPlot[look_back+len(train_Y_Predict):look_back+len(train_Y_Predict)+len(test_Y_Predict), :] = test_Y_Predict

    testPredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(dataset):(len(dataset)+predict_time), :] = testPredict
    '''
    Date = add_date(len(dataset)+predict_time,'2020-1-18',7)

    fig = plt.figure(dpi=128, figsize=(8, 5))
    plt.plot(Date[0:len(dataset)],dataset_or,linewidth = 1,label='Original data',c='red')
    plt.plot(Date, trainPredictPlot,linewidth = 1, label='Training result',c='dodgerblue')
    plt.plot(Date, testPredictPlot,linewidth = 1, label='Test result',c='green')
    plt.plot(Date, PredictPlot,linewidth = 1, label='Trend forecast',c='darkorange')

    fig.autofmt_xdate()
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Number of cases',fontsize=10)
    plt.legend()
    plt.show()


def output_adjust(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict):
    
    test_Y_before = numpy.reshape(test_Y,(1,test_Y.shape[0]))
    test_Y_noadjust_before = numpy.reshape(test_Y_noadjust,(test_Y_noadjust.shape[0],1))
    test_Y_Predict_before = numpy.reshape(test_Y_Predict,(test_Y_Predict.shape[0],1))
    
    print("test_Y_before",test_Y_before)
    print("test_Y_noadjust_before",test_Y_noadjust_before)
    print("test_Y_Predict_before",test_Y_Predict_before)

    print("==============No fine-tuning before data restoration:test==============")
    print(test_Y_before.shape)
    true_before = test_Y_before[0,:]
    predict_no_before = test_Y_noadjust_before[:,0]

    print("true_before",true_before)
    print("predict_no_before",predict_no_before)
    print("test_Y_Predict_before",test_Y_Predict_before[:,0])

    print('MAPE:%.5f' % (mape(true_before,predict_no_before)))
    #RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true_before, predict_no_before))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true_before, predict_no_before)))

    print("==============fine-tuning before data restoration:test==============")
    predict_adjust_before = test_Y_Predict_before[:,0]
    print('MAPE:%.5f' % (mape(true_before,predict_adjust_before)))
    #RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true_before, predict_adjust_before))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true_before, predict_adjust_before)))
    
    train_Y_Predict = scaler.inverse_transform(train_Y_Predict) #Convert the standardized data to the original data
    train_Y = scaler.inverse_transform([train_Y])
    
    adjust_Y_Predict = scaler.inverse_transform(adjust_Y_Predict) #Convert the standardized data to the original data
    adjust_Y = scaler.inverse_transform([adjust_Y])

    test_Y_Predict = scaler.inverse_transform(test_Y_Predict) #Convert the standardized data to the original data
    test_Y_noadjust= scaler.inverse_transform(test_Y_noadjust)
    test_Y = scaler.inverse_transform([test_Y])


    testPredict = scaler.inverse_transform(testPredict)
    ##the preservation of training, fine-tuning, testing results preservation
    data1 = pd.DataFrame(train_Y_Predict)
    data1.to_csv('train_Y_Predict.csv')
    data2 = pd.DataFrame(adjust_Y_Predict)
    data2.to_csv('adjust_Y_Predict.csv')
    data3 = pd.DataFrame(test_Y_Predict)
    data3.to_csv('test_Y_Predict.csv')
    data4 = pd.DataFrame(testPredict)
    data4.to_csv('testPredict.csv')

    print("test_Y",test_Y)
    print("test_Y_noadjust",test_Y_noadjust)
    print("test_Y_Predict",test_Y_Predict)
    #print("test_Y_Predict",test_Y_Predict)

    print("test_Y_noadjust",test_Y_noadjust.shape)

    print("==============No_fine-tuning (test)==============")
    print(test_Y.shape)
    true = test_Y[0,:] 
    predict_no = test_Y_noadjust[:,0]
    print('MAPE:%.5f' % (mape(true,predict_no)))
    #RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict_no))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.5f' % (r2_score(true, predict_no)))

    print("==============fine-tuning (test)==============")
    predict_adjust = test_Y_Predict[:,0]
    print('MAPE:%.5f' % (mape(true,predict_adjust)))
    #RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict_adjust))
    print('Test Score: %.2f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true, predict_adjust)))

    print("==============Predict future trends==============")
    print("testPredict=",testPredict)


    #Show picture and view model prediction results
    train_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    train_Y_PredictPlot[look_back:len(train_Y_Predict)+look_back, :] = train_Y_Predict

    adjust_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    adjust_Y_PredictPlot[look_back+len(train_Y_Predict):look_back+len(train_Y_Predict)+len(adjust_Y_Predict), :] = adjust_Y_Predict

    test_Y_PredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    test_Y_PredictPlot[look_back+len(train_Y_Predict)+len(adjust_Y_Predict):look_back+len(train_Y_Predict)+len(adjust_Y_Predict)+len(test_Y_Predict), :] = test_Y_Predict

    testPredictPlot = numpy.reshape(numpy.array([None]*(len(dataset)+predict_time)),((len(dataset)+predict_time),1))
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(dataset):(len(dataset)+predict_time), :] = testPredict

    Date = add_date(len(dataset)+predict_time,'2020-2-15',7)

    fig = plt.figure(dpi=128, figsize=(8, 5))
    plt.plot(Date[0:len(dataset)],dataset_or,linewidth = 1,label='Original data',c='red')
    plt.plot(Date, train_Y_PredictPlot,linewidth = 1, label='Training result',c='dodgerblue',marker='.')
    plt.plot(Date, adjust_Y_PredictPlot,linewidth = 1, label='Fine-tuning result',c='yellow',marker='.')
    plt.plot(Date, test_Y_PredictPlot,linewidth = 1, label='Testing result',c='green',marker='.')
    plt.plot(Date, testPredictPlot,linewidth = 1, label='Trend forecast',c='darkorange',marker='.')

    fig.autofmt_xdate()
    plt.xlabel('Date',fontsize=10)
    plt.ylabel('Number of cases',fontsize=10)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    start = time.clock()

    #The cumulative confirmed cases of top 5 in the world
    #US:     usecols-1, skiprows-0
    #India:  usecols-2, skiprows-1
    #Brazil: usecols-3, skiprows-5
    #France: usecols-4, skiprows-0
    #Russia: usecols-5, skiprows-1
    dataframe = read_csv('data/covid19_confirmed_global_top5_week.csv', encoding='gbk', usecols=[2], skiprows=1)
    
    #The cumulative death cases of top 5 in the world
    #US:     usecols-1, skiprows-5
    #India:  usecols-2, skiprows-7
    #Brazil: usecols-3, skiprows-8
    #France: usecols-4, skiprows-3
    #Russia: usecols-5, skiprows-8
    #dataframe = read_csv('data/covid19_deaths_global_top5_week.csv', encoding='gbk', usecols=[2], skiprows=7)



    # The cumulative confirmed and death cases of China
    #dataframe = read_csv('China-daily-data.CSV', encoding='utf-8', usecols=[2], skiprows=None)
    #dataframe = read_csv('data/China-daily-data-week.CSV', encoding='utf-8', usecols=[2], skiprows=None)
    
    #The cumulative confirmed and death cases of the world
    #dataframe = read_csv('data/covid19_confirmed_global.csv', encoding='gbk', usecols=[6], skiprows=None)

    dataset = dataframe.values  # numpy.ndarray
    dataset_or = dataset

    # Set random seed, standardize data
    numpy.random.seed(7)
    # Normalized
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train = dataset  
    
    
    #========================================================================
    # Set time sliding window, create dataset
    trainX, trainY = create_dataset(train, look_back)
    ###########Reshape the dataset##########
    print(trainX.shape)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  # (53, 1, 7) 
    
    #0-No testing set, no fine-tuning，1-testing set, fine-tuning
    flag = 2
    tune = 5
    if flag ==0:
        ###########No testing set##########
        train_rate = int(trainX.shape[0]-6)
        train_X = trainX[0:train_rate]
        test_X = trainX[train_rate:trainX.shape[0]]
        train_Y = trainY[0:train_rate]
        test_Y = trainY[train_rate:trainX.shape[0]]
        trainPredict, testPredict,Predict = train_model(train_X, train_Y, test_X,test_Y,train, predict_time)
        output_notest(dataset,dataset_or,trainPredict,train_Y,testPredict,test_Y,Predict)
    if flag == 1:
        ###########Testing set###########
        #train_rate = int(trainX.shape[0]*0.8)
        train_rate = int(trainX.shape[0]-6)
        train_X = trainX[0:train_rate]
        adjust_X = trainX[train_rate:train_rate+tune]
        test_X = trainX[train_rate+tune:trainX.shape[0]]

        train_Y = trainY[0:train_rate]
        adjust_Y = trainY[train_rate:train_rate+tune]
        test_Y = trainY[train_rate+tune:trainX.shape[0]]
        
        print(trainX.shape)
        print(adjust_X.shape)
        print(test_X.shape)
    
        #train_Y_Predict,adjust_Y_Predict,test_Y_Predict, testPredict = train_test_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time)
        #output_havetest(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict)
        only_train(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time)

    if flag == 2:
        train_rate = int(trainX.shape[0]-6)
        train_X = trainX[0:train_rate]
        adjust_X = trainX[train_rate:train_rate+tune]
        test_X = trainX[train_rate+tune:trainX.shape[0]]

        train_Y = trainY[0:train_rate]
        adjust_Y = trainY[train_rate:train_rate+tune]
        test_Y = trainY[train_rate+tune:trainX.shape[0]]
        
        print(trainX.shape)
        print(adjust_X.shape)
        print(test_X.shape)
    
        #train_X,test_X,train_Y,test_Y = train_test_split(trainX, trainY,test_size = 0.3,random_state = 0)
        train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict, testPredict = adjust_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time)
        output_adjust(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict)


    print("test_Y_noadjust Time:",test_noadjust_time)
    print("adjust Time:",adjust_time)

    #Print running time
    end = time.clock()
    print("Runtime is: ",end-start)
    K.clear_session()
