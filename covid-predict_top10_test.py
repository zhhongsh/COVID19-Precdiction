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
from keras.layers import LSTM, Activation, Dropout,Bidirectional
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

import random

#import os
#os.environ["PATH"] += os.pathsep + '/usr/local/lib/python3.5/dist-packages/graphviz/bin/'
#from keras.callbacks import ModelCheckpoint, Tensorboad
#checkpoint = ModelCheckpoint(filepath='model_nice.h5',monitor = 'loss',save_best_only = 'True',mode='min',period = 1) #存放最好模型的地方
#Tensorboad = Tensorboad(log_dir='log')
#monitor = 'loss'  #最想监视的值
#verbose = 1 #进度条
#save_best_only = 'True'  #只保存最好的模型
#mode='min'
#period = 1 #checkpoint之间间隔的epoch数

#plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
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

#逆转数据
def reverse_data(data):
    reverse_dataset = []
    j = len(dataset) - 1
    for i in range(len(data)):
        reverse_dataset.append(data[j])
        j = j - 1
    reverse_dataset = numpy.array(reverse_dataset)
    return reverse_dataset

#为后续lstm的输入创建一个数据处理函数
def create_dataset(dataset, look_back=1):#look_back为滑窗
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):   #range(60-7)调用为7
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_time_steps(dataset, look_back=1):#look_back为滑窗
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):   #range(60-7)调用为7
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

#注意力机制
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

#搭建lstm网络
def build_before_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    attention_mul = attention_3d_block(inputs)
    lstm_out1 = LSTM(25, return_sequences=True)(attention_mul)
    lstm_out2 = LSTM(50, return_sequences=True)(lstm_out1)
    lstm_out3 = LSTM(50, return_sequences=False)(lstm_out2)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(1)(lstm_out3)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model

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
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model

#搭建lstm网络
def build_model(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    #plot_model(model,to_file = '../Model_3lstm.jpg',show_shapes = True)
    model.summary()
    return model

#微调lstm网络
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
    model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back

    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))

    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    #x = base_model.output
    #x = Dense(1,activation = 'sigmoid')(x)
    #output = Activation('linear')(x)
    #model = Model(input=[inputs], output=output)

    for layer in model.layers[:3]:     #冻结dense层之前的参数
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
   

    #plot_model(model,to_file = '../Model_3lstm.jpg',show_shapes = True)
    model.summary()
    return model

def LSTM_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm= LSTM(128,return_sequences=True)(inputs)
    lstm=Flatten()(lstm)
    dense_out = Dense(1)(lstm)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def adjust_LSTM_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    lstm= LSTM(128,return_sequences=True)(inputs)
    lstm=Flatten()(lstm)
    dense_out = Dense(1)(lstm)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    for layer in model.layers[:2]:     #冻结dense层之前的参数
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def GRU_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    gru= GRU(128,return_sequences=True)(inputs)
    gru=Flatten()(gru)
    dense_out = Dense(1)(gru)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def adjust_GRU_model(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    gru= GRU(128,return_sequences=True)(inputs)
    gru=Flatten()(gru)
    dense_out = Dense(1)(gru)
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    for layer in model.layers[:2]:     #冻结dense层之前的参数
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def Bi_LSTM(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(128, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    #model.add(Bidirectional(LSTM(32, return_sequences=True)))
    #model.add(Bidirectional(LSTM(16, return_sequences=True)))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def adjust_Bi_LSTM_model(look_back):
    '''
    model = Sequential()

    model.add(Bidirectional(LSTM(128, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    '''
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

    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    for layer in model.layers[:3]:     #冻结dense层之前的参数
        layer.trainable = False  

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def Bi_LSTM_LSTM(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(25, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    model.add(Dropout(0.5))
    model.add(LSTM(50, return_sequences=True))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def Bi_LSTM_2(look_back):
    model = Sequential()
    #model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, look_back)))
    model.add(Bidirectional(LSTM(25, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    model.add(Bidirectional(LSTM(50,return_sequences=True)))
    model.add(Flatten())
    #model.add(Dropout(0.6))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

def Bi_LSTM_before_attention(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    bi_lstm  = Bidirectional(LSTM(25,return_sequences=True))(attention_mul)
    fla = Flatten()(bi_lstm)
    dense_out = Dense(1)(fla)
    output = Activation('linear')(dense_out)

    model = Model(input=[inputs], output=output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))

    model.summary()
    return model

def build_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    indrnn = IndRNN(128)(inputs)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(1)(indrnn)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model

def adjust_build_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    indrnn = IndRNN(128)(inputs)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(1)(indrnn)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.load_weights('my_model_weights.h5',by_name = True)  #加载模型权重

    for layer in model.layers[:2]:     #冻结dense层之前的参数
        layer.trainable = False  

    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))

    model.summary()
    return model

def build_IndRNN_6(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    indrnn = IndRNN(128)(inputs)
    indrnn = Reshape((None,1,128))(indrnn)
    indrnn = IndRNN(128)(indrnn)
    indrnn = Reshape((None,1,128))(indrnn)
    indrnn = IndRNN(128)(indrnn)
    indrnn = Reshape((None,1,128))(indrnn)
    indrnn = IndRNN(128)(indrnn)
    indrnn = Reshape((None,1,128))(indrnn)
    indrnn = IndRNN(128)(indrnn)
    indrnn = Reshape((None,1,128))(indrnn)
    indrnn = IndRNN(128)(indrnn)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
    dense_out = Dense(1)(indrnn)
    #model.add(Dropout(0.6))
    output = Activation('linear')(dense_out)
    model = Model(input=[inputs], output=output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model

def Bi_LSTM_IndRNN(look_back):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(128)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
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
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop

    model.summary()
    return model
'''
def Bi_LSTM_IndRNN(look_back):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    model.add(IndRNN(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(4))
    model.add(Dense(2))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model
'''
def adjust_Bi_LSTM_IndRNN(look_back):
    '''
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(1, look_back),return_sequences=True)))#输出节点为25，输入的每个样本长度为look_back
    model.add(IndRNN(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Dense(4))
    model.add(Dense(2))
    model.add(Dense(1))#添加一个全连接层，输出维度为1
    model.add(Activation('linear'))
    '''
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    #lstm_units = 25
    
    bi_lstm= Bidirectional(LSTM(128,return_sequences=True))(inputs)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm= Bidirectional(LSTM(64,return_sequences=True))(bi_lstm)
    #bi_lstm = Dropout(0.5)(bi_lstm)
    indrnn = IndRNN(128)(bi_lstm)
    #model.add(LSTM(25, input_shape=(1, look_back),return_sequences=True))#输出节点为25，输入的每个样本长度为look_back
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

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#使用均方差做损失函数。优化器用adam、rmsprop
    model.build((None,1, look_back))
    model.summary()
    return model

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

#训练模型，预测
def train_model(trainX, trainY, testX,testY,train, predict_time):
    ###########训练模型###########
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

    #ReduceLROnPlateau,当指标停止提升时，降低学习速率。
    #一旦学习停止，模型通常会将学习率降低2-10倍。该回调监测数量，如果没有看到epoch的 'patience' 数量的改善，那么学习率就会降低。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    model.fit(trainX, trainY, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])#训练模型，100epoch，批次为1，每一个epoch显示一次日志，学习率动态减小

    ###########若使用加载好的模型，则直接预测###########
    #model = load_model("model.h5")
    
        
    #预测-无测试集
    trainPredict = model.predict(trainX)#预测训练集type(trainPredict)= <class 'numpy.ndarray'>
    testPredict = model.predict(testX)

    #testx = [0.]*(predict_time+look_back) #testx= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if(predict_time<=look_back):
        testx = [0.]*(2*look_back)
    else:
        testx = [0.]*((int(predict_time/look_back)+2)*look_back)

    #print("testx.shape",testx.shape)
    testx[0:look_back] = train[-look_back:]  #train（即读取的60个数据）中后7个
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    Predict = [0]*predict_time  #Predict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #用testxx(testx的后七个数据)来预测数据
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #预测出来的一个数据加入testx中
        Predict[i] = testy

    
    Predict = numpy.array(Predict)  #testPredict.shape= (14, 1, 1)
    Predict = numpy.reshape(Predict,(predict_time,1))
    

    #print("保存模型")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return trainPredict, testPredict,Predict

#训练模型，预测
def train_test_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    ###########训练模型###########
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

    
    #ReduceLROnPlateau,当指标停止提升时，降低学习速率。
    #一旦学习停止，模型通常会将学习率降低2-10倍。该回调监测数量，如果没有看到epoch的 'patience' 数量的改善，那么学习率就会降低。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    
    model.fit(train_X, train_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')

    ###########若使用加载好的模型，则直接预测###########
    #model = load_model("model.h5")

    train_Y_Predict = model.predict(train_X)
    
    #测试集加载模型后微调训练
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
    testx[0:look_back] = train[-look_back:]  #train（即读取的60个数据）中后7个
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    testPredict = [0]*predict_time  #testPredict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #用testxx(testx的后七个数据)来预测数据
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #预测出来的一个数据加入testx中
        testPredict[i] = testy

    
    testPredict = numpy.array(testPredict)  #testPredict.shape= (14, 1, 1)
    testPredict = numpy.reshape(testPredict,(predict_time,1))
    

    #print("保存模型")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return train_Y_Predict,adjust_Y_Predict,test_Y_Predict, testPredict

def only_train(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    ###########训练模型###########
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
    model = Bi_LSTM(look_back)
    #model = LSTM_model(look_back)
    #model = GRU_model(look_back)

    
    #ReduceLROnPlateau,当指标停止提升时，降低学习速率。
    #一旦学习停止，模型通常会将学习率降低2-10倍。该回调监测数量，如果没有看到epoch的 'patience' 数量的改善，那么学习率就会降低。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto',epsilon = 0.0001,cooldown = 0,min_lr = 0)
    
    model.fit(train_X, train_Y, epochs=3000, batch_size=1, verbose=2, callbacks=[reduce_lr])
    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')

def adjust_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time):
    #测试集加载模型后微调训练
    #model = adjust_Bi_LSTM_model(look_back)
    #model = adjust_Bi_LSTM_IndRNN(look_back)
    model = adjust_build_IndRNN(look_back)
    #model = adjust_LSTM_model(look_back)
    #model = adjust_GRU_model(look_back)
    #model = adjust_Bi_LSTM4_IndRNN(look_back)

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
    testx[0:look_back] = train[-look_back:]  #train（即读取的60个数据）中后7个
    testx = numpy.array(testx) #type(testx)= <class 'numpy.ndarray'>
    testPredict = [0]*predict_time  #testPredict= [0, 0, 0, 0, 0, 0, 0]

    for i in range(predict_time):
        testxx = testx[i:look_back+i]

        testxx = numpy.reshape(testxx, (1, 1, look_back))
        #print("testxx=",testxx)
        testy = model.predict(testxx)  #用testxx(testx的后七个数据)来预测数据
        if(testy<0):
            testy=-testy
        testx[look_back+i] = testy  #预测出来的一个数据加入testx中
        testPredict[i] = testy

    
    testPredict = numpy.array(testPredict)  #testPredict.shape= (14, 1, 1)
    testPredict = numpy.reshape(testPredict,(predict_time,1))
    
    adjust_end = time.clock()
    print("adjust Time:",adjust_end-adjust_start)
    adjust_time = adjust_end-adjust_start

    #print("保存模型")
    #model.save('model.h5')
    #model.save_weights('my_model_weights.h5')

    return train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict, testPredict

#为数据增加日期，并画图
def subtract_date(length,date_begin):
    Date = []
    day = datetime.datetime.strptime(date_begin, '%Y/%m/%d')
    delta=datetime.timedelta(days=1)
    for i in range(length):
        day = day-delta
        Date.append(day)
    return Date

def add_date(length,date_begin,add_number):
    Date = []
    day = datetime.datetime.strptime(date_begin, '%Y-%m-%d')
    delta=datetime.timedelta(days = add_number)
    for i in range(length):
        day = day+delta
        Date.append(day)
    return Date

#有训练集
def output_havetest(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict):
    train_Y_Predict = scaler.inverse_transform(train_Y_Predict) #将标准化后的数据转换为原始数据
    train_Y = scaler.inverse_transform([train_Y])

    adjust_Y_Predict = scaler.inverse_transform(adjust_Y_Predict) #将标准化后的数据转换为原始数据
    adjust_Y = scaler.inverse_transform([adjust_Y])

    test_Y_Predict = scaler.inverse_transform(test_Y_Predict) #将标准化后的数据转换为原始数据
    test_Y = scaler.inverse_transform([test_Y])


    testPredict = scaler.inverse_transform(testPredict)

    '''
    data1 = pd.DataFrame(testPredict)
    data1.to_csv('E:\data_top10.csv')
    data2 = pd.DataFrame(trainPredict)
    data2.to_csv('E:\data_trainpredict.csv')
    '''

    print("==============输出精度==============")
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
    print("==============微调==============")
    #print(mean_absolute_error(true,predict))
    print('MAPE:%.5f' % (mape(true,predict)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict))
    print('Test Score: %.2f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true, predict)))

    print("==============预测未来趋势==============")
    print("testPredict=",testPredict)


    #画图查看模型预测结果
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
    #plt.title('全国每日新增病例预测', fontsize=20)
    #plt.savefig('E:\研究生\新冠\picture\全国.png')
    plt.show()

def output_notest(dataset,dataset_or,trainPredict,train_Y,testPredict,test_Y,Predict):
    #无训练集
    trainPredict = scaler.inverse_transform(trainPredict) #将标准化后的数据转换为原始数据
    train_Y = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    test_Y = scaler.inverse_transform([test_Y])
    Predict = scaler.inverse_transform(Predict)

    data1 = pd.DataFrame(testPredict)
    data1.to_csv('E:\data_top10.csv')
    data2 = pd.DataFrame(trainPredict)
    data2.to_csv('E:\data_trainpredict.csv')

    print("==============输出精度==============")

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

    #输出训练RMSE
    print('MAPE:%.5f' % (mape(true,predict)))
    testScore = math.sqrt(mean_squared_error(test_Y[0,:], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    print('R2: %.2f' % (r2_score(test_Y[0,:], testPredict[:,0])))

    #画图查看模型预测结果
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
    #plt.title('全国每日新增病例预测', fontsize=20)
    #plt.savefig('E:\研究生\新冠\picture\全国.png')
    plt.show()

def dataset_split_train_test(dataX,dataY):
    a = range(4,len(dataX)-4)
    i = random.sample(a,4)
    print(i) 
    i.sort()
    print(i)
    trainX = numpy.reshape(numpy.array([None]*((len(dataX)-4)*look_back)),(len(dataX)-4,look_back))
    trainY = numpy.array([None]*(len(dataX)-4))
   
    print(dataX.shape)
    print(trainX.shape)
    trainX[0:i[0],:] = dataX[0:i[0],:]
    trainX[i[0]:i[1]-1,:] = dataX[i[0]+1:i[1],:]
    trainX[i[1]-1:i[2]-2,:] = dataX[i[1]+1:i[2],:]
    trainX[i[2]-2:i[3]-3,:] = dataX[i[2]+1:i[3],:]
    trainX[i[3]-3:len(dataX)-4,:] = dataX[i[3]+1:len(dataX),:]
    print(trainX.shape)

    print(dataY.shape)
    print(trainY.shape)
    trainY[0:i[0]] = dataY[0:i[0]]
    trainY[i[0]:i[1]-1] = dataY[i[0]+1:i[1]]
    trainY[i[1]-1:i[2]-2] = dataY[i[1]+1:i[2]]
    trainY[i[2]-2:i[3]-3] = dataY[i[2]+1:i[3]]
    trainY[i[3]-3:len(dataX)-4] = dataY[i[3]+1:len(dataX)]
    print(trainY.shape)

    testX = numpy.reshape(numpy.array([None]*(4*look_back)),(4,look_back))
    testY = numpy.array([None]*4)
    testX[0] = dataX[i[0]]
    testX[1] = dataX[i[1]]
    testX[2] = dataX[i[2]]
    testX[3] = dataX[i[3]]
    print(testX.shape)

    testY[0] = dataY[i[0]]
    testY[1] = dataY[i[1]]
    testY[2] = dataY[i[2]]
    testY[3] = dataY[i[3]]
    print(testY.shape)
    return trainX,trainY,testX,testY

def output_adjust(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,test_Y_noadjust,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict):
    
    test_Y_before = numpy.reshape(test_Y,(1,test_Y.shape[0]))
    test_Y_noadjust_before = numpy.reshape(test_Y_noadjust,(test_Y_noadjust.shape[0],1))
    test_Y_Predict_before = numpy.reshape(test_Y_Predict,(test_Y_Predict.shape[0],1))
    
    print("test_Y_before",test_Y_before)
    print("test_Y_noadjust_before",test_Y_noadjust_before)
    print("test_Y_Predict_before",test_Y_Predict_before)

    print("==============还原前无微调test==============")
    print(test_Y_before.shape)
    true_before = test_Y_before[0,:]
    predict_no_before = test_Y_noadjust_before[:,0]

    print("true_before",true_before)
    print("predict_no_before",predict_no_before)
    print("test_Y_Predict_before",test_Y_Predict_before[:,0])

    print('MAPE:%.5f' % (mape(true_before,predict_no_before)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true_before, predict_no_before))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true_before, predict_no_before)))

    print("==============还原前有微调test==============")
    predict_adjust_before = test_Y_Predict_before[:,0]
    print('MAPE:%.5f' % (mape(true_before,predict_adjust_before)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true_before, predict_adjust_before))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true_before, predict_adjust_before)))
    
    train_Y_Predict = scaler.inverse_transform(train_Y_Predict) #将标准化后的数据转换为原始数据
    train_Y = scaler.inverse_transform([train_Y])
    
    adjust_Y_Predict = scaler.inverse_transform(adjust_Y_Predict) #将标准化后的数据转换为原始数据
    adjust_Y = scaler.inverse_transform([adjust_Y])

    test_Y_Predict = scaler.inverse_transform(test_Y_Predict) #将标准化后的数据转换为原始数据
    test_Y_noadjust= scaler.inverse_transform(test_Y_noadjust)
    test_Y = scaler.inverse_transform([test_Y])


    testPredict = scaler.inverse_transform(testPredict)
    ##拟合、微调、测试结果保存
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

    print("==============无微调test==============")
    print(test_Y.shape)
    true = test_Y[0,:] 
    predict_no = test_Y_noadjust[:,0]
    print('MAPE:%.5f' % (mape(true,predict_no)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict_no))
    print('Test Score: %.5f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.5f' % (r2_score(true, predict_no)))

    print("==============有微调test==============")
    predict_adjust = test_Y_Predict[:,0]
    print('MAPE:%.5f' % (mape(true,predict_adjust)))
    #输出训练RMSE
    #testScore = math.sqrt(mean_squared_error(test_Y[0,:], test_Y_Predict[:,0]))
    testScore = math.sqrt(mean_squared_error(true, predict_adjust))
    print('Test Score: %.2f RMSE' % (testScore))

    #print('R2: %.2f' % (r2_score(test_Y[0,:], test_Y_Predict[:,0])))
    print('R2: %.2f' % (r2_score(true, predict_adjust)))

    print("==============预测未来趋势==============")
    print("testPredict=",testPredict)


    #画图查看模型预测结果
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
    #plt.title('全国每日新增病例预测', fontsize=20)
    #plt.savefig('E:\研究生\新冠\picture\全国.png')
    plt.show()

if __name__ == '__main__':
    start = time.clock()
    #全球累计确诊前10  2020-2-9~2020-6-21
    #1-1-巴西 2-智利  3-印度  4-伊朗  5-墨西哥 6-秘鲁 7-俄罗斯 8-西班牙 9-英国 10-美国
    #
    #print(dataframe)
    dataframe = read_csv('data/covid19_confirmed_global_top5_week.csv', encoding='gbk', usecols=[5], skiprows=1)
    #dataframe = read_csv('data/covid19_deaths_global_top5_week.csv', encoding='gbk', usecols=[5], skiprows=8)

    #exit(0)

    # 全国累计确诊2020-1-19~6-14 ，每周数据，共22条
    #dataframe = read_csv('China-daily-data.CSV', encoding='utf-8', usecols=[2], skiprows=None)
    #dataframe = read_csv('data/China-daily-data-week.CSV', encoding='utf-8', usecols=[2], skiprows=None)
    #全球累计确诊2-16～7-5 共21条数据
    #dataframe = read_csv('data/covid19_confirmed_global.csv', encoding='gbk', usecols=[6], skiprows=None)

    dataset = dataframe.values  # numpy.ndarray
    dataset_or = dataset

    # 设置随机种子，标准化数据
    numpy.random.seed(7)
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train = dataset  # (60,1)
    
    
    #==================================test为后四周======================================
    # 设置时间滑窗，创建训练集
    trainX, trainY = create_dataset(train, look_back)
    ###########对训练集x做reshape##########
    print(trainX.shape)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  # (53, 1, 7) 
    
    #0-无训练集不微调，1-有训练集微调
    flag = 2
    tune = 5
    if flag ==0:
        ###########无训练集##########
        train_rate = int(trainX.shape[0]-6)
        train_X = trainX[0:train_rate]
        test_X = trainX[train_rate:trainX.shape[0]]
        train_Y = trainY[0:train_rate]
        test_Y = trainY[train_rate:trainX.shape[0]]
        trainPredict, testPredict,Predict = train_model(train_X, train_Y, test_X,test_Y,train, predict_time)
        output_notest(dataset,dataset_or,trainPredict,train_Y,testPredict,test_Y,Predict)
    if flag == 1:
        ###########有训练集###########
        #train_rate = int(trainX.shape[0]*0.8)
        train_rate = int(trainX.shape[0]-6)
        train_X = trainX[0:train_rate]
        adjust_X = trainX[train_rate:train_rate+3]
        test_X = trainX[train_rate+3:trainX.shape[0]]

        train_Y = trainY[0:train_rate]
        adjust_Y = trainY[train_rate:train_rate+3]
        test_Y = trainY[train_rate+3:trainX.shape[0]]
        
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
    '''
    #==================================test为后四周前取随机======================================
    # 设置时间滑窗，创建训练集
    dataX, dataY = create_dataset(train, look_back)
    trainX,trainY,testX,testY = dataset_split_train_test(dataX,dataY)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))  # (53, 1, 7) 
    
    #0-无训练集不微调，1-有训练集微调
    flag = 0
    if flag ==0:
        ###########无训练集##########
        trainPredict, testPredict,Predict = train_model(trainX, trainY, testX,testY,train, predict_time)
        output_notest(dataset,dataset_or,trainPredict,trainY,testPredict,testY,Predict)
    else:
        ###########有训练集###########
        train_rate = int(trainX.shape[0]-4)
        train_X = trainX[0:train_rate]
        adjust_X = trainX[train_rate:train_rate+4]
        test_X = trainX[train_rate+4:trainX.shape[0]]

        train_Y = trainY[0:train_rate]
        adjust_Y = trainY[train_rate:train_rate+4]
        test_Y = trainY[train_rate+4:trainX.shape[0]]
    
        #train_X,test_X,train_Y,test_Y = train_test_split(trainX, trainY,test_size = 0.3,random_state = 0)
        train_Y_Predict,adjust_Y_Predict,test_Y_Predict, testPredict = train_test_model(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y, train, predict_time)
        output_havetest(train_X,adjust_X,test_X,train_Y,adjust_Y,test_Y,train_Y_Predict,adjust_Y_Predict,test_Y_Predict,dataset,dataset_or,testPredict)
    '''
    ###########加载模型##########
    '''
    sampleX = trainX[trainX.shape[0]-1]
    sampleY = numpy.array(trainX.shape[0]-1)
    sampleX = numpy.reshape(sampleX, (1, 1, look_back))
    trainPredict, testPredict = train_model(sampleX, sampleY, train, predict_time)
    '''


    print("test_Y_noadjust Time:",test_noadjust_time)
    print("adjust Time:",adjust_time)

    #输出时间
    end = time.clock()
    print("Runtime is: ",end-start)
    K.clear_session()
