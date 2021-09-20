import numpy as np
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import pickle
from scipy.io import wavfile

# os.environ["CUDA_DEVICE_ORDER"]="0"
import tensorflow as tf
from tensorflow import keras as K

with tf.device('/device:GPU:0'):

    import tensorflow.keras as keras
    import tensorflow as tf

# Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices, True)

# from tensorflow.keras.utils import multi_gpu_model
# from keras.backend.tensorflow_backend import set_session
    import librosa
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Permute, Reshape, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Flatten
    from tensorflow.compat.v1.keras.layers import CuDNNLSTM as CuLSTM
    from tensorflow.compat.v1.keras.layers import Input, Dense, Lambda, Layer
    from tensorflow.keras.layers import add
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers.experimental import preprocessing
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import LeakyReLU

    normalize = preprocessing.Normalization()

def get_recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    # f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return recall

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


class LSTM_MODEL(object):
    @staticmethod
    def build_simple_lstm(data_input_shape, classes, learning_rate):
        model = Sequential()
        model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=data_input_shape))
        model.add(LSTM(units=128,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=classes, activation="softmax"))
        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
    
    @staticmethod
    def build_bilstm(data_input_shape, classes, learning_rate):
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=data_input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()

        return model

    @staticmethod
    def build_residual_bilstm(data_input_shape, classes, learning_rate):
        inp = Input(shape=data_input_shape)
        z1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
        z2 = Bidirectional(LSTM(units=128, return_sequences=True))(z1)
        z3 = add([z1, z2])  # residual connection
        z4 = Bidirectional(LSTM(units=128, return_sequences=True))(z3)
        z5 = Bidirectional(LSTM(units=128, return_sequences=False))(z4)
        z6 = add([z4, z5])  # residual connection    
        z61 = Flatten()(z6)        
        z7 = Dense(256, activation='relu')(z61)
        z8 = Dropout(0.5)(z7)
        out = Dense(classes, activation='softmax')(z8)
        model = Model(inputs=[inp], outputs=out)
        opt = Adam(lr=learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.summary()
        return model

    @staticmethod
    def get_trainable_model_bilstm(data_input_shape, classes, learning_rate):
        inp1 = Input(shape=data_input_shape) # detect
        inp2 = Input(shape=data_input_shape) # classify

        z1 = Bidirectional(CuLSTM(128, return_sequences=True))(inp1)
        z2 = Bidirectional(CuLSTM(units=128, return_sequences=True))(z1)
        z3 = add([z1, z2])  # residual connection
        z4 = Bidirectional(CuLSTM(units=128, return_sequences=True))(z3)
        z5 = Bidirectional(CuLSTM(units=128, return_sequences=False))(z4)
        z6 = add([z4, z5])  # residual connection    
        z61 = Flatten()(z6) 
        z7 = Dense(256, activation='relu')(z61)
        z8 = Dropout(0.5)(z7)
        # out = Dense(classes, activation='softmax')(z8)
        out1 = Dense(3, activation='softmax', name='output_1')(z8)
        out2 = Dense(2, activation='softmax', name='output_2')(z8)

        model = Model(inputs=inp1, outputs=[out1, out2])
        opt = Adam(lr=learning_rate)
        losses = {
            "output_1": "categorical_crossentropy",
            "output_2": "categorical_crossentropy"
        }
        metricss = {
            "output_1": ['accuracy', tf.keras.metrics.Precision(), get_recall, get_f1],
            "output_2": ['accuracy', tf.keras.metrics.Precision(), get_recall, get_f1]
        }
        model.compile(loss=losses, optimizer=opt, metrics=metricss)
        model.summary()
        return model

