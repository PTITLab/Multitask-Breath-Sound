import numpy as np
import os
import json
import pickle

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import keras
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model
from keras.backend.tensorflow_backend import set_session
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint

from LSTM_MODEL import LSTM_MODEL
from dataset import BreathDataGenerator

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

from training_constant import (
    BATCH_SIZE,
    LIST_LABELS,
    N_CLASSES,
    N_EPOCHS,
    INPUT_SIZE,
    TRAINING_SOURCE,
    VALID_SOURCE,
    MODEL_OUTPUT,
    RUN_TITLE,
)


# Generate data for training
train_generator = BreathDataGenerator(
        TRAINING_SOURCE,
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)

N_TRAIN_SAMPLES = len(train_generator.wavs)
print("Train samples: {}".format(N_TRAIN_SAMPLES))

validation_generator = BreathDataGenerator(
        VALID_SOURCE,
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
N_VALID_SAMPLES = len(validation_generator.wavs)
print("Validation samples: {}".format(N_VALID_SAMPLES))

# build LSTM model 

# model = LSTM_MODEL.build_simple_lstm(data_input_shape=INPUT_SIZE, classes=N_CLASSES, learning_rate=0.001)

# model = LSTM_MODEL.build_residual_bilstm(data_input_shape=INPUT_SIZE, classes=N_CLASSES, learning_rate=0.001)

# model = LSTM_MODEL.build_bilstm(data_input_shape=INPUT_SIZE, classes=N_CLASSES, learning_rate=0.001)

model = LSTM_MODEL.get_trainable_model_bilstm(data_input_shape=INPUT_SIZE, classes=N_CLASSES, learning_rate=0.001)

model.summary()

# Checkpoint
if not os.path.exists(MODEL_OUTPUT):
    os.makedirs(MODEL_OUTPUT)

filepath= os.path.join(MODEL_OUTPUT, RUN_TITLE + "LSTM-weights-improvement_bi_lstm-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5") 
# filepath="./model_output/LSTM-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, mode='max')
callbacks_list = [checkpoint]

# Start training
model.fit_generator(
        train_generator,
        steps_per_epoch= N_TRAIN_SAMPLES // BATCH_SIZE,
        initial_epoch=0,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=N_VALID_SAMPLES // BATCH_SIZE,
        callbacks=callbacks_list,
        #use_multiprocessing=True,
)

