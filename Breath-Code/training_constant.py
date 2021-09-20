# Set the config for training

BATCH_SIZE = 32
LIST_LABELS = ['normal', 'deep', 'strong']
N_CLASSES = len(LIST_LABELS)
N_EPOCHS = 30
# INPUT_SIZE = (40, 126, 1) # Input size for CNN training
INPUT_SIZE = (40, 126) # Input size for LSTM training
TRAINING_SOURCE = '/home/trananhdat/tupa/breath_data_preprocessed/5s_50/train'
VALID_SOURCE = '/home/trananhdat/tupa/breath_data_preprocessed/5s_50/test'
MODE = 'TRAINING'
MODEL_OUTPUT = './model_output'

RUN_TITLE = "5S-50-Residual"