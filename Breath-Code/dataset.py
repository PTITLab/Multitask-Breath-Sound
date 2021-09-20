import numpy as np
import keras
from scipy.io import wavfile
import librosa
import os
from keras.utils import to_categorical

class BreathDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory, 
                    list_labels=['normal', 'deep', 'strong'], 
                    batch_size=32,
                    dim=None,
                    classes=None, 
                    shuffle=True):
        'Initialization'
        self.directory = directory
        self.list_labels = list_labels
        self.dim = dim
        self.__flow_from_directory(self.directory)
        self.batch_size = batch_size
        self.classes = len(self.list_labels)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.wavs) / self.batch_size))

    def __getitem__(self, index):
        # print("In get Item!!")
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        rawX = [self.wavs[k] for k in indexes]
        rawY = [self.labels[k] for k in indexes]

        # Generate data
        X, Y = self.__feature_extraction(rawX, rawY)
        # print("Done getting data")
        return X, Y

    def __flow_from_directory(self, directory):
        self.wavs = []
        self.labels = []
        for dir in os.listdir(directory):
            sub_dir = os.path.join(directory, dir)
            if os.path.isdir(sub_dir) and dir in self.list_labels:
                label = self.list_labels.index(dir)
                for file in os.listdir(sub_dir):
                    self.wavs.append(os.path.join(sub_dir, file))
                    self.labels.append(label)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.wavs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __feature_extraction(self, list_wav, list_label):
        # print("Go to feature extraction!!!")
        'Generates data containing batch_size samples'
        X = []
        Y = []
        for i in range(self.batch_size):
            rate, data = wavfile.read(list_wav[i]) #bug in here
            # print("End")
            data = np.array(data, dtype=np.float32)
            data *= 1./32768
            # feature = librosa.feature.melspectrogram(y=data, sr=rate, n_fft=2048, hop_length=512, power=2.0)
            feature = librosa.feature.mfcc(y=data, sr=rate, 
                                           n_mfcc=40, fmin=0, fmax=8000,
                                           n_fft=int(16*64), hop_length=int(16*32), power=2.0)
            feature = np.resize(feature, self.dim)
            category_label =  to_categorical(list_label[i], num_classes= len(self.list_labels) )
            X.append(feature)
            Y.append(category_label)
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=int)
        return X, Y



# train_generator = BreathDataGenerator(
#         'D:/Do An/breath-deep/data/datawav_filter/train',
#         list_labels=LIST_LABELS,
#         batch_size=BATCH_SIZE,
#         dim=INPUT_SIZE,
#         shuffle=False)        

# X, Y = train_generator.__getitem__(3)


# rate, data = wavfile.read("D:/Do An/breath-deep/data/datawav_filter/deep/01_male_23_BQuyen_1230_1270.wav")
# print(data)