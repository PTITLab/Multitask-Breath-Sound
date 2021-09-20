import os
import pandas as pd

from pydub import AudioSegment


from preprocess_data_constant import (
    DEV_INPUT_PATH,
    DEV_OUTPUT_PATH,
    DEV_LABEL_PATH,
    TEST_INPUT_PATH,
    TEST_OUTPUT_PATH,
    TEST_LABEL_PATH,
    CHUNK,
    OVERLAP,
    LABEL_FOLDER,
    BREATH_TYPE,
    TRAIN_OUTPUT_FOLDER,
    TEST_OUTPUT_FOLDER,

)



def check_directory(origin_path, folder):
    directory = os.path.join(origin_path, folder)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_audio_segment (filename, source_path, destination_path, start, end, status, chunk, overlap):
    """[summary]
    
    Arguments:
        filename {[type]} -- [filename of the audio file]
        source_path {[type]} -- [source of a audio file]
        destination_path {[type]} -- [destination of a output file]
        start {[type]} -- [description]
        end {[type]} -- [description]
        status {[type]} -- [type of breath]
        chunk {[type]} -- [length of the segment]
        overlap {[type]} -- [overlap % ex 0.5 == 50%]
    """
    
    # Pydub works in milliseconds
    start = start * 1000 
    end = end * 1000 
    
    # Get the audio file
    src_Audio = AudioSegment.from_wav(source_path)
    
    #Cut the right part
    output_audio = src_Audio[start:end]
    
    #split it into 5s each and then export 
    lenInSec = ( len(output_audio)/(chunk*1000)).__round__() #split each path by chunk second
    
    start = 0
    metaInfo=[["",""]]
    for i in range(0, lenInSec):
        audio_partition = output_audio[start * 1000: (start + chunk) * 1000]
        fname =  filename + str(i) + "-" + status + '.wav'
        # print(destination_path)
        audio_partition.export(destination_path + '/' + fname, 
                         format="wav")  # Exports to a wav file in the current path.
        start = start + start*overlap
        # start += chunk #None overlap 


def split_by_label(source_path, destination_path, label_path, output_folder):

    # Check the output directory status 

    check_directory(destination_path, output_folder)

    output_path = os.path.join(destination_path, output_folder) + '/'
    # Get all the audio files
    filenames = os.listdir(source_path)
    
    meta_data=[["",""]]
    
    # Go through all the file 
    for filename in filenames:
        
        # take the file name without dot
        filename =  filename.split(".")[0]
        
        #get wav file name path
        wav_path = source_path + filename + ".wav"
        
        #get label filename path
        csv_path = label_path + filename + ".txt"
        
        #read the label file path 
        label = pd.read_csv(csv_path, delim_whitespace= True)
        
        #Normal breath
        for breath in BREATH_TYPE:
            if (breath in label['Status'].values):
                breath_start = label[label['Status'] == breath]['Start'].iloc[0]
                breath_end   = label[label['Status'] == breath]['End'].iloc[-1]
                output_by_label = os.path.join(output_path, breath)               
                if not os.path.exists(output_by_label):
                    os.makedirs(output_by_label)
                #Export the file
                # print(output_by_label)
                get_audio_segment(filename, wav_path, output_by_label, breath_start, breath_end, breath, CHUNK, OVERLAP)


# Devset
split_by_label(DEV_INPUT_PATH, DEV_OUTPUT_PATH, DEV_LABEL_PATH, TRAIN_OUTPUT_FOLDER)
split_by_label(DEV_INPUT_PATH, DEV_OUTPUT_PATH, DEV_LABEL_PATH, TEST_OUTPUT_FOLDER)

# # Testset
# split_by_label(TEST_INPUT_PATH, TEST_OUTPUT_PATH, TEST_LABEL_PATH)




