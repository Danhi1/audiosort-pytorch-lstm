# The dataset consists of noise, speech and music (split into training and testing sets)
# Our goal is to correctly label the testing set
# The dataset can be downloaded here:
# https://www.kaggle.com/c/silero-audio-classifier/data
import os
import csv
from tqdm import tqdm

# File labels are located in "train.csv" file, files themselves are split into 16 folders
# First we need to sort them into folders according to their labels
SORT_THE_FILES = True

if SORT_THE_FILES:
    traincsv = os.path.join("Audio", "silero-audio-classifier", "train.csv")
    DATADIR = os.path.join("Audio", "silero-audio-classifier", "train")
    SORTEDDIR = os.path.join("AudioSorted" + os.sep)



    input_file = csv.DictReader(open(traincsv))
    for row in tqdm(input_file):
        path = os.path.join(DATADIR, row["wav_path"].replace("/", os.sep))
        command = 'copy' + ' ' + path + ' ' + SORTEDDIR + row["label"] + os.sep + row[""] + ".wav"
        os.popen(command)
        
        
        
