# This code makes a .npy dataset from sorted data
import numpy as np
import librosa
import os
import csv
from tqdm import tqdm

BUILD_DATA = True

SAMPLE_RATE = 16000 
DATADIR = os.path.join("AudioSorted")
n_mfcc = 20

class AudioData():
    SPEECH = "speech"
    MUSIC = "music"
    NOISE = "noise"
    
    LABELS = {SPEECH: 0, MUSIC: 1, NOISE: 2}
    
    training_data = []
    speech_count = 0
    music_count = 0
    noise_count = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            for f in tqdm(os.listdir(DATADIR + os.sep + label)):
                path = os.path.join(DATADIR, label, f)
                signal, sr = librosa.load(path, sr = SAMPLE_RATE)
                
                # MFCCs (Mel Frequency Cepstral Coefficents) are the standart tool for audio classification
                mfcc = librosa.feature.mfcc(signal,
                                            sr = sr,
                                            n_mfcc = n_mfcc
                                            )

                # np.eye(3) makes the label one-hot encoded
                self.training_data.append([np.array(mfcc), np.eye(3)[self.LABELS[label]]])

                if label == self.SPEECH:
                    self.speech_count += 1
                elif label == self.MUSIC:
                    self.music_count += 1
                elif label == self.NOISE:
                    self.noise_count += 1
        
        # Shuffling and saving the data
        np.random.shuffle(self.training_data)
        np.save("audiodata.npy", self.training_data)
        # Checking the data balance
        print("SPEECH: ", self.speech_count)
        print("MUSIC: ", self.music_count)
        print("NOISE: ", self.noise_count)
    
if BUILD_DATA:
    audiodata = AudioData()
    audiodata.make_training_data()
