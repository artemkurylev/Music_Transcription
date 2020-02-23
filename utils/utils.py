import numpy as np
import argparse
from scipy import signal
from midiutil.MidiFile import MIDIFile
import matplotlib.pyplot as plt
import soundfile
import librosa
import csv
import time
import h5py
import pickle
import os
from sklearn import preprocessing
from scipy.io import wavfile
import pandas as pd
from sklearn import metrics
from librosa import display


class DataProcessor:

    def __init__(self, path):
        self.path = path
        try:
            self.txt_files = [file for file in os.listdir(path) if file.endswith('.txt')]
            self.audio_files = [file for file in os.listdir(path) if file.endswith('.wav')]
            # self.midi_files = [file for file in os.listdir(path) if file.endswith('.midi')]

        except FileNotFoundError:
            print('Please enter correct Path - some of your files not found')
            return
        self.annotations = [pd.read_csv(os.path.join(self.path, filename),sep='\t') for filename in self.txt_files]
        self.audios = [librosa.load(os.path.join(self.path, filename)) for filename in self.audio_files]
        # self.annotations = [pd.read_csv(filename) for filename in txt_files]

    def generate_cqt_images(self, output_path):
        for i in range(len(self.audios)):
            chroma_cqt = librosa.feature.chroma_cqt(self.audios[i][0], self.audios[i][1])
            plt.figure(figsize=(15, 20))
            plt.subplot(2, 1, 2)
            display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
            plt.title('chroma_cqt')
            plt.colorbar()
            plt.savefig(os.path.join(output_path, self.txt_files[i][:3] + 'png'), bbox_inches='tight')

    def get_data(self):
        # ToDo Get X, Y

        return self.audios, self.annotations
