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
        self.chroma_cqts = []
        self.cqts = []
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

    def generate_chroma_cqt_images(self, output_path):
        for i in range(len(self.audios)):
            chroma_cqt = librosa.feature.chroma_cqt(self.audios[i][0], self.audios[i][1])
            self.chroma_cqts.append(chroma_cqt)
            plt.figure(figsize=(15, 20))
            plt.subplot(2, 1, 2)
            display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
            plt.title('chroma_cqt')
            plt.colorbar()
            plt.savefig(os.path.join(output_path, self.txt_files[i][:3] + 'png'), bbox_inches='tight')

    def generate_cqts(self):
        for i in range(len(self.audios)):
            chroma_cqt = np.abs(librosa.cqt(self.audios[i][0], self.audios[i][1]))
            self.cqts.append(chroma_cqt)

    @staticmethod
    def one_hot(series):
        res = np.zeros(128)
        list_values = series.values.tolist()
        for val in list_values:
            if type(val) == int:
                try:
                    res[val] = 1
                except Exception:
                    print('-0000-0-0-0-')
                    print(val)
                    print(list_values)

        return res.tolist()

    def get_cqt_data(self, freq):
        X = []
        y = []
        shapes = []
        for chroma in self.cqts:
            shapes.append(chroma.shape[1])
            X.extend(chroma.reshape(chroma.shape[1], chroma.shape[0]))

        for i in range(len(shapes)):
            new_freq = shapes[i]
            self.annotations[i]['frame'] = pd.cut(self.annotations[i].OnsetTime, new_freq)
            grouped_annotation = self.annotations[i][['MidiPitch', 'frame']].groupby(by='frame', as_index=False)\
                .agg(DataProcessor.one_hot)
            grouped_annotation['FixedMidiPitch'] = grouped_annotation['MidiPitch'].apply(
                lambda x: [0.0] * 128 if x is None else x)
            y.extend(grouped_annotation['FixedMidiPitch'])
        return X, y

