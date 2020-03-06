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


def recursive_file_search(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    folders = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isdir(os.path.join(folder, file))]
    if folders:
        for next_folder in folders:
            files.extend(recursive_file_search(next_folder))

    return files


class DataProcessor:

    def __init__(self, path):
        self.path = path
        self.chroma_cqts = []
        self.cqts = []
        self.annotations = None
        self.audios = None
        self.files = recursive_file_search(self.path)
        try:
            self.txt_files = [file for file in self.files if file.endswith('.txt')]
            self.audio_files = [file for file in self.files if file.endswith('.wav')]

        except FileNotFoundError:
            print('Please enter correct Path - some of your files not found')
            return

    def load_annotations(self):
        self.annotations = [pd.read_csv(filename, sep='\t') for filename in self.txt_files]

    def load_audios(self, sr=None):
        if sr is not None:
            self.audios = [librosa.load(filename, sr) for filename in self.audio_files]
        else:
            self.audios = [librosa.load(filename) for filename in self.audio_files]

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

    def generate_cqts(self, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12,
                      tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect'):

        for i in range(len(self.audios)):
            chroma_cqt = np.abs(librosa.cqt(self.audios[i][0], self.audios[i][1],  hop_length, fmin, n_bins,
                                bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode))
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

    @staticmethod
    def simple_one_hot(series):
        list_values = np.array(series.values.tolist())
        return list_values.argmax()

    def get_cqt_data(self, freq):
        X = []
        y = []
        shapes = []
        for chroma in self.cqts:
            shapes.append(chroma.shape[1])
            X.extend(chroma.reshape(chroma.shape[1], chroma.shape[0]))

        for i in range(len(shapes)):
            new_freq = shapes[i]
            try:
                self.annotations[i]['frame'] = pd.cut(self.annotations[i].OnsetTime, new_freq)
            except IndexError:
                print('Index out of range: ', i, ' > ', len(self.annotations))
            grouped_annotation = self.annotations[i][['MidiPitch', 'frame']].groupby(by='frame', as_index=False)\
                .agg(DataProcessor.one_hot)
            grouped_annotation['FixedMidiPitch'] = grouped_annotation['MidiPitch'].apply(
                lambda x: [0]*128 if x is None else x)
            y.extend(grouped_annotation['FixedMidiPitch'])
        return X, y

