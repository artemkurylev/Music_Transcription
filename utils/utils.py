import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import pandas as pd
from librosa import display
from joblib import Parallel, delayed
from Repository.utils.CFP import PreProcessSong
import torch
import pickle


def recursive_file_search(folder):
    files = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    folders = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isdir(os.path.join(folder, file))]
    if folders:
        for next_folder in folders:
            files.extend(recursive_file_search(next_folder))

    return files


class DataProcessor:
    class DataReader:

        def __init__(self):
            self.audios = []
            self.annotations = []

        def read_audios(self, audio_file_paths):
            results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
                map(delayed(self.read_audio), audio_file_paths))
            return results

        def read_audio(self, path):
            audio = librosa.load(path, sr=16000)
            if audio:
                self.audios.append(audio)

    def __init__(self, path):
        self.song_datas = []
        self.path = path
        self.chroma_cqts = []
        self.cqts = []
        self.cfps = []
        self.ready_cfps = []
        self.ready_cqts = []
        self.annotations = None
        self.audios = None
        self.files = recursive_file_search(self.path)
        try:
            self.txt_files = [file for file in self.files if file.endswith('.txt')]
            self.audio_files = [file for file in self.files if file.endswith('.wav')]

        except FileNotFoundError:
            print('Please enter correct Path - some of your files not found')
            return

    def create_cfp(self, path):
        data = PreProcessSong(path)
        self.cfps.append(data)

    def process_audios(self, from_file=False):
        if from_file:
            try:
                with open('cfps.p', 'rb') as rf:
                    self.cfps = pickle.load(rf)
                    return
            except FileNotFoundError:
                pass
        results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
            map(delayed(self.create_cfp), self.audio_files))
        return results

    def load_annotations(self):
        self.annotations = [pd.read_csv(filename, sep='\t') for filename in self.txt_files]

    def load_audios(self, sr=None):
        if sr is not None:
            dr = DataProcessor.DataReader()
            dr.read_audios(self.audio_files)
            self.audios = dr.audios
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
            chroma_cqt = np.abs(librosa.cqt(self.audios[i][0], self.audios[i][1], hop_length, fmin, n_bins,
                                            bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale,
                                            pad_mode))
            self.cqts.append(chroma_cqt.reshape((chroma_cqt.shape[1],1,chroma_cqt.shape[0],1)))

    def generate_staff(self):
        for i in self.audio_files:
            self.song_datas.append(PreProcessSong(i))

    @staticmethod
    def one_hot(series):
        res = np.zeros(88)
        list_values = series.values.tolist()
        for val in list_values:
            if type(val) == int:
                try:
                    res[val - 21] = 1
                except Exception:
                    print('-0000-0-0-0-')
                    print(val)
                    print(list_values)
        return res.tolist()

    @staticmethod
    def simple_one_hot(series):
        list_values = np.array(series.values.tolist())
        return list_values.argmax()
    
    
    def partial_cqt(self, data):
        data = DataProcessor.rolling_window(data, 5)
        self.ready_cqts.append(data)

    
    def partial_cqts(self, from_file=False):
        if from_file:
            try:
                with open('ready_cfps.p', 'rb') as rf:
                    print('File with cfps opened...')
                    self.ready_cfps = pickle.load(rf)
                    print('Ready to go;)')
                    return
            except FileNotFoundError:
                pass
        results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
            map(delayed(self.partial_cqt), self.cqts))
        return results

    def get_cqt_data(self, step=5, from_file=False, cnn=True):
        X = []
        y = []
        shapes = []
        self.partial_cqts()
        for chroma in self.ready_cqts:
            shapes.append(len(chroma))
            X.extend(chroma)

        for i in range(len(shapes)):
            new_freq = shapes[i]
            try:
                self.annotations[i]['frame'] = pd.cut(self.annotations[i].OnsetTime, new_freq)
            except IndexError:
                print('Index out of range: ', i, ' > ', len(self.annotations))
            grouped_annotation = self.annotations[i][['MidiPitch', 'frame']].groupby(by='frame', as_index=False) \
                .agg(DataProcessor.one_hot)
            grouped_annotation['FixedMidiPitch'] = grouped_annotation['MidiPitch'].apply(
                lambda x: [0] * 88 if x is None else x)
            y.extend(grouped_annotation['FixedMidiPitch'])
        Xy = torch.tensor(X)
        y = pd.Series(y, name='FixedMidiPitch')
        return Xy, y

    def partial_cfp(self, data):
        data = DataProcessor.rolling_window(data, 5)
        self.ready_cfps.append(data)

    def partial_cfps(self, from_file=False):
        if from_file:
            try:
                with open('ready_cfps.p', 'rb') as rf:
                    print('File with cfps opened...')
                    self.ready_cfps = pickle.load(rf)
                    print('Ready to go;)')
                    return
            except FileNotFoundError:
                pass
        results = Parallel(n_jobs=-1, verbose=5, backend="threading")(
            map(delayed(self.partial_cfp), self.cfps))

        return results

    @staticmethod
    def rolling_window(a, window, step_size=0):
        shape = [a.shape[0] // window, window, a.shape[2], a.shape[3]]
        return np.lib.stride_tricks.as_strided(a, shape=shape)

    def save_cfps(self):
        with open('cfps.p', 'wb') as wf:
            pickle.dump(self.cfps, wf)

    def save_ready_cfps(self):
        with open('ready_cfps.p', 'wb') as wf:
            pickle.dump(self.ready_cfps, wf)

    def get_cfp_data(self, step=5, from_file=True, cnn=True):
        X = []
        y = []
        shapes = []
        new_chroma = []
        if cnn:
            self.partial_cfps(from_file)
            for chroma in self.ready_cfps:
                shapes.append(len(chroma))
                X.extend(chroma)
        else:
            for chroma in self.cfps:
                shapes.append(len(chroma))
                X.extend(chroma)
        print('X is ready')
        for i in range(len(shapes)):
            new_freq = shapes[i]
            try:
                self.annotations[i]['frame'] = pd.cut(self.annotations[i].OnsetTime, new_freq)
            except IndexError:
                print('Index out of range: ', i, ' > ', len(self.annotations))
            grouped_annotation = self.annotations[i][['MidiPitch', 'frame']].groupby(by='frame', as_index=False) \
                .agg(DataProcessor.one_hot)
            grouped_annotation['FixedMidiPitch'] = grouped_annotation['MidiPitch'].apply(
                lambda x: [0] * 88 if x is None else x)
            y.extend(grouped_annotation['FixedMidiPitch'])
        print('Finished!')
        Xy = torch.tensor(X)
        y = pd.Series(y, name='FixedMidiPitch')
        return Xy, y
