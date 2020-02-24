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
from utils.utils import DataProcessor
from utils import constant
from models.baseline import Baseline
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch. utils.data import DataLoader


if __name__ == '__main__':
    # Read and process data
    path = constant.PATH_AK_ISOL_CH
    dp = DataProcessor(constant.PATH_AK_ISOL_CH)
    dp.load_audios(sr=16000)
    dp.load_annotations()
    dp.generate_cqts(bins_per_octave=36)

    # Split Data
    X, y = dp.get_cqt_data(None)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.from_numpy(np.array(X_train))
    X_test = torch.from_numpy(np.array(X_test))

    # DataLoaders
    temp_train = []
    temp_test = []
    for i in y_train:
        temp_train.append(i)
    for i in y_test:
        temp_test.append(i)
    y_train = torch.tensor(temp_train)
    y_test = torch.tensor(temp_test)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    batch_size = 50

    # Data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size, shuffle=True)

    # Model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Baseline().to(device)

    # Train model
    optimizer = optim.Adam(model.parameters())
    epoch_num = 10
    log_interval = 50
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.reshape(data.shape[0], data.shape[1], 1)
            data = data.to(device)
            labels = np.argmax(target, axis=1)
            target = labels.clone().detach().squeeze().to(device)
            optimizer.zero_grad()
            output = model(data.float())
            # print(output)
            # print(target)

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


