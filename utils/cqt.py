import os
import librosa
import numpy as np


def audio_CQT(file_num, start, dur):  # start and dur in seconds

    # Load audio and define paths
    path = 'GuitarSet_audio_and_annotation/GuitarSet/audio/audio_hex-pickup_debleeded'
    audio_file = os.listdir(path)
    audio_path = os.path.join(path, audio_file[file_num])

    # Function for removing noise
    def cqt_lim(CQT):
        new_CQT = np.copy(CQT)
        new_CQT[new_CQT < -60] = -120
        return new_CQT

    # Perform the Constant-Q Transform
    data, sr = librosa.load(audio_path, sr=None, mono=True, offset=start, duration = dur)
    CQT = librosa.cqt(data, sr = 44100, hop_length = 1024, fmin=None, n_bins = 96, bins_per_octave = 12)
    CQT_mag = librosa.magphase(CQT)[0]**4
    CQTdB = librosa.core.amplitude_to_db(CQT_mag, ref = np.amax)
    new_CQT = cqt_lim(CQTdB)

