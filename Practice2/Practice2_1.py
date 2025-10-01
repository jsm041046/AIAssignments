import io
import os

import matplotlib.pyplot as plt
import torch
from IPython.display import Audio
import torchaudio
from pydub import AudioSegment

import torchaudio.transforms as T

import librosa


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
        
    if title is not None:
        ax.set_title(title)
        
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")



for i in ["audio_dataset\\095522039","audio_dataset\\095522040","audio_dataset\\095522041","audio_dataset\\095522042"]:
    if not os.path.isfile(i + ".wav"):
        tmp = AudioSegment.from_file(i + ".m4a", format="m4a")
        tmp.export(i + ".wav", format="wav")
    waveform, sample_rate = torchaudio.load(i + ".wav")

    spectrogram = T.Spectrogram(n_fft=512)

    spec = spectrogram(waveform)
    fig, axs = plt.subplots(2, 1)
    plot_waveform(waveform, sample_rate, title=i + " : Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
    fig.tight_layout()

    
plt.show()
