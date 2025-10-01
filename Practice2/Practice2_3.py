import io
import os

import matplotlib.pyplot as plt
import torch
from IPython.display import Audio
import torchaudio
from pydub import AudioSegment

import torchaudio.functional as F
import torchaudio.transforms as T

import librosa


def plot_waveform(waveform, samplerate, title="Waveform", ax=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

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


def plot_fbank(fbank, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    
    if title is not None:
        ax.set_title(title)
        
    ax.imshow(fbank, aspect="auto")
    ax.set_ylabel("frequency bin")
    ax.set_xlabel("mel bin")

def plot_pitch(waveform, sample_rate,  pitch, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_title("Pitch Feature")
    ax.grid(True)
    
    if title is not None:
        ax.set_title(title)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    ax.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.5)

    ax2 = ax.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ax2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    ax2.legend(loc=0)



for i in ["audio_dataset\\095522039","audio_dataset\\095522040","audio_dataset\\095522041","audio_dataset\\095522042"]:
    if not os.path.isfile(i + ".wav"):
        tmp = AudioSegment.from_file(i + ".m4a", format="m4a")
        tmp.export(i + ".wav", format="wav")
    waveform, sample_rate = torchaudio.load(i + ".wav")
    
    n_fft = 1024
    
    spectrogram = T.Spectrogram(n_fft=n_fft)
    spec = spectrogram(waveform)

    #reconstruct waveform from spectrogram
    griffin_lim = T.GriffinLim(n_fft=n_fft)

    recon_waveform = griffin_lim(spec)

    #Mel Filter Bank
    n_mels = 64

    mel_filters = F.melscale_fbanks(int(n_fft // 2 + 1), n_mels=n_mels,
                                    f_min=0.0,f_max=sample_rate / 2.0,
                                    sample_rate=sample_rate,norm="slaney")

    #MelSpectrogram
    win_length = None
    hop_length = 512
    
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,win_length=win_length,
                                       hop_length=hop_length,center=True, pad_mode="reflect",
                                       power=2.0,norm="slaney",n_mels=n_mels,mel_scale="htk")

    melspect = mel_spectrogram(waveform)

    #MFCC
    n_mfcc = n_mels

    mfcc_transform = T.MFCC(sample_rate=sample_rate,
                            n_mfcc=n_mfcc,
                            melkwargs={
                                "n_fft": n_fft,
                                "n_mels": n_mels,
                                "hop_length": hop_length,
                                "mel_scale": "htk"
                            })

    mfcc = mfcc_transform(waveform)

    #LFCC
    n_lfcc = n_mels // 4

    lfcc_transform = T.LFCC(sample_rate=sample_rate,
                            n_lfcc=n_lfcc,
                            speckwargs={
                                "n_fft": n_fft,
                                "win_length": win_length,
                                "hop_length": hop_length
                                })

    lfcc = lfcc_transform(waveform)

    #Get Pitch
    pitch = F.detect_pitch_frequency(waveform, sample_rate)
    
    #Out
    fig, axs = plt.subplots(3, 2)

    axs[1][0].sharex(axs[0][0])
    axs[1][0].sharey(axs[0][0])
    
    plot_waveform(recon_waveform,samplerate=sample_rate, title=i + ":Reconstructed Waveform", ax=axs[1][0])
    plot_fbank(mel_filters, "Mel Filter Bank", ax=axs[2][0])
    plot_spectrogram(melspect[0], title="MelSpectrogram", ax=axs[0][1], ylabel="mel freq")
    plot_spectrogram(mfcc[0], title="MFCC", ax=axs[1][1])
    plot_spectrogram(lfcc[0], title="LFCC", ax=axs[2][1])
    plot_pitch(waveform, sample_rate, pitch,i + ":Original Waveform" , axs[0][0])
    fig.tight_layout()

    
plt.show()
