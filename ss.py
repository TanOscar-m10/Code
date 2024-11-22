import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import scipy.signal as sps
from scipy.io import wavfile
import math
import queue
import threading
from threading import Event
import sys

# Thông số xử lý âm thanh
word_threshold = 0.71  # ngưỡng xác suất
rec_duration = 0.1  # giây
sample_rate = 48000  # min sample rate là 44.1khz
resample_rate = 8000
num_channels = 1
sample_length = 1.0
window_slide = np.zeros(int(rec_duration * resample_rate) * int(sample_length/rec_duration))
num_mfcc = 16
TimeCallback = 100

#Chuẩn hóa âm thanh
def normalizeAudio(audio_data):
    max_amplitude = np.max(np.abs(audio_data))
    normalized_audio_data = np.array([(audio_data / max_amplitude) * 32767], np.int16)
    return normalized_audio_data

#Tính FFT từ file âm thanh
def CalculateFFT(filepath):
    old_sr, sample_data = wavfile.read(filepath)
    numSamples = round(len(sample_data) * float(resample_rate) / old_sr)
    sample_audio = sps.resample(sample_data, numSamples)
    if len(sample_audio) > resample_rate:
        sample_audio = sample_audio[:resample_rate]
    sample_audio = normalizeAudio(sample_audio)
    sample_audio_fft = np.abs(np.fft.fft(sample_audio))
    sample_audio_fft = np.array(sample_audio_fft)
    sample_audio_fft = sample_audio_fft.reshape(1, -1)
    return sample_audio_fft

#Tính độ tương đồng cosine giữa 2 vector
def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(v1.shape[1]):
        x = v1[0][i]; y = v2[0][i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy / math.sqrt(sumxx * sumyy)

# Đọc và tính FFT từ hai file WAV
file_1_path = 're5.wav'
file_2_path = 're21.wav'

fft_1 = CalculateFFT(file_1_path)
fft_2 = CalculateFFT(file_2_path)

# Tính độ tương đồng cosine
similarity = cosine_similarity(fft_1, fft_2) * 100
print(f"Độ tương đồng giữa hai file: {similarity:.2f}%")