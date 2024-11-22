import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import scipy.signal as sps
from scipy.io import wavfile
import math
import queue
import threading
from threading import Event
import sys
from tflite_runtime.interpreter import Interpreter
    
# Tạo hàng đợi
the_queue = queue.Queue()

# Thông số xử lý âm thanh
word_threshold = 0.71  # ngưỡng xác suất
rec_duration = 0.1  # giây
sample_rate = 48000  # min sample rate là 44.1khz
resample_rate = 8000
num_channels = 1
sample_length = 1.0
window_slide = np.zeros(int(rec_duration * resample_rate) * int(sample_length/rec_duration))
count = 0
old_index = 5
num_mfcc = 16
TimeCallback = 100

#Chuẩn hóa âm thanh
def normalizeAudio(audio_data):
    max_amplitude = np.max(np.abs(audio_data))
    normalized_audio_data = np.array([(audio_data / max_amplitude) * 32767], np.int16)
    return normalized_audio_data

# decimate (lọc và giảm mẫu)
def decimate(signal, old_fs, new_fs):
    if new_fs > old_fs:
        print('Lỗi: Tần số mẫu đích cao hơn ban đầu')
        return signal, old_fs
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print('Lỗi: Chỉ giảm mẫu theo hệ số nguyên')
        return signal, old_fs
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    return resampled_signal, new_fs

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

# Đọc file âm thanh cắt sẵn và tính FFT
file_1_path = 're1.wav'
sample_1_fft = CalculateFFT(file_1_path)
file_2_path = 're2.wav'
sample_2_fft = CalculateFFT(file_2_path)
file_3_path = 're3.wav'
sample_3_fft = CalculateFFT(file_3_path)
file_4_path = 're4.wav'
sample_4_fft = CalculateFFT(file_4_path)
file_5_path = 're5.wav'
sample_5_fft = CalculateFFT(file_5_path)

#Tính độ tương đồng cosine giữa 2 vector
def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(v1.shape[1]):
        x = v1[0][i]; y = v2[0][i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy / math.sqrt(sumxx * sumyy)

#Gọi hàm callback cho stream âm thanh
def sd_callback(rec, frames, time, status):
    global count, old_index
    # Bắt đầu tính thời gian
    start = timeit.default_timer()
    
    if status:
        print('Error:', status)

    # Giảm mẫu và chuẩn hóa
    rec = np.squeeze(rec)
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    rec = normalizeAudio(rec)

    # Cập nhật cửa sổ trượt
    window_slide[:int(len(window_slide)/10*9)] = window_slide[int(len(window_slide)/10):]
    window_slide[int(len(window_slide)/10*9):] = rec

    # Tính FFT cho đoạn âm thanh nhận được
    rec_fft = np.abs(np.fft.fft(window_slide))
    rec_fft = np.array(rec_fft)
    rec_fft = rec_fft.reshape(1, -1)
    # So sánh với các mẫu đã lưu
    if count == old_index:
        count = count + 1
        if count == 5:
            count = 0
    
    similarity = 0
    if count == 0:
        similarity = cosine_similarity(rec_fft, sample_1_fft) * 100
    elif count == 1:
        similarity = cosine_similarity(rec_fft, sample_2_fft) * 100
    elif count == 2:
        similarity = cosine_similarity(rec_fft, sample_3_fft) * 100
    elif count == 3:
        similarity = cosine_similarity(rec_fft, sample_4_fft) * 100
    elif count == 4:
        similarity = cosine_similarity(rec_fft, sample_5_fft) * 100
    print(f"Similarity: {similarity}")
    # Kiểm tra ngưỡng xác suất
    if similarity >= word_threshold * 100:
        old_index = count
        print(f'Phát hiện đoạn nhạc {count+1}, tỷ lệ: {similarity:.2f}%')
        result_text = f'{count+1}: {similarity:.2f}%'
        the_queue.put(result_text)
        
    count = count + 1
    if count == 5:
        count = 0
    
    # Debug thời gian xử lý
    #print("Time:", timeit.default_timer() - start)

# Chạy stream âm thanh
class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()

    def run(self):
        with sd.InputStream(channels=num_channels,
                            samplerate=sample_rate,
                            blocksize=int(sample_rate * rec_duration),
                            callback=sd_callback):
            while True:
                if self.stop_event.is_set():
                    break
                pass

myThread = MyThread()
myThread.start()

# Dừng thread sau khi hoàn thành xử lý
try:
    while True:
        message = the_queue.get()
        print('Keyword:', message)
except KeyboardInterrupt:
    myThread.stop_event.set()
