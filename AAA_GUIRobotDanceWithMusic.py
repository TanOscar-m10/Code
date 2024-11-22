import serial
import serial.tools.list_ports
from tkinter import *
from tkinter import ttk
import tkinter as tk
import time
from tkinter import font as tkFont
import os
from PIL import Image, ImageTk

import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
from tflite_runtime.interpreter import Interpreter

import queue
import threading
from threading import Event
import sys
import scipy.signal as sps
from scipy.io import wavfile
import math

the_queue = queue.Queue()

if os.path.exists('/dev/rfcomm5') == False:
    path = 'sudo rfcomm bind 5 00:22:04:01:1F:E5'
    os.system(path)
    time.sleep(1)
    
try:
    bluetoothSerial = serial.Serial("/dev/rfcomm5", baudrate=9600)
    print('Connect to HC-05 successfully!')
except:
    print('Connect to HC-05 failed!')

# parameters
debug_time = 0
debug_acc = 1
word_threshold = 0.71 # threshold probality: < 0.5 -> not stop and reverse
rec_duration = 0.1 # seconds
window_stride = 0.5 #seconds
sample_rate = 48000 # min sample rate is 44.1khz
resample_rate = 8000
num_channels = 1
num_mfcc = 16
sample_length = 1.0
window_slide = np.zeros(int(rec_duration * resample_rate) * int(sample_length/rec_duration))
count = 0
old_index = 5
TimeCallback = 100

def normalizeAudio(audio_data):
    max_amplitude = np.max(np.abs(audio_data))
    normalized_audio_data = np.array([(audio_data / max_amplitude) * 32767], np.int16)
    return normalized_audio_data

# decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # check to make sure we're downsampling
    if new_fs > old_fs:
        print('Error: target sample rate higher than original')
        return signal, old_fs
    # we can only downsample by an interger factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print('Error: only downsample by an interger factor')
        return signal, old_fs
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    return resampled_signal, new_fs

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

file_1_path = 'cut_01.wav'
sample_1_fft = CalculateFFT(file_1_path)
file_2_path = 'cut_02.wav'
sample_2_fft = CalculateFFT(file_2_path)
file_3_path = 'cut_03.wav'
sample_3_fft = CalculateFFT(file_3_path)
file_4_path = 'cut_04.wav'
sample_4_fft = CalculateFFT(file_4_path)
file_5_path = 'cut_05.wav'
sample_5_fft = CalculateFFT(file_5_path)

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(v1.shape[1]):
        x = v1[0][i]; y = v2[0][i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def sd_callback(rec, frames, time, status):
    global count
    global old_index
    # start time for testing
    start = timeit.default_timer()
    # notify if error:
    if status:
        print('Error:', status)
    # remove 2nd dimesion from recording sample
    rec = np.squeeze(rec)
    # resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    rec = normalizeAudio(rec)
#     window_slide[:len(window_slide)//2] = window_slide[len(window_slide)//2:]
#     window_slide[len(window_slide)//2:] = rec
    window_slide[:int(len(window_slide)/10*9)] = window_slide[int(len(window_slide)/10):]
    window_slide[int(len(window_slide)/10*9):] = rec
    rec_fft = np.abs(np.fft.fft(window_slide))
    rec_fft = np.array(rec_fft)
    rec_fft = rec_fft.reshape(1, -1)
    
    if count == old_index:
        count = count + 1
        if count == 5:
            count = 0
    if count == 0:
        si = cosine_similarity(rec_fft, sample_1_fft) * 100
    elif count == 1:
        si = cosine_similarity(rec_fft, sample_2_fft) * 100
    elif count == 2:
        si = cosine_similarity(rec_fft, sample_3_fft) * 100
    elif count == 3:
        si = cosine_similarity(rec_fft, sample_4_fft) * 100
    elif count == 4:
        si = cosine_similarity(rec_fft, sample_5_fft) * 100
    if si >= word_threshold * 100:
        old_index = count
        print('Phat hien doan nhac', count+1, 'ty le:%.2f'%si)
        result_text = str(count+1) + ': ' + str('%.2f'%si) + '%'
        the_queue.put(result_text)
    count = count + 1
    if count == 5:
        count = 0
    
    if debug_acc:
        pass
        #print(val)
    if debug_time:
        print(timeit.default_timer() - start)

def after_callback():
    global event_id
    try:
        message = the_queue.get(block=False)
    except queue.Empty:
        event_id = window.after(TimeCallback, after_callback)
        return
#     print('Keyword:', message)
    KeywordLabel['text'] = message;
    mess = message.split(':')
    if mess[0] == '1':
        bluetoothSerial.write(b'1')
    elif mess[0] == '2':
        bluetoothSerial.write(b'2')
    elif mess[0] == '3':
        bluetoothSerial.write(b'3')
    elif mess[0] == '4':
        bluetoothSerial.write(b'4')
    elif mess[0] == '5':
        bluetoothSerial.write(b'5')
    event_id = window.after(TimeCallback, after_callback)
    
def CmdForward(event):
    bluetoothSerial.write(b'f')

def CmdLeft(event):
    bluetoothSerial.write(b'l')
    
def CmdStop(event):
    bluetoothSerial.write(b's')
    
def CmdRight(event):
    bluetoothSerial.write(b'r')
    
def CmdBackward(event):
    bluetoothSerial.write(b'b')
    
def CmdDanceMove01(event):
    bluetoothSerial.write(b'1')
    
def CmdDanceMove02(event):
    bluetoothSerial.write(b'2')

def CmdDanceMove03(event):
    bluetoothSerial.write(b'3')
    
def CmdDanceMove04(event):
    bluetoothSerial.write(b'4')
    
def CmdExit():
    global event_id
    window.after_cancel(event_id)
    window.destroy()
    myThread.stop_event.set()
    
def resize_image(image_path, width, height):
    image = Image.open(image_path)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)

# creat window
window = tk.Tk()
window.title('Hexapod Robot Dance With Music GUI')
scr_width, scr_height = window.winfo_screenwidth(), window.winfo_screenheight()
scr_height = int (scr_height*9/10) 
window.geometry('%dx%d+0+0' % (scr_width,scr_height))
window.resizable(False, False)

# devide window into frames
InfoFrame_width = scr_width - 10
InfoFrame_height = int(scr_height / 6 - 10)
InfoFrame = tk.Frame(window, width = InfoFrame_width, height = InfoFrame_height)
InfoFrame.pack(side = 'top', fill = 'both')

ManualFrame_width = int(scr_width * 7 / 12 - 10)
ManualFrame_height = int(scr_height * 5/6 - 10)
ManualFrame = tk.Frame(window, width = ManualFrame_width, height = ManualFrame_height)
ManualFrame.pack(side = 'left', fill = 'both')

AutoFrame_width = int(scr_width * 5 / 12 - 10)
AutoFrame_height = int(scr_height * 5/6 - 10)
AutoFrame = tk.Frame(window, width = AutoFrame_width, height = AutoFrame_height)
AutoFrame.pack(side = 'right', fill = 'both')

# load images
TLU_icon_width = int (InfoFrame_width / 10)
TLU_icon_height = InfoFrame_height - 10
TLU_icon = resize_image(image_path= './figs/TLU_logo.png',
                        width = TLU_icon_width, height = TLU_icon_height)

KhoaCK_icon_width = int (InfoFrame_width / 10)
KhoaCK_icon_height = InfoFrame_height - 10
KhoaCK_icon = resize_image(image_path= './figs/KhoaCK_logo.png',
                           width = KhoaCK_icon_width, height = KhoaCK_icon_height)

Forward_icon_width = int(ManualFrame_width / 5 - 10)
Forward_icon_height = int(ManualFrame_height * 2 / 5 - 10)
Forward_icon = resize_image(image_path= './figs/a02.png',
                            width = Forward_icon_width, height = int(scr_height/3-10))

Left_icon_width = int(ManualFrame_width * 2 / 5 - 10)
Left_icon_height = int(ManualFrame_height / 5 - 10)
Left_icon = resize_image(image_path= './figs/a04.png',
                         width = Left_icon_width, height = Left_icon_height)

Stop_icon_width = int(ManualFrame_width / 5 - 10)
Stop_icon_height = int(ManualFrame_height / 5 - 10)
Stop_icon = resize_image(image_path= './figs/a05.png',
                         width = Stop_icon_width, height = Stop_icon_height)

Right_icon_width = int(ManualFrame_width * 2 / 5 - 10)
Right_icon_height = int(ManualFrame_height / 5 - 10)
Right_icon = resize_image(image_path= './figs/a06.png',
                          width = Right_icon_width, height = Right_icon_height)

Backward_icon_width = int(ManualFrame_width / 5 - 10)
Backward_icon_height = int(ManualFrame_height * 2 / 5 - 10)
Backward_icon = resize_image(image_path= './figs/a08.png',
                             width = Backward_icon_width, height = Backward_icon_height)

DanceMove01_icon_width = int(ManualFrame_width * 2 / 5 - 10)
DanceMove01_icon_height = int(ManualFrame_height / 5 - 10)
DanceMove01_icon = resize_image(image_path= './figs/a16.png',
                                width = DanceMove01_icon_width, height = DanceMove01_icon_height)

DanceMove02_icon_width = int(ManualFrame_width * 2 / 5 - 10)
DanceMove02_icon_height = int(ManualFrame_height / 5 - 10)
DanceMove02_icon = resize_image(image_path= './figs/a17.png',
                                width = DanceMove02_icon_width, height = DanceMove02_icon_height)

DanceMove03_icon_width = int(ManualFrame_width * 2 / 5 - 10)
DanceMove03_icon_height = int(ManualFrame_height / 5 - 10)
DanceMove03_icon = resize_image(image_path= './figs/a18.png',
                                width = DanceMove03_icon_width, height = DanceMove03_icon_height)

DanceMove04_icon_width = int(ManualFrame_width * 2 / 5 - 10)
DanceMove04_icon_height = int(ManualFrame_height / 5 - 10)
DanceMove04_icon = resize_image(image_path= './figs/a19.png',
                                width = DanceMove04_icon_width, height = DanceMove04_icon_height)

Exit_icon_width = int(AutoFrame_width / 2 - 10)
Exit_icon_height = int(AutoFrame_height / 5 - 10)
Exit_icon = resize_image(image_path= './figs/a13.png',
                         width = Exit_icon_width, height = Exit_icon_height)

# create buttons
TLU_logo = ttk.Label(InfoFrame, image = TLU_icon)
TLU_logo.grid(row=0, column=0, padx = 25, pady = 5)

KhoaCK_logo = ttk.Label(InfoFrame, image = KhoaCK_icon)
KhoaCK_logo.grid(row=0, column=1, padx = 25, pady = 5)

Intro_Text = 'ThuyLoi University\nFaculty of Mechanical Engineering\nDepartment of Mechatronic Engineering'
Intro_Label = ttk.Label(InfoFrame, text = Intro_Text,
                        font=('Arial', 40, 'bold'), foreground = 'blue')
Intro_Label.grid(row=0, column=2, padx = 25, pady = 5)

Btn_Forward = ttk.Button(ManualFrame, image = Forward_icon)
Btn_Forward.grid(row=0, column=1)

Btn_Left = ttk.Button(ManualFrame, image = Left_icon)
Btn_Left.grid(row=1, column=0)

Btn_Stop = ttk.Button(ManualFrame, image = Stop_icon)
Btn_Stop.grid(row=1, column=1)

Btn_Right = ttk.Button(ManualFrame, image = Right_icon)
Btn_Right.grid(row=1, column=2)

Btn_Backward = ttk.Button(ManualFrame, image = Backward_icon)
Btn_Backward.grid(row=2, column=1)

Btn_Dance01 = ttk.Button(ManualFrame, image = DanceMove01_icon)
Btn_Dance01.grid(row=0, column=0)

Btn_Dance02 = ttk.Button(ManualFrame, image = DanceMove02_icon)
Btn_Dance02.grid(row=0, column=2)

Btn_Dance03 = ttk.Button(ManualFrame, image = DanceMove03_icon)
Btn_Dance03.grid(row=2, column=0)

Btn_Dance04 = ttk.Button(ManualFrame, image = DanceMove04_icon)
Btn_Dance04.grid(row=2, column=2)

Keyword_Intro_width = AutoFrame_width - 10
Keyword_Intro_text1 = 'Part of Lyric detect from\n'
Keyword_Intro_text2 = 'song \"Bat tinh yeu len\"\n'
Keyword_Intro_text3 = 'Singer: Hoa Minzy & Tang Duy Tan'
Keyword_Intro_text = Keyword_Intro_text1 + Keyword_Intro_text2 + Keyword_Intro_text3
Keyword_Intro = ttk.Label(AutoFrame, text = Keyword_Intro_text,
                          font=('Arial', 35, 'bold'))
Keyword_Intro.grid(row=0, column=0, padx = 5, pady = 5)

KeywordLabel_width = AutoFrame_width - 10
KeywordLabel = ttk.Label(AutoFrame, font=('Arial', 50, 'bold'),
                         background = 'white', text = 'none')
KeywordLabel.grid(row=1, column=0, padx = 5, pady = 5)

Help_text1 = 'Help:\n'
Help_text2 = '+) Part 1: \"Rot mat vao tim em\"\n'
Help_text3 = '+) Part 2: \"235 anh dang roi\"\n'
Help_text4 = '+) Part 3: \"Muon noi voi em anh e ngai\"\n'
Help_text5 = '+) Part 4: \"Sao tim khong nghe loi\"\n'
Help_text6 = '+) Part 5: \"Chi can thuc giac\"'
KeywordHelp_text = Help_text1 + Help_text2 + Help_text3 + Help_text4 + Help_text5 + Help_text6
KeywordHelp_width = AutoFrame_width - 10
KeywordHelp = ttk.Label(AutoFrame, font=('Arial', 30, 'italic'),
                        text = KeywordHelp_text)
KeywordHelp.grid(row=2, column=0, padx = 5, pady = 5)

Btn_Exit = ttk.Button(AutoFrame, image = Exit_icon, command = CmdExit)
Btn_Exit.grid(row=3, column=0, padx = 5, pady = 5)

# handle press/release events
Btn_Forward.bind("<ButtonPress>", CmdForward)
Btn_Forward.bind("<ButtonRelease>", CmdStop)

Btn_Left.bind("<ButtonPress>", CmdLeft)
Btn_Left.bind("<ButtonRelease>", CmdStop)

Btn_Stop.bind("<ButtonPress>", CmdStop)

Btn_Right.bind("<ButtonPress>", CmdRight)
Btn_Right.bind("<ButtonRelease>", CmdStop)

Btn_Backward.bind("<ButtonPress>", CmdBackward)
Btn_Backward.bind("<ButtonRelease>", CmdStop)

Btn_Dance01.bind("<ButtonPress>", CmdDanceMove01)
Btn_Dance02.bind("<ButtonPress>", CmdDanceMove02)
Btn_Dance03.bind("<ButtonPress>", CmdDanceMove03)
Btn_Dance04.bind("<ButtonPress>", CmdDanceMove04)

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_event = threading.Event()
    def run(self):
        with sd.InputStream(channels = num_channels,
                            samplerate = sample_rate,
                            blocksize = int(sample_rate * rec_duration),
                            callback = sd_callback):
            while True:
                if self.stop_event.is_set():
                    break
                pass

myThread = MyThread()
myThread.start()
event_id = None 
event_id = window.after(TimeCallback, after_callback)
window.mainloop()
