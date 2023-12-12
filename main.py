import io
import random
import os
import re
import wave
import glob
import pyaudio
import subprocess
import numpy as np
import librosa as lr
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model

SR = 22050
FFT = 2048
LENGTH = 128
OVERLAP = 64

def filter_audio(audio):
    """Filter every audio file in raw data in several parameters"""
    # Calculate voice energy for every 123 ms block
    apower = lr.amplitude_to_db(np.abs(lr.stft(audio, n_fft=2048)), ref=np.max)
    # Summarize energy of every rate, normalize
    apsums = np.sum(apower, axis=0) ** 2
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)
    # Smooth the graph for saving short spaces and pauses, remove sharpness
    apsums = np.convolve(apsums, np.ones((9,)), 'same')
    # Normalize again
    apsums -= np.min(apsums)
    apsums /= np.max(apsums)
    # Set noise limit to 35% over voice
    apsums = np.array(apsums > 0.35, dtype=bool)
    # Extend the blocks every on 125ms
    # before separated samples (2048 at block)
    apsums = np.repeat(apsums, np.ceil(len(audio) / len(apsums)))[:len(audio)]
    return audio[apsums]

def prepare_audio(a_name, target=False):
    """Feature Extraction for further neuron model using"""
    #print('loading %s' % a_name) - скрыто
    audio, _ = lr.load(a_name, sr=SR)
    audio = filter_audio(audio)
    data = lr.stft(audio, n_fft=FFT).swapaxes(0, 1)
    samples = []
    for i in range(0, len(data) - LENGTH, OVERLAP):
        samples.append(np.abs(data[i:i + LENGTH]))

    samples = np.array(samples)
    if len(samples.shape) == 2:
        samples = np.expand_dims(samples, axis=0)

    results_shape = (samples.shape[0], 1)
    results = np.ones(results_shape) if target else np.zeros(results_shape)

    return samples, results

def create_model(list_of_voices, num_of_epoch=30):
    """Prepare raw data of input list, create, train and save the model"""
    # Unite all training samples
    X, Y = prepare_audio(list_of_voices[0][0], list_of_voices[0][1])
    for voice in list_of_voices[1:]:
        dx, dy = prepare_audio(voice[0], voice[1])
        X = np.vstack((X, dx))
        Y = np.concatenate((Y, dy), axis=0)
        del dx, dy
    # Shake all blocks randomly
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]
    # Create model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=X.shape[1:]))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation('hard_sigmoid'))
    # Compile and train model
    model.compile(Adam(learning_rate=0.004), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=num_of_epoch, batch_size=32, validation_split=0.2, verbose = 0)
    # Testing resulted model
    #print(model.evaluate(X, Y)) - скрыто
    # Save the model for next using
    model.save('model.hdf5')
    return None

def random_sentence(input_book, num_of_sentences=1):
    """Generate random text from book and return to list format"""
    with io.open(input_book, encoding='utf-8') as file:
        file = file.read().split('.')
    """text generation"""
    list_sentences = []
    for i in range(0, num_of_sentences):
        list_sentences.append(file.pop(random.randint(0, len(file) - 1)))
    text = ' '.join(list_sentences)
    return text

def voice_recorder(output_filename, seconds_of_audio):
    """Record a voice of target within PyAudio interface with device(7) - USB MICRO"""
    chunk = 2024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2  # Stereo
    fs = 44100  # Record at 44100 samples per second
    p = pyaudio.PyAudio()  # an interface to PortAudio
    print('Recording')
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    input_device_index=7,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []  # Initialize array to store frames
    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds_of_audio)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    print('stream stopped')
    stream.close()
    p.terminate()
    print('Finished recording')
    # Save the recorded data as a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return None

def find_wavs(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern"""
    return glob(os.path.join(directory, pattern), recursive=True)

def wav_reader(directory):
    """Find all wav files in directory and compose it in list of tuples with 'True' mark at target"""
    wav_list = find_wavs(directory)
    res_list = []
    for wav in wav_list:
        temp_list = [wav]
        if re.match(r'.*target1.*\.wav$', wav):
            temp_list.append(True)
        else:
            temp_list.append(False)
        res_list.append(tuple(temp_list))
    return res_list

def split_audio(file_name, path_to_save):
    """split a music track into specified sub-tracks by calling ffmpeg from the shell"""
    # create a template of the ffmpeg call in advance
    cmd_string = 'ffmpeg -y -i {tr} -acodec copy -ss {st} -to {en} {nm}.wav'
    timings = [25, 20, 15, 10]  # timings to split input file with
    start_pos = 0
    out_name_num = 11
    for t in timings:
        name = path_to_save + 'target' + str(out_name_num)
        command = cmd_string.format(tr=file_name, st=start_pos, en=start_pos+t, nm=name)
        start_pos += t
        out_name_num += 1
        # use subprocess to execute the command in the shell
        subprocess.call(command, shell=True)
    # delete prerecorded voice of target
    if os.path.exists("target.wav"):
        os.remove("target.wav")
    else:
        print("The file does not exist")
    return None

def check_access(target_path):
    """check access status of target with pre-trained model"""
    model = load_model('model.hdf5')
    # tokenize target audio
    new_audio, _ = prepare_audio(target_path)  # ignore labels
    # use model to predict
    prediction = model.predict_on_batch(new_audio)
    val_sum = 0
    for val in prediction:
        val_sum += val[0]
    print('%.3f' % (100 * (val_sum / len(prediction))) + '%')  # percent of target similarity to owner
    if (val_sum / len(prediction)) * 100 > 80.0:
        print('access is allowed')
        return True
    else:
        print('access is denied')
        return False
