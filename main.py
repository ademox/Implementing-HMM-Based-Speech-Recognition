import os
import argparse 
import numpy as np
import pyaudio
import wave
import pyautogui

from email.mime import audio
from scipy.io import wavfile 
from hmmlearn import hmm
from python_speech_features import mfcc


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44800
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "real_time_audio.wav"

import pyautogui

def press_right():
    pyautogui.press("r")

def press_left():
    pyautogui.press("l")

def press_up():
    pyautogui.press("u")

def press_down():
    pyautogui.press("d")

# Dispatch dictionary to map labels to actions
action_dispatch = {
    "right": press_right,
    "left": press_left,
    "up": press_up,
    "down": press_down,
}



# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)


if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder
    # input_folder = 'audio'

    hmm_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []

        # Iterate through the audio files (leaving 1 file for testing in each class)
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq,nfft=2048)

            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            # Append the label
            y_words.append(label)

        print ('X.shape =', X.shape)
        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        
    
    # Input file to speech recognition
    input_file_audio = 'real_time_audio.wav'

    while True:

        # start Recording
        print ("recording...")

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        # Read input file
        sampling_freq, audio = wavfile.read(input_file_audio)

        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq,nfft=2048)

        # Define variables
        max_score =  float('-inf') 
        output_label =  [ float('-inf') ] 

        # Iterate through all HMM models and pick 
        # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            print ('{}: {}'.format(label, score))
            if score > max_score:
                max_score = score
                output_label = label

        output_label = output_label.split()
        print ("Predicted:", output_label[0], '\n' )
        break
        action = action_dispatch.get(output_label[0])
        if action:
            action()  # Call the associated function
        else:
            print(f"No action defined for label: {predicted_label}")


    # # Classify input data
    # for input_file in input_files_audio:
    #     # Read input file
    #     sampling_freq, audio = wavfile.read(input_file)

    #     # Extract MFCC features
    #     mfcc_features = mfcc(audio, sampling_freq)

    #     # Define variables
    #     max_score = [ float('-inf') ]
    #     output_label = [ float('-inf') ]

    #     # Iterate through all HMM models and pick 
    #     # the one with the highest score
    #     for item in hmm_models:
    #         hmm_model, label = item
    #         # print( label )
    #         score = hmm_model.get_score(mfcc_features)
    #         # print ('{}: {}'.format(label, score))

    #         if score > max_score:
    #             max_score = score
    #             output_label = label
    

    #     # Print the output
    #     # print ("\nTrue:", input_file[input_file.find('/')+1:input_file.rfind('/')])

    #     output = output_label.split()
    #     output_label = output[0].split('\\')[1]
    #     print ("Predicted:", output_label )


    #     if ( output_label == 'down' ):
    #         pyautogui.press("d")
    #     if ( output_label == 'left' ):
    #         pyautogui.press("l")
    #     if ( output_label == 'right' ):
    #         pyautogui.press("r")
    #     if ( output_label == 'spin' ):
    #         pyautogui.press("s")
    #     if ( output_label == 'default' ):
    #         pyautogui.press("y")
        

