# *Hidden-Markov-Model-Speech-Recognition HMM and MFCC* ✅

*Hidden Markov model (HMM) is the base of a set of successful techniques for acoustic modeling in speech recognition systems. The main reasons for this success are due to this model's analytic ability in the speech phenomenon and its accuracy in practical speech recognition systems.*

### Run:
```
python main.py --input-folder audio


# Hidden Markov Model-Based Speech Recognition

## Project Description
This project implements a speech recognition system using Hidden Markov Models (HMMs). It recognizes predefined voice commands in real-time and maps them to specific actions using MFCC (Mel-Frequency Cepstral Coefficients) features for audio processing.

## Features
- Records audio commands in real-time using a microphone.
- Extracts MFCC features from the audio data.
- Trains an HMM for each predefined command.
- Recognizes commands in real-time and performs actions based on the recognized command.

## Prerequisites
To run this project, you need the following:

### Hardware
- A computer with a microphone.

### Software
- Python 3.9 or above
- Libraries:
  - `numpy`
  - `pyaudio`
  - `wave`
  - `pyautogui`
  - `scipy`
  - `python_speech_features`
  - `hmmlearn`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/ademox/Implementing-HMM-Based-Speech-Recognition.git
   cd Implementing-HMM-Based-Speech-Recognition
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies with ``` pip install NamePakage ```
   ```

## Usage
### Training the Model

* Record your own instruction audios with the file **recording_audio.py** and replace in each command folder, the folder name is the one that defines the word or instruction.

1. Organize the training data in folders within an input directory. Each folder should correspond to a command label and contain `.wav` files of that command.
   Example structure:
   ```
   input-folder/
   ├── left/
   │   ├── left1.wav
   │   ├── left2.wav
   ├── right/
   │   ├── right1.wav
   │   ├── right2.wav
   ```
2. Run the script to train the HMMs:
   ```bash
   python main.py --input-folder <path-to-your-input-folder>
   ```

### Real-Time Recognition
The script will:
- Continuously record audio in real-time.
- Recognize the spoken command.
- Perform a corresponding keyboard action.

Run the recognition process:
```bash
python main.py --input-folder <path-to-your-input-folder>
```


## Key Components
### `HMMTrainer`
- Encapsulates training and scoring of HMMs using `hmmlearn`.
- Configured with Gaussian HMMs for modeling command-specific audio features.

### MFCC Features
- Extracted using `python_speech_features` for efficient audio signal processing.

## Challenges and Solutions
### Challenges
- **Noise in audio data**: Affected model accuracy.
- **Limited dataset**: Predefined commands with a small training set can lead to overfitting.

### Solutions
- Used short, clear recordings for training.
- Explored preprocessing techniques to improve signal quality.

## Future Improvements
- Extend to handle more commands.
- Enhance the system with deep learning models for higher accuracy.
- Add noise reduction and environment adaptation.

## Troubleshooting
1. **`OSError: [Errno -9998] Invalid number of channels`**:
   - Ensure your microphone supports the specified number of channels (`CHANNELS = 2`). Adjust to `1` if needed.
2. **Input Overflow Errors**:
   - Reduce the sampling rate or buffer size if encountering overflows.
3. **Accuracy Issues**:
   - Ensure consistent, noise-free audio recordings for training.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Libraries used: `hmmlearn`, `python_speech_features`, and others.
- Inspired by practical speech recognition challenges.
