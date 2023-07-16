import parselmouth
import numpy as np
from scipy.signal import savgol_filter
import os
import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Inference
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Segment, SlidingWindowFeature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch import Tensor
from typing import Tuple
import contextlib
from collections import Counter
import math
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler

class AudioAnalysis:
    model, utils = None, None  # Class variables for the model and utils

    @classmethod
    def load_model(cls):
        if cls.model is None or cls.utils is None:
            cls.model, cls.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=True)
            cls.model = cls.model.to(torch.device('cpu'))  # gpu or cpu
        return cls.model, cls.utils

    def __init__(self, file_path, sampling_rate, frame_period=0.01):
        self.file_path = file_path
        self.sampling_rate = sampling_rate
        self.frame_period = frame_period
        self.device = torch.device('cpu')
        self.bins = np.arange(-600, 6000+1, 16.67)

        self.model, self.utils = self.load_model()

    def get_voiced_segments(self, file_path):
        # Load the audio data using librosa
        audio_data, sampling_rate = librosa.load(file_path, sr=None)

        # Resample the audio to 22050 Hz if necessary
        if sampling_rate != 22050:
            audio_data = librosa.resample(audio_data, sampling_rate, 22050)
            sampling_rate = 22050

        # Convert the audio data to a tensor and send to device
        audio_data = torch.tensor(audio_data).to(self.device)

        # Get the speech timestamps
        (get_speech_ts, _, _, _, _) = self.utils
        speech_timestamps = get_speech_ts(audio_data, self.model)

        print("Speech timestamps: ", speech_timestamps)

        # Convert timestamps to samples
        # The Silero VAD works with a frame rate of 8000 Hz, so multiply by 8000 to convert from frames to samples
        voiced_segments = [(segment['start'] / 24000, segment['end'] / 24000) for segment in speech_timestamps]
        print("voiced segments: ", voiced_segments)
        return voiced_segments

    def calculate_pitch_contours(self, segment):
        sound = parselmouth.Sound(segment)
        pitch = sound.to_pitch_ac(self.frame_period, pitch_floor=75)  # Adjust the pitch floor value
        pitch_values = pitch.selected_array['frequency']
        return pitch_values

    def envelope_smoothing(self, pitch_contour, segment_length=1):
        pitch_contour = np.array(pitch_contour)
        num_segments = len(pitch_contour) // segment_length

        smoothed_pitch_contour = []
        for i in range(num_segments):
            segment = pitch_contour[i * segment_length:(i + 1) * segment_length]
            
            # Find the local maxima and minima of the pitch contour
            maxima = argrelextrema(segment, np.greater)[0]
            minima = argrelextrema(segment, np.less)[0]
            
            if len(maxima) == 0 or len(minima) == 0:
                # print("No local maxima or minima found in the pitch contour. Using the original segment.")
                smoothed_segment = segment
            else:   
                # Interpolate between these points to create the envelopes
                maxima_interpolator = interp1d(maxima, segment[maxima], kind='cubic', fill_value='extrapolate')
                minima_interpolator = interp1d(minima, segment[minima], kind='cubic', fill_value='extrapolate')

                # Calculate the upper and lower envelopes
                upper_envelope = maxima_interpolator(np.arange(len(segment)))
                lower_envelope = minima_interpolator(np.arange(len(segment)))

                # Calculate the smoothed pitch contour as the mean of the upper and lower envelopes
                smoothed_segment = (upper_envelope + lower_envelope) / 2
                
            smoothed_pitch_contour.extend(smoothed_segment)

        return np.array(smoothed_pitch_contour)



    
    def convert_to_cents(self, pitch):
        cents = []
        for f0 in pitch:
            if f0 > 0:  # Only convert positive pitch values
                cents.append(1200 * np.log2(f0 / 440.0) + 5800)
        return cents

    # def discretize_to_semitones(self, smoothed_cents):
    #     semitones = np.round(smoothed_cents / 100.0) * 100
    #     return semitones

    def discretize_to_semitones(self, smoothed_cents, steps_per_semitone=1):
        # Convert to semitones with the specified resolution
        semitones_contour = np.round(smoothed_cents / (100.0 / steps_per_semitone)) * (100.0 / steps_per_semitone)
        return semitones_contour

    def smooth_and_discretize_contour(self, cents_contour, min_length):
        min_length = min_length  # Replace with your duration requirement, in frames
        if len(cents_contour) >= min_length:
            smoothed = self.envelope_smoothing(cents_contour)
            semitones_contour = self.discretize_to_semitones(smoothed)
        else:
            semitones_contour = cents_contour  # Or handle short sequences differently, if needed
        semitones_contour = semitones_contour
        # print("SEMI CONTOUR ISSSSSSSSSSS", semitones_contour)
        return semitones_contour

    
    def semitone_to_note(self, semitone_sequence):
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        note_name_sequence = [note_names[int(semitone % 12)] + str(int(semitone // 12) - 1) for semitone in semitone_sequence]
        most_common_note_name, _ = Counter(note_name_sequence).most_common(1)[0]
        return most_common_note_name


    def find_longest_note(self, seq, min_length, max_variation):
        min_length_in_samples = int(min_length / self.frame_period)
        longest_note = None
        longest_note_length = 0
        longest_note_start = None
        longest_note_end = None
        for i in range(len(seq)):
            for j in range(i + min_length_in_samples, len(seq)):
                segment = seq[i:j]
                if np.max(segment) - np.min(segment) > max_variation:
                    break
                if (j - i) > longest_note_length:
                    longest_note = segment
                    longest_note_length = j - i
                    longest_note_start = i
                    longest_note_end = j
        return longest_note, longest_note_start, longest_note_end

    def split_sequence(self, seq, note_start, note_end):
        left_seq = seq[:note_start]
        right_seq = seq[note_end:]
        return left_seq, right_seq
    
    def detect_notes(self, sequence, min_length, max_variation):
        note_list = []
        label_list = [0] * len(sequence)  # Initialize all labels as 0 (no note)
        sequences = [sequence]  # Start with the entire sequence
        while sequences:
            new_sequences = []
            for seq in sequences:
                longest_note, note_start, note_end = self.find_longest_note(seq, min_length, max_variation)
                if longest_note is not None:
                    note_list.append(longest_note)
                    label_list[note_start:note_end] = [1] * len(longest_note)  # Set labels as 1 (note) for frames in the note
                    left_seq, right_seq = self.split_sequence(seq, note_start, note_end)
                    if len(left_seq) >= min_length:
                        new_sequences.append(left_seq)
                    if len(right_seq) >= min_length:
                        new_sequences.append(right_seq)
            sequences = new_sequences
        return note_list, label_list
    
    def determine_if_singing(self):
        return 'Singing' in self.file_path
    
    # def semitone_to_cents(self, semitone):
    #     A4_semitone = 69  # The semitone number of A4 is 69
    #     A4_frequency = 440.0  # The frequency of A4 is 440 Hz
    #     frequency = A4_frequency * 2**((semitone - A4_semitone) / 12)  # Calculate the frequency of the semitone
    #     cent = 1200 * np.log2(frequency / A4_frequency) + 5800  # Convert the frequency to cents
    #     return cent

    def analyze(self, min_length, max_variation):
        print("ANALYZING FILES: ", self.file_path)
        voiced_segments = self.get_voiced_segments(self.file_path)
        print(f"Number of voiced segments: {len(voiced_segments)}")
        print('file path is: ', self.file_path)
        if not voiced_segments:
            print("Voiced segments is empty.")
            
        audio_data, sampling_rate = librosa.load(self.file_path, sr=None)

        features = []
        for segment_start, segment_end in voiced_segments:
            # Convert segment start and end times from seconds to samples
            segment_start_samples = int(segment_start * sampling_rate)
            segment_end_samples = int(segment_end * sampling_rate)

            # Slice the audio data using these sample indices
            segment = audio_data[segment_start_samples:segment_end_samples]
            # print("SEGMENT:", segment)
            pitch_contour = self.calculate_pitch_contours(segment)
            cents_contour = self.convert_to_cents(pitch_contour)
            # print("CENTS CONTOUR:", cents_contour)
            # semitones_contour = self.smooth_and_discretize_contour(cents_contour, min_length=0.15)
            # print("SEMITONES CONTOUR:", semitones_contour)

            # # Compute the time axis (in seconds)
            # time_axis = np.arange(len(cents_contour)) * self.frame_period

            # # Create a new figure
            # plt.figure()

            # # Plot the original pitch contour (F0 in cent scale)
            # plt.plot(time_axis, cents_contour, label='Original')

            # # Plot the discretized pitch contour
            # plt.plot(time_axis, semitones_contour, label='Discretized')  # Multiply by 100 to convert back to cents for comparison

            # # Add labels and title
            # plt.xlabel('Time (s)')
            # plt.ylabel('Frequency (cents relative to A4)')
            # plt.title('Pitch contour and discretized contour')

            # # Add a legend
            # plt.legend()

            # # Show the plot
            # plt.show()
            notes, labels = self.detect_notes(cents_contour, min_length, max_variation)
            
            total_frames = len(segment)
            voiced_frames = len([p for p in pitch_contour if p > 0])
            note_frames = sum([len(note) for note in notes])

            for note in notes:
                note_duration = len(note) * self.frame_period  # Calculate the note duration in seconds
                note_name = self.semitone_to_note(note)
                print(f"Note: {note_name}")
                # print(f"Duration: {note_duration} seconds")

            # print(f"Number of voiced frames: {voiced_frames}")
            if voiced_frames == 0:
                print("voiced_frames is zero.")
                print('file path is:', self.file_path)
                proportion_notes = 0
            else:
                proportion_notes = note_frames / voiced_frames

            proportion_voiced = voiced_frames / total_frames if total_frames > 0 else 0

            print(f"Proportion of voiced frames: {proportion_voiced}")
            print(f"Proportion of note frames: {proportion_notes}")

            features.append((proportion_voiced, proportion_notes))

            # if len(notes) > 0:
            #     print(f"Notes detected in segment {segment_start}-{segment_end}: {notes}")
            # features.extend(notes)
        label = 1 if 'Singing' in self.file_path else 0

        return features, label
        # return features
    
def collect_data(audio_files, min_length, max_variation):
    features = []
    labels = []
    for file_path in audio_files:
        audio_analysis = AudioAnalysis(file_path, sampling_rate=24000)
        file_features, label = audio_analysis.analyze(min_length, max_variation)
        features.extend(file_features)
        labels.extend([label] * len(file_features))
    print("FEATURES:", features)
    print("LABELS:", labels)
    return features, labels

def train_and_test_svm(singing_data, speech_data, min_length, max_variation):
    # Collect the data
    singing_features, singing_labels = collect_data(singing_data, min_length, max_variation)
    speech_features, speech_labels = collect_data(speech_data, min_length, max_variation)

    # Combine the features and labels
    features = singing_features + speech_features
    labels = singing_labels + speech_labels

    print("FEATURES AREEEEEEEEEEEEE", features)
    print("LABELS AREEEEEEEEEEEEE", labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    print("X_train:", X_train)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the SVM
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    # Test the SVM
    y_pred = clf.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))

    # Print the confusion matrix
    print(confusion_matrix(y_test, y_pred))

def load_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                filepath = os.path.join(root, file)
                audio_files.append(filepath)
    return audio_files

# def analyze_and_plot(audio_files, min_length, max_variation, color, label):
#     scaler = MinMaxScaler()
#     count = 0  # Initialize the counter
#     for file_path in audio_files:
#         audio_analysis = AudioAnalysis(file_path, sampling_rate=16000)
#         features = audio_analysis.analyze(min_length, max_variation)

#         # Split the features into two lists for the x and y coordinates
#         if features:  # Checks if the features list is not empty
#             PV, PN = zip(*features)
#             # Only plot the samples where PV is not 0
#             filtered_features = [(pv, pn) for pv, pn in zip(PV, PN) if pv != 0]
#             if filtered_features:  # Check if the list is not empty
#                 PV, PN = zip(*filtered_features)
#                 # Reshape and scale the features
#                 # PV = np.array(PV).reshape(-1, 1)
#                 # PN = np.array(PN).reshape(-1, 1)
#                 # PV = scaler.fit_transform(PV)
#                 # PN = scaler.fit_transform(PN)
#                 plt.scatter(PV, PN, color=color, label=f'{label} ({len(PV)})')
#                 count += len(PV)  # Update the counter
#         else:
#             print('No features to plot for the current audio segment.')

#     return count  # Return the total count

def analyze_and_plot(audio_files, min_length, max_variation, color):
    scaler = MinMaxScaler()
    PV = []
    PN = []
    lengths = []
    notes = []
    files = []
    for file_path in audio_files:
        audio_analysis = AudioAnalysis(file_path, sampling_rate=16000)
        features, labels = audio_analysis.analyze(min_length, max_variation)

        # Split the features into two lists for the x and y coordinates
        if features:  # Checks if the features list is not empty
            pv, pn = zip(*features)
            # Only plot the samples where PV is not 0
            filtered_features = [(pv_, pn_) for pv_, pn_ in zip(pv, pn) if pv_ != 0]
            if filtered_features:  # Check if the list is not empty
                pv, pn = zip(*filtered_features)
                # Reshape and scale the features
                PV.extend(pv)
                PN.extend(pn)
                lengths.extend([len(f) for f in features])
                notes.extend([audio_analysis.semitone_to_note(f) for f in features])
                files.extend([file_path for _ in features])

    # Convert to numpy arrays
    PV = np.array(PV)
    PN = np.array(PN)

    # Scale the features to a range of 0 to 1
    PV = (PV - PV.min()) / (PV.max() - PV.min())
    PN = (PN - PN.min()) / (PN.max() - PN.min())

    # Create a trace for the plot
    trace = go.Scatter(
        x=PV,  # x-coordinates
        y=PN,  # y-coordinates
        mode='markers',
        marker=dict(color=color),
        text=[f"Length: {length}, Note: {note}, Proportion Voiced: {pv}, Proportion Notes: {pn}, File: {file}"
              for length, note, pv, pn, file in zip(lengths, notes, PV, PN, files)],  # hover text
        hoverinfo='text'  # only display the custom hover text
    )

    return trace

if __name__ == "__main__":
    singing_data = load_audio_files('Singing')[:100000]  # Select only the first 1000 singing audio files
    print(f"Singing data loaded. Number of files: {len(singing_data)}")
    singing_trace = analyze_and_plot(singing_data, 0.2, 100, 'red')

    speech_data = load_audio_files('Speech')[:100000]  # Select only the first 1000 speech audio files
    print(f"Speech data loaded. Number of files: {len(speech_data)}")
    speech_trace = analyze_and_plot(speech_data, 0.2, 100, 'blue')

    print("SINGING TRACE IS:", singing_trace)

    train_and_test_svm(singing_data, speech_data, 0.15, 150)

    singing_count = len(singing_trace['x'])
    speech_count = len(speech_trace['x'])

    # Create the layout for the plot
    layout = go.Layout(
        title=f'Audio Analysis (Singing: {singing_count}, Speech: {speech_count})',
        hovermode='closest',
        xaxis=dict(title='Proportion of voiced frames'),
        yaxis=dict(title='Proportion of note frames')
    )

    fig = go.Figure(data=[singing_trace, speech_trace], layout=layout)

    fig.show()
