import os
import librosa

# specify your directory here
audio_dir = 'Singing'

for dir_path, dirs, files in os.walk(audio_dir):
    for file_name in files:
        if file_name.endswith('.wav'):  # checks if the file is a wav file
            file_path = os.path.join(dir_path, file_name)
            audio_data, sampling_rate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sampling_rate)
            if duration < 1.0:  # check if the duration is less than 1 second
                os.remove(file_path)  # delete the file
                print(f'Deleted {file_path} due to short duration.')