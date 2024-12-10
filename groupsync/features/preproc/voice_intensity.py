import librosa
import parselmouth
import numpy as np

def extract_intensity(wav_path):
    snd = parselmouth.Sound(wav_path)
    intensity = snd.to_intensity().values.T
    pitch = snd.to_pitch().selected_array['frequency']

    return intensity, pitch