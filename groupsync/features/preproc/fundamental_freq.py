import librosa
import numpy as np

def extract_f0(waveform, method='lib_yin'):

    if method == 'lib_pyin':   
        f0, voiced_flag, voiced_probs = librosa.pyin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    elif method == 'lib_yin':
        f0 = librosa.yin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    else:
        f0 = librosa.yin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    return np.nan_to_num(f0)