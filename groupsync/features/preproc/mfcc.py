import librosa

def extract_mfcc(waveform, sr, n_mfcc=5):
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc