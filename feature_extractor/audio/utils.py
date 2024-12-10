import os
import sys
import time 
import torch
import itertools
import torchaudio
import numpy as np
import pandas as pd
from scipy import stats

# preproc methods
from groupsync.features.preproc.fundamental_freq import extract_f0 #pitch/f0
from groupsync.features.preproc.mfcc import extract_mfcc #mfcc
from groupsync.features.preproc.voice_intensity import extract_intensity #voice_intensity

import constants as c

def batch(data, n=100):
    return np.array_split(data, n)

def unbatch(batch):
    return np.concatenate(batch)

def batched_preproc_wav(wav, method, sr, nbatch=100):
    batched_feats = []
    batch_wav = batch(wav, nbatch)
    for i, sample in enumerate(batch_wav):
        # start_time = time.time()
        feat = extract_feats(np.squeeze(sample), method, sr)
        batched_feats.append(feat)
        # print("Time Taken for 1 sample  - ", time.time()-start_time, " secs")
    feats = unbatch(batched_feats)
    # feats = pd.DataFrame(feats) # To pandas dataframe, aligning datatype for synchrony extract
    return feats

def batch_as_segments(data, seg_size=15, sr=c.MEMO_AUDIO_SR, modality="audio"):
    """
    Batch input audio/video with respect to segment size
    Input: if audio- (T,) if video- (T, W, H, C)  [T = sr*total audio lenght in secs]
    Output: if audio- (T/(seg_size*sr), seg_size*sr) if video- (T/(seg_size*sr), seg_size*sr, W, H, C) 
    
    Args:
        seg_size (int, optional): _description_. Defaults to 15 secs.
        sr (_type_, optional): _description_. Defaults to c.MEMO_AUDIO_SR Hz.
    """
    
    T = data.shape[0]
    nsegments = int(T/(seg_size*sr))
    data = batch(data, n=nsegments)
    
    assert data[0].shape[0] == int(seg_size*sr), "Wrong Segment SIZE in batching .. " # Checks whether the sequence lenght is exactly seg_size of annotations (i.e., 15 secs for MEMO)
    return data

def extract_feats(sample, method, sr):
    feats = sample
    if method == 'pitch' or method == 'f0':
        feats = extract_f0(sample, method='lib_pyin')
    elif method == 'mfcc':
        feats = extract_mfcc(sample, sr, n_mfcc=5)
    elif method == 'intensity':
        feats, _ = extract_intensity(sample)
    return feats