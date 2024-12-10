import cv2
import torch
import skvideo.io
import numpy as np
import pandas as pd

import constants as c


def read_video_as_numpy(path):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
        print(str(len(frames))+"/"+str(length))

    video = np.stack(frames, axis=0) # dimensions (T, H, W, C)  Time, Height, Width, ColorChannel
    
    return video

def read_video_sk(vpath):
    videodata = skvideo.io.vread(vpath)  
    return videodata

def concat_tensor(tensor1, tensor2, axis=0):
    if tensor1 is None:
        concated_tensor = tensor2
    else:
        concated_tensor = torch.cat((tensor1, tensor2), axis)
    return concated_tensor

def concat_np(tensor1, tensor2, axis=0):
    if tensor1 is None:
        concated_tensor = tensor2
    else:
        concated_tensor = np.concatenate((tensor1, tensor2), axis)
    return concated_tensor

def concat_pd(df1, df2, axis=0):
    if df1 is None:
        concated_tensor = df2
    else:
        concated_tensor = pd.concat([df1, df2], axis=axis)
    return concated_tensor

def read_feats(fpath, ext):
    if ext == "npy":
        feats = np.load(fpath)
    elif ext == "csv":
        feats = pd.read_csv(fpath)
    else:
        feats = np.load(fpath)
    return feats

def expand_dims_grouped_df(grouped):
    expanded_df = None
    for name, group in grouped:
        group = np.expand_dims(group, axis=0)
        expanded_df = concat_np(expanded_df, group, axis=0)
    return expanded_df

def get_videofeats_index(social_sig="pitch", video_sub_feat=""):
    if social_sig == "aucs":
        return c.AUCS_FEATURES.index(video_sub_feat)
    if social_sig == "facepose":
        return c.FACEPOSE_FEATURES.index(video_sub_feat)
    if social_sig == "emotions":
        return c.EMOTIONS_FEATURES.index(video_sub_feat)
    return 0

def convert_dictvalues_to_array(dictionary):
    '''Converts lists of values in a dictionary to numpy arrays'''
    return pd.DataFrame.from_dict(dictionary).values # np.array([np.array(v) for k, v in dictionary.items()])


def remove_none_in_dict(d: dict) -> dict:
    res_d = {k:v for k,v in d.items() if v is not None}
    return res_d

def NormalizeData(data, min=1, max=9):
    return (data - min) / (max - min) 