"""
.. moduleauthor:: Navin Raj Prabhu
"""

"""
 -------------------- Pitch Extractor for Dataset --------------------------

"""

from pathlib import Path
from tqdm import tqdm 
import numpy as np
import pandas as pd
import argparse
import resampy
import glob
import sys
import os
import time


import feature_extractor.audio.utils as utils
from utilities.dataset import get_MEMO_dataset
import constants as c

def extractor():
    
    memoObj = get_MEMO_dataset("raw")
    memoInteractions = memoObj.interactions_df
    
    for interaction in memoInteractions:
        for participant in interaction.participants:
            # Folder Orga.
            outfile = participant.get_features_path("audio_v2", "pitch")
            outfldr = "/".join(outfile.split("/")[:-1])
            os.makedirs(outfldr, exist_ok=True)
            if not os.path.exists(outfile):
                print("Extracting PITCH for Interaction: ", interaction.id + " and Participant: ", participant.id) 
                waveform = participant.get_audio()
                print("Waveform - ", waveform.shape)
                
                segmented_waveforms = utils.batch_as_segments(waveform, c.MEMO_ANNOTATION_SEG_SIZE, c.MEMO_AUDIO_SR, "audio")
                segmented_features = []
                print("num segmeents = ", len(segmented_waveforms))
                for i, segment in tqdm(enumerate(segmented_waveforms)):
                    segment = resampy.resample(segment, c.MEMO_AUDIO_SR, c.MEMO_AUDIO_FEATEXTRACT_SR)
                    # start_time = time.time()
                    seg_feat = utils.extract_feats(np.squeeze(segment), "pitch", c.MEMO_AUDIO_SR)
                    segmented_features.append(seg_feat)
                    # print("Time Taken for 1 sample  - ", time.time()-start_time, " secs")
                print("no.of.segments = ", len(segmented_features), ", Pitch Shape - ", seg_feat.shape)
                segmented_features = np.stack(segmented_features, axis=0 )

                np.save(outfile, segmented_features)
            
def main():
    print('Initializing Pitch extraction Process..')
    extractor()
    
    
    
if __name__ == '__main__':
    main()

