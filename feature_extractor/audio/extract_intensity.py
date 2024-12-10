"""
.. moduleauthor:: Navin Raj Prabhu
"""

"""
 -------------------- Intensity Extractor for Dataset --------------------------

"""

from pathlib import Path
from tqdm import tqdm 
import numpy as np
import pandas as pd
import argparse
import glob
import sys
import os
import time

import feature_extractor.audio.utils as utils
from utilities.dataset import get_MEMO_dataset
import utilities.data as data_utils
import constants as c

def extractor():
    
    memoObj = get_MEMO_dataset("raw")
    
    for interaction in memoObj.interactions_df:
        for participant in interaction.participants:
            # Folder Orga
            outfile = participant.get_features_path("audio_v2", "intensity")
            outfldr = "/".join(outfile.split("/")[:-1])
            os.makedirs(outfldr, exist_ok=True)
            if not os.path.exists(outfile):
                print("Extracting Intensity for Interaction: ", interaction.id + " and Participant: ", participant.id) 
                waveform = participant.get_audio()
                
                segmented_waveforms = utils.batch_as_segments(waveform, c.MEMO_ANNOTATION_SEG_SIZE, c.MEMO_AUDIO_SR, "audio")
                segmented_features = None
                print("num segmeents = ", len(segmented_waveforms))
                for i, sample in enumerate(segmented_waveforms):
                    # start_time = time.time()
                    seg_feat = utils.extract_feats(np.squeeze(sample), "intensity", c.MEMO_AUDIO_SR)
                    seg_feat = np.transpose(seg_feat)
                    segmented_features = data_utils.concat_np(segmented_features, seg_feat, axis=0)
                    # print("Time Taken for 1 sample  - ", time.time()-start_time, " secs")
                print("no.of.segments = ", segmented_features.shape[0], ", Intensity Shape - ", segmented_features[0].shape, " in ", segmented_features.shape)

                np.save(outfile, segmented_features)
            
def main():
    print('Initializing Intensity extraction Process..')
    extractor()
    
    
    
if __name__ == '__main__':
    main()

