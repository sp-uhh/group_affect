import cv2
import torch
import skvideo.io
import numpy as np
import pandas as pd

import readers.memo_ga as MEMO
import constants as c


#################################################
# MEMO DATASET UTIL functions.
#################################################

def get_MEMO_dataset(mode="raw", config: dict = {}) -> MEMO.MEMOGroupAff:
    memoObj = MEMO.MEMOGroupAff(config=config)
    memoObj.get_interactions_df()
    if mode=="dataset":
        memoObj.load_labels() # Load Group Affect labels for all interactions
        memoObj.load_indiv_feats_all() # Load Indiv level features before group-level feats
        memoObj.load_features() # Load Group-level Features required from the config file 
        memoObj.load_dataset() # For Dataset tuple
    return memoObj

#################################################
#################################################