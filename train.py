import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import json
import glob
import librosa
import argparse

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from simple_train import simple_train
from batched_train import batched_train
from readers.memo_ga import MEMODataset
from utilities.dataset import get_MEMO_dataset
from utilities.train_infer import scan_checkpoint, load_checkpoint, save_checkpoint, build_env, AttrDict, delete_checkpoint, get_feats_list

# "features": ["pitch--mean-", "pitch--std-", "pitch--grad-", "mfcc-0-mean-", "intensity--mean-", "intensity--grad-",
#                  "pitch--synchrony_corrcoeff-mean", "aucs-AU07-synchrony_corrcoeff-mean", "aucs-AU20-synchrony_maxcorr-mean", "aucs-AU20-convergence_symmetric-mean"],
# "nfeatures": 10,

def train(config: AttrDict):

    torch.cuda.manual_seed(config.seed)
    device = torch.device('cuda:{:d}'.format(config.gpu))

    # Setup config features 
    features, nfeatures = get_feats_list(config.base_features, with_sync=config.use_syncfeats, only_sync=config.only_syncfeats)
    config.__dict__["features"] = features
    config.__dict__["nfeatures"] = nfeatures
    
    
    # Prepare Dataset and Dataloader (using config)
    memo = get_MEMO_dataset(mode="dataset", config=config)
    train_data, valid_data, test_data = memo.partition_dataset()
    
    trainset, validset, testset = MEMODataset(train_data), MEMODataset(valid_data), MEMODataset(test_data)
    
    if config.train_type == "simple":
        simple_train(config, trainset, validset, testset)
    elif config.train_type == "batched":
        batched_train(config, trainset, validset, testset)


# Define Main function and read parameters
def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint_path', required=True)

    parser.add_argument('--training_epochs', default=100, type=int)
    parser.add_argument('--checkpoint_interval', default=2, type=int)

    a = parser.parse_args()

    # Read Config
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = AttrDict(json_config)
    checkpoint_path = os.path.join(a.checkpoint_path, config.train_name)
    build_env(a.config, checkpoint_path, 'config.json')
    config.checkpoint_path = checkpoint_path
    
    train(config)

# python train.py --config 'configs/memo_train.json' --checkpoint_path 'checkpoints/'

if __name__ == '__main__':
    main()
