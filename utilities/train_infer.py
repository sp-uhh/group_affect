import os
import glob
import shutil

import torch
import numpy as np
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import constants as c
from utilities.loss import CCCLoss
from utilities.data import concat_np
from utilities.model import build_mlp
from sklearn.metrics import mean_squared_error

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")
    
def delete_checkpoint(filepaths):
    for file in filepaths:
        if os.path.isfile(file):
            print("Deleting old checkpoint at {}".format(file))
            os.remove(file)
            print("Complete.")

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def build_env(config, cp_path, config_name):
    t_path = os.path.join(cp_path, config_name)
    if config != t_path:
        os.makedirs(cp_path, exist_ok=True)
        shutil.copyfile(config, os.path.join(cp_path, config_name))

def get_model(config):
    
    if config.model_name == "svm":
        model = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler(), SVR(C=1.0, epsilon=0.2))
    elif config.model_name == "regression":
        model = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'), PolynomialFeatures(degree=4), LinearRegression())
    elif config.model_name == "mlp":
        model = build_mlp(config)

    return model

def calc_loss(loss_type, y_gt, y_pred):
    # Arousal in 0th column and Valence at 1st
    if loss_type == "ccc":
        arous_loss = CCCLoss(y_gt[:,0], y_pred[:,0])
        valen_loss = CCCLoss(y_gt[:,1], y_pred[:,1])
    elif loss_type == "mse":
        arous_loss = mean_squared_error(y_gt[:,0], y_pred[:,0])
        valen_loss = mean_squared_error(y_gt[:,1], y_pred[:,1])

    total_loss = (arous_loss + valen_loss)/2
    return arous_loss, valen_loss, total_loss
        
def stack_av_labels(config, arousGT, valenGT, arousAvg, valenAvg):
    arous, valen = (arousGT, valenGT) if config.labels == "GT" else (arousAvg, valenAvg)
    if len(arous.shape) == 1:
        arous = torch.unsqueeze(arous, axis=-1)
        valen = torch.unsqueeze(valen, axis=-1)
    labels = torch.cat((arous, valen), axis=1)
    return labels

def get_model_nparams(model):
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return nparams

def get_feats_list(base_features, with_sync=False, only_sync=False):
    feats_list = []
    
    for feat in base_features:
        tag = feat+"-"
        subfeats_list = c.SUBFEATS[feat] if feat in c.SUBFEATS.keys() else []
        if len(subfeats_list) == 0:
            tag = tag+"-"
            if with_sync:
                for sync, conv in zip(c.SYNCHRONY_FEATURES, c.CONVERGENCE_FEATURES):
                    tag_sync = tag+sync+"-"
                    tag_conv = tag+conv+"-"
                    for agg in c.GROUP_AGGS:
                        feats_list.append(tag_sync+agg)
                        feats_list.append(tag_conv+agg)
            
            if not only_sync:
                for agg in c.GROUP_AGGS:
                    feats_list.append(tag+agg+"-")
        else:
            for subfeat in subfeats_list:
                if with_sync:
                    for sync, conv in zip(c.SYNCHRONY_FEATURES, c.CONVERGENCE_FEATURES):
                        tag_sync = tag+subfeat+"-"+sync+"-"
                        tag_conv = tag+subfeat+"-"+conv+"-"
                        for agg in c.GROUP_AGGS:
                            feats_list.append(tag_sync+agg)
                            feats_list.append(tag_conv+agg)
                
                if not only_sync:
                    for agg in c.GROUP_AGGS:
                        feats_list.append(tag+subfeat+"-"+agg+"-")
                
    if only_sync:
        print("Removing featuires for only sync")
        feats_list = [ x for x in feats_list if "synchrony" in x or "convergence" in x]
                        
    print(feats_list)
    print("Total Features = ", len(feats_list))
    return feats_list, len(feats_list)