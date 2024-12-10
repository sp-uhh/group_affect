import pandas as pd
import numpy as np
import json

def get_featurekey(ftype, feature, sub_feature):
    if "-" in ftype:
        feature_key = ftype.split("-")[0]+"_"+ftype.split("-")[1]+"_"+feature+"_"+sub_feature if sub_feature != '' else ftype+"_"+feature                
    else:
        feature_key = ftype+"_"+feature+"_"+sub_feature if sub_feature != '' else ftype+"_"+feature
    return feature_key

def get_interaction_id(groupid='1', session='1'):
    return groupid + "_" + session

def get_interaction_fname(groupid='1', session='1'):
    filename = 'group'+groupid+'_session'+session
    return filename

def dict_from_json(file):
    with open(file) as f:
        d = json.load(f)
    return d