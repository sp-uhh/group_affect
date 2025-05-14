import pandas as pd
import numpy as np
import json
import os

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

def get_groupsize_for_batch(groupids, sessions):
    group_sizes = []
    for groupid, session in zip(groupids, sessions):
        # Convert groupid and session tensors to string
        groupid = groupid.numpy().astype(str)[0]
        session = session.numpy().astype(str)[0]
        group_size = get_groupsize(str(groupid), str(session))
        group_sizes.append(group_size)
    return group_sizes

def get_groupsize(groupid, session):
    fname = get_interaction_fname(groupid, session) + '.json'
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/MEMO_GroupAffect/Configs/") + fname
    data = dict_from_json(fpath)["participant_alias"]
    
    group_size = len(data.keys())-1 # -1 for tech support
    return group_size
