import os
import sys
import cv2
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import glob
import math

import constants as c
from torch.utils.data import Dataset, DataLoader

from readers.memo_interactions import Interactions
from readers.utils.utils import get_interaction_fname, get_interaction_id, get_featurekey
from utilities.data import concat_pd, NormalizeData, concat_np
from utilities.agreement_utils import calculate_goldstandard_gt
from readers.loader_features import load_individual_features, load_group_features

np.set_printoptions(threshold=sys.maxsize)

class MEMODataset(Dataset):
    def __init__(self, dataset_df):

        self.interactionID = dataset_df[0] #dataset_df["interactionID"].values
        self.onsetTime  = torch.IntTensor(dataset_df[2]) #torch.IntTensor(dataset_df["onsetTime"].values)
        self.offsetTime = torch.IntTensor(dataset_df[3]) #torch.IntTensor(dataset_df["offsetTime"].values)
        self.ArousalAvg = torch.FloatTensor(NormalizeData(dataset_df[4])) #torch.FloatTensor(NormalizeData(dataset_df["ArousalAvg"].values))
        self.ValenceAvg = torch.FloatTensor(NormalizeData(dataset_df[5])) #torch.FloatTensor(NormalizeData(dataset_df["ValenceAvg"].values))
        self.ArousalGT  = torch.FloatTensor(NormalizeData(dataset_df[6])) #torch.FloatTensor(NormalizeData(dataset_df["ArousalGT"].values))
        self.ValenceGT  = torch.FloatTensor(NormalizeData(dataset_df[7])) #torch.FloatTensor(NormalizeData(dataset_df["ValenceGT"].values))
                
        # features = dataset_df.drop(["interactionID", "onsetTime", "offsetTime", "ArousalAvg", "ValenceAvg", "ArousalGT", "ValenceGT"], axis=1, inplace=False)
        
        # self.feats_cols = features.columns.values
        self.features = torch.FloatTensor(dataset_df[1])
        
    def __len__(self):
        return len(self.interactionID)
        
    def __getitem__(self, idx):
        grpId   = torch.tensor([int(self.interactionID[idx].split('_')[0])])
        sesId   = torch.tensor([int(self.interactionID[idx].split('_')[1])])
        
        onsetTime       = self.onsetTime[idx]
        offsetTime      = self.offsetTime[idx]
        
        ArousalAvg      = self.ArousalAvg[idx]
        ValenceAvg      = self.ValenceAvg[idx]
        ArousalGT       = self.ArousalGT[idx]
        ValenceGT       = self.ValenceGT[idx]
        
        features = self.features[idx]

        return grpId, sesId, onsetTime, offsetTime, ArousalAvg, ValenceAvg, ArousalGT, ValenceGT, features
    

class MEMOGroupAff:
    def __init__(self, config: dict = {}):

        self.label = "MEMO-GroupAffect"
        self.root_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset/MEMO_GroupAffect/")
        self.config =config
        
        self.labels_folder  = self.root_folder + "Labels/"
        self.annotators_labels_folder  = self.root_folder + "Labels/Interaction/"
        self.interactions_labels_folder = self.root_folder + "Labels/Interaction/"
        self.annotator_agreement_folder = self.labels_folder + "Agreement/"
        self.labels_df_path  = self.labels_folder + "labels.csv"
        self.joint_labels_df_path  = self.labels_folder + "joint_labels.csv"
        self.joint_gt_df_path  = self.labels_folder + "joint_labels_gt.csv"        
        self.partitions_path  = self.root_folder + "spkr_partitions.txt"  if len(config.keys()) == 0 else self.root_folder + config.datasplit_manifest
        self.partitions_data = pd.read_csv(self.partitions_path, sep=";", header=0)
        
        self.videos_folder = self.root_folder + "Videos/"
        self.audios_folder = self.root_folder + "Audios/"
        self.config_folder = self.root_folder + "Configs/"
        self.features_folder = self.root_folder + "Features/"
        
        
        self.timestamps_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset/MEMO_GroupAffect/Labels/timestamps_used.csv")

        self.interactions_df = None
        self.dataset = None
        
        # Labels
        self.arousal = "Arousal"
        self.valence = "Valence"
        self.pleasure = "Pleasure"
        
        # Groups and Sessions 
        self.groups = ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.sessions = ['1', '2', '3']
        self.emo_dims = ["arousal", "valence"]
        self.annotators = ['001', '002', '003', '004', '005', '006', '007', '008']
        
        self.onset   = "Onset_Time"
        self.offset  = "Offset_Time"
        
        self.labels_df = None


    def get_interactions_df(self):
        interactions = []
        for groupId in self.groups:
            for sessId in self.sessions:
                if groupId + '_' + sessId not in ["8_3", "9_3"]:# Skip this particular interaction as separated audio not available
                    interactionObj = Interactions(groupId, sessId,\
                                            self.audios_folder, self.videos_folder, self.features_folder, self.config_folder, self.annotators_labels_folder)
                    interactions.append(interactionObj)
        self.interactions_df = interactions
        return interactions
    
    def load_indiv_feats_all(self):
        # Audio
        self.load_featureset("pitch")
        self.load_featureset("mfcc")
        self.load_featureset("intensity")
        self.load_featureset("vggish")
        # Video
        self.load_featureset("aucs")
        self.load_featureset("facepose")
    
    def combine_features_for_interaction(self, interaction):
        feats_df = None
        # if self.config.model_dynamics:
        for key in interaction.audio_feats.keys():
            curr_feat = np.expand_dims(interaction.audio_feats[key], axis=1)
            feats_df = concat_np(feats_df, curr_feat, axis=1)
        for key in interaction.video_feats.keys():
            curr_feat = np.expand_dims(interaction.video_feats[key], axis=1)
            feats_df = concat_np(feats_df, curr_feat, axis=1)
        # else:
        #     if len(interaction.audio_feats.keys()) > 0 and len(interaction.video_feats.keys()) > 0:
        #         audio_feat_df = pd.DataFrame.from_dict(interaction.audio_feats)
        #         video_feat_df = pd.DataFrame.from_dict(interaction.video_feats)
        #         feats_df = concat_pd(audio_feat_df, video_feat_df, axis=1)
        #     elif len(interaction.audio_feats.keys()) > 0:
        #         audio_feat_df = pd.DataFrame.from_dict(interaction.audio_feats)
        #         feats_df = audio_feat_df
        #     elif len(interaction.video_feats.keys()) > 0:
        #         video_feat_df = pd.DataFrame.from_dict(interaction.video_feats)
        #         feats_df = video_feat_df
        
        return feats_df            
    
    def dataset_stack_features(self):

        data_df = None
        interactionIDs = []
        print("Audio features preloaded: ", self.interactions_df[0].audio_feats.keys())
        print("Video features preloaded: ", self.interactions_df[0].video_feats.keys())
        print("Total num. features = ", len(self.interactions_df[0].audio_feats.keys()) + len(self.interactions_df[0].video_feats.keys()))
        
        for interaction in self.interactions_df:
            feats_df = self.combine_features_for_interaction(interaction)
            data_df = concat_np(data_df, feats_df, axis=0)
            
            interactionIDs.extend(np.repeat(interaction.group+'_'+interaction.session, feats_df.shape[0]))
            
        # data_df["interactionID"] = interactionIDs
        return data_df, np.array(interactionIDs)
    
              
    def load_dataset(self):
        # Loading Dataset
        features, interactionIDs = self.dataset_stack_features() # get_featurekey(ftype, feature, sub_feature)
        
        ## Load Labels
        ids = self.labels_df["interactionID"].values
        onset  = self.labels_df["onsetTime"].values
        offset = self.labels_df["offsetTime"].values
        arousAvg = self.labels_df["ArousalAvg"].values
        valenAvg = self.labels_df["ValenceAvg"].values
        arousGT  = self.labels_df["ArousalGT"].values
        valenGT  = self.labels_df["ValenceGT"].values
        
        # Check Sync
        if np.array_equal(interactionIDs, ids):
            print("DATASET STACKING is IN SYNC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        else:
            print(np.setdiff1d(ids, interactionIDs))
            print("WARNING!!!!!!!!!!!!!!!!!! DATASET STACKING is NOT IN SYNC !!!!!!!!!!!!!!!")

        # self.dataset = {}
        # self.dataset["features"]       = features 
        # self.dataset["interactionID"]  = ids
        # self.dataset["onsetTime"]      = onset
        # self.dataset["offsetTime"]     = offset
        # self.dataset["ArousalAvg"]     = arousAvg
        # self.dataset["ValenceAvg"]     = valenAvg
        # self.dataset["ArousalGT"]      = arousGT
        # self.dataset["ValenceGT"]      = valenGT
        
        self.dataset = (ids, features, onset, offset, arousAvg, valenAvg, arousGT, valenGT)
    
    def tuple_datapartitions(self, partition='Train'):
        req_ids = self.partitions_data.loc[self.partitions_data['Partition'] == partition]["InteractionID"].values.tolist()
        
        picker = np.argwhere(np.isin(self.dataset[0], req_ids)).ravel()
    
        ids_dataset         = self.dataset[0][picker]
        features_dataset    = self.dataset[1][picker]
        onsets_dataset      = self.dataset[2][picker]
        offsets_dataset     = self.dataset[3][picker]
        arousAvg_dataset    = self.dataset[4][picker]
        valenAvg_dataset    = self.dataset[5][picker]
        arousGT_dataset     = self.dataset[6][picker]
        valenGT_dataset     = self.dataset[7][picker]
        
        return (ids_dataset, features_dataset, onsets_dataset, offsets_dataset, arousAvg_dataset, valenAvg_dataset, arousGT_dataset, valenGT_dataset)
    
    def partition_dataset(self):
        assert self.dataset is not None, "Config for dataset creation not set.!!!!"
        
        # train_ids  = self.partitions_data.loc[self.partitions_data['Partition'] == 'Train']["InteractionID"].values        
        # valid_ids  = self.partitions_data.loc[self.partitions_data['Partition'] == 'Validation']["InteractionID"].values
        # test_ids   = self.partitions_data.loc[self.partitions_data['Partition'] == 'Test']["InteractionID"].values
                
        # train_data = self.dataset.loc[self.dataset['interactionID'].isin(train_ids)] # list(filter(lambda x: x[0] in train_ids, self.dataset))
        # valid_data = self.dataset.loc[self.dataset['interactionID'].isin(valid_ids)] # list(filter(lambda x: x[0] in valid_ids, self.dataset))
        # test_data  = self.dataset.loc[self.dataset['interactionID'].isin(test_ids)]  # list(filter(lambda x: x[0] in test_ids, self.dataset))

        train_data = self.tuple_datapartitions(partition='Train')
        valid_data = self.tuple_datapartitions(partition='Validation')
        test_data  = self.tuple_datapartitions(partition='Test')

        print("Train Data size: ", len(train_data[0]), "Valid Data size: ", len(valid_data[0]), "Test Data size: ", len(test_data[0]))
        
        return train_data, valid_data, test_data
        
    def load_features(self):
        assert len(self.config.keys()) > 0, "Config for dataset creation not set.!!!!"
    
        # for indiv_ftype, group_ftype, sub_feat in zip(self.config.indivfeats_list, self.config.aggfeats_list, self.config.subfeats_list): #self.config.features:
        for feattag in self.config.features:
            # print("feture current: ", feattag)
            indiv_ftype, sub_feat, group_ftype, syncagg_type = feattag.split("-")
            if "synchrony" in group_ftype or "convergence" in group_ftype:
                self.load_synchrony_convergence_featset(group_ftype, indiv_ftype, sub_feat, syncagg_type)
            else:
                agg_fn = c.groupfeat_aggs[group_ftype]
                self.load_aggregated_featset(indiv_ftype, agg_type=agg_fn, normalize=False, sub_feat=sub_feat if sub_feat != '' else None, temporal_agg=not self.config.model_dynamics)
            
    # Labels and Annotations processing ..................................................................................
    def get_annot_files_for(self, groupid='1', session='1'):
        filename = get_interaction_fname(groupid, session) #'group'+groupid+'_session'+session
        # print("Number of Annotations = " + str(len(glob.glob(self.labels_folder+"**/"+filename+".csv", recursive=True))))
        files = glob.glob(self.labels_folder+"**/"+filename+".csv", recursive=True)
        return np.array(files)
    
    def get_annot_for(self, groupid='1', sessionid='1'):
        files = self.get_annot_files_for(groupid, sessionid)
        interaction_id = groupid+"_"+sessionid
        
        arousal_annots = []
        valence_annots = []
        onset_times = []
        offset_times = []
        interact_ids = []
        group_ids = []
        session_ids = []
        annotator_ids = []
        
        print()
        for i, fpath in enumerate(files):
            annot_id = str(fpath.split("/")[-2])
            curr_arous, curr_valen, curr_onsets, curr_offsets = self.clean_ga_df(fpath)
            print("len annot = ", len(curr_valen))
            arousal_annots.extend(curr_arous)
            valence_annots.extend(curr_valen)
            onset_times.extend(curr_onsets)
            offset_times.extend(curr_offsets)
            
            session_ids.extend([str(sessionid)]*len(curr_arous))
            group_ids.extend([str(groupid)]*len(curr_arous))
            interact_ids.extend([str(interaction_id)]*len(curr_arous))
            annotator_ids.extend([str(annot_id)]*len(curr_arous))
        
        if len(files) != 0:
            if len(curr_arous)*len(files) - len(arousal_annots) != 0 :
                print("group ", groupid, " and session ", sessionid,)
                print("Total Arousal Len= ", str(len(arousal_annots)), ", Current Arous File Len= ", str(len(curr_arous)), "No.of annotations = ", str(len(files))) 
                print("Expected = ", str(len(curr_arous)*len(files)))
                print("Diff = ", str(len(curr_arous)*len(files) - len(arousal_annots)))
        
        return arousal_annots, valence_annots, onset_times, offset_times, interact_ids, group_ids, session_ids, annotator_ids

    def clean_ga_df(self, file):
        print(file)
        data = pd.read_csv(file, sep=";")
        notna_ind = data[self.arousal].notna()
        arousal_annot, valence_annot = self.clean_ga_annotations(data)
        if len(valence_annot) == 0:
            # Fixing missing annot for 1 file
            valence_annot = [0.0]*len(arousal_annot)
        onsets, offsets = self.clean_ga_times(data, len(arousal_annot), notna_ind)
        return arousal_annot, valence_annot, onsets, offsets
    
    def clean_timestamp(self, times):
        times = [math.ceil(float(x.replace(",", "."))) for x in times]
        return times
        
    def clean_ga_times(self, data, nframes, notna_ind):
        if self.onset and self.offset in data.columns:
            # Filter based on Non-Nan in Arousal Annot
            onsets  = self.clean_timestamp(data[notna_ind][self.onset].astype(str).values) # data[self.onset].dropna().values
            offsets = self.clean_timestamp(data[notna_ind][self.offset].astype(str).values) # data[self.offset].dropna().values
        else:
            onsets  = np.arange(0, nframes*15, 15)
            offsets = onsets + 15
        return onsets, offsets
            
    def clean_ga_annotations(self, data):
        # Arousal Cleanup
        arousal_annot = data[self.arousal].dropna().values
        # Valence Cleanup
        if self.valence in data.columns:
            valence_annot = data[self.valence].dropna().values
        elif self.pleasure in data.columns:
            valence_annot = data[self.pleasure].dropna().values
            
        return arousal_annot.astype(float), valence_annot.astype(float)
    
    def get_interaction_duration_for(self, groupid='1', session='1'):
        pass
    
    def read_labels_from_store(self, store='joint'):
        if store == 'joint':
            labels_df = pd.read_csv(self.joint_gt_df_path)
        else:
            labels_df = pd.read_csv(self.labels_df_path)
        labels_df = labels_df.loc[labels_df['interactionID'] != '9_3']
        self.labels_df = labels_df
        return labels_df
    
    def get_dataset_labels(self, from_store=False, store_mode='joint'):
        
        if from_store:
            return self.read_labels_from_store(store=store_mode)
        
        arous_all = []
        valen_all = []
                
        onsets_all = []        
        offsets_all = []        
        interactions_all = [] 
        groups_all = []        
        sessions_all = []        
        annotators_all = []        
        
        labels_df = pd.read_csv(self.joint_labels_df_path)
    
        for group in self.groups:
            for sess in self.sessions:
                if group+"_"+sess != "8_3":
                
                    arousal_annots, valence_annots,\
                        onset_times, offset_times,\
                            interact_ids, group_ids, session_ids, annotator_ids = self.get_annot_for(groupid=group, sessionid=sess)
                    
                    arous_all.extend(arousal_annots)
                    valen_all.extend(valence_annots)
                    onsets_all.extend(onset_times)
                    offsets_all.extend(offset_times)
                    interactions_all.extend(interact_ids)
                    groups_all.extend(group_ids)
                    sessions_all.extend(session_ids)
                    annotators_all.extend(annotator_ids)
                
        labels_dict = {
            "interactionID" : interactions_all,
            "groupID": groups_all,
            "sessionID": sessions_all,
            "annotatorID" : annotators_all,
            "onsetTime" : onsets_all,
            "offsetTime" : offsets_all,
            
            "Arousal" : arous_all,
            "Valence" : valen_all,
            
        } 
        
        labels_df = pd.DataFrame(data=labels_dict)
        
        labels_df.to_csv(self.labels_df_path, sep=',')
        self.labels_df = labels_df
        return labels_df

    def load_labels(self):
        assert self.interactions_df is not None, "Interaction DF is None. Run- --from utilities.data import get_MEMO_dataset-- First . "
        self.read_labels_from_store(store='joint')
        for interaction in self.interactions_df:
            interaction.load_group_affect()
    
    def load_annotators_for_interactions(self):
        assert self.interactions_df is not None, "Interaction DF is None. Run- --from utilities.data import get_MEMO_dataset-- First . "
        for interaction in self.interactions_df:
            interaction.load_interaction_annotators()
    ######################################## Labels and Annotations processing ########################################
    
    # Partitioning Utils. ..................................................................................
    def get_interactions_for(self, partition='train'):
        pass
    
    ######################################## Partitioning Utils. ########################################


    def load_featureset(self, ftype, level=["participant"]):
        assert self.interactions_df is not None, "Interaction DF is None. Run- --from utilities.data import get_MEMO_dataset-- First . "
        for interaction in self.interactions_df:
            if "participant" in level:
                interaction.load_participant_feature(ftype)
            if "group" in level:
                interaction.load_group_feature(ftype)    
    
    def load_aggregated_featset(self, ftype, agg_type=np.mean, temporal_agg=True, normalize=False, sub_feat=None):
        # assert getattr(self.interactions_df[0], ftype) is not None, "Feature is not extracted at participant level. "
        for interaction in self.interactions_df:
            agg_feats = interaction.extract_aggregate_group_feats(ftype, agg_type, temporal_agg, normalize)
            # if ftype == "mfcc":
            #     agg_feats = agg_feats[..., int(sub_feat)]
            # elif ftype == "aucs":
            #     agg_feats = agg_feats[..., c.AUCS_FEATURES.index(sub_feat)]
            # elif ftype == "facepose":
            #     agg_feats = agg_feats[..., c.FACEPOSE_FEATURES.index(sub_feat)]

            # if interaction.id == "group12_session3" and interaction.get_ftype_modality(ftype) == "audio":
            #     if agg_feats.shape[0] != 188:
            #         print(agg_feats.shape)
            #         agg_feats = concat_np(agg_feats, np.expand_dims(agg_feats[-1], axis=0), axis=0) # Repeat last row once
            #         agg_feats = concat_np(agg_feats, np.expand_dims(agg_feats[-1], axis=0), axis=0) # Repeat last row once
            #         print("after", agg_feats.shape)
                
            interaction.set_aggregate_group_feats(agg_feats, ftype, agg_type=agg_type.__name__)#, sub_feat=sub_feat)
            
    def load_synchrony_convergence_featset(self, ftype, feature, sub_feature, groupagg=None):
        for interaction in self.interactions_df:
            modality = interaction.get_ftype_modality(feature)
            synconv_feats = interaction.get_group_feats(ftype, feature, sub_feature, modality)
            feature_key = get_featurekey(ftype, feature, sub_feature)

            # if groupagg is not None:
            #     synconv_feats = synconv_feats[groupagg].values
            #     feature_key = feature_key + "_" + groupagg
            # if interaction.id == "group12_session3" and modality == "audio":
            #     synconv_feats = concat_np(synconv_feats, np.expand_dims(synconv_feats[-1], axis=0), axis=0) # Repeat last row once
            #     synconv_feats = concat_np(synconv_feats, np.expand_dims(synconv_feats[-1], axis=0), axis=0) # Repeat last row once
                
            interaction.set_group_feats(synconv_feats, feature_key, modality)
        
    
    # Videos Processing for Dataset ..................................................................................
    def get_video_frames_for(self, groupid='1', session='1'):
        interaction_id = get_interaction_id(groupid, session)
        pass
    
    def get_dataset_videos(self, req_interactions):
        pass
    
    def get_video_path(self, fname):
        return self.videos_folder + fname + ".mp4"
    ######################################## Videos Processing ########################################
    

    # Audios Processing for Dataset  ..................................................................................
    def get_audio_frames_for(self, groupid='1', session='1'):
        pass
    
    def get_dataset_audios(self, req_interactions):
        pass
    ######################################## Audios Processing ########################################
    
    
    
    
