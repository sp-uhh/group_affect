import os
import cv2

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import glob
import math

import constants as c
import utilities.data as data_util
from readers.utils.utils import dict_from_json 

feats_inf = c.FEATURES_INFO
def is_extracted_at_group_level(ftype):
    return feats_inf[ftype]["extracted_at_group"]

def get_ext(ftype):
    return feats_inf[ftype]["ext"]

def get_modality(ftype):
    return feats_inf[ftype]["modality"]

class Participant:
    def __init__(self, interactionId, participantId, participantName,\
                        base_audios_path, base_videos_path, base_features_path, load_data=False):

        self.id = participantId
        self.name = participantName
        self.pid = participantId + "_" + participantName.lower()
        
        self.interactionId = interactionId
        
        self.base_audios_path = base_audios_path
        self.base_videos_path = base_videos_path
        self.base_features_path = base_features_path

        self.audio_sr = c.MEMO_AUDIO_SR
        self.audiofeat_sr = c.MEMO_AUDIO_FEATEXTRACT_SR
        self.audio_path = self.get_audio_path()
        self.video_sr = c.MEMO_VIDEO_SR
        self.video_path = self.get_video_path()
        
        if load_data:
            self.audio = self.get_audio()
            self.video = self.get_video()


        # Feature holders
        ## Audio
        self.pitch  = None
        self.mfcc   = None
        self.vggish = None
        self.intensity = None
        
        ## Video
        self.aucs           = None
        self.emotions_cat   = None
        self.facepose       = None
        self.resnet50       = None
        
    def get_features_path(self, modality="audio", ftype="pitch", ext="npy"):
        path = os.path.join(self.base_features_path, modality, ftype, self.interactionId, self.pid + "." + ext)
        return path
    
    def get_audio_path(self):
        path = os.path.join(self.base_audios_path, "separated_v2", self.interactionId, self.pid + ".wav")
        return path

    def get_video_path(self):
        path = os.path.join(self.base_videos_path, self.interactionId, self.pid + ".avi")
        return path
    
    def get_audio(self, mono=True):
        audio, _ = librosa.load(self.audio_path, sr=self.audio_sr, mono=mono) 
        # audio = librosa.resample(audio, self.audio_sr, self.audiofeat_sr)
        ############## HARDCODED BUG FIXES ###################################################
        # Wrong audio cuts fix for interactions 11_2 and 5_3
        if self.interactionId == "group11_session2":
            audio = audio[:(3105*self.audio_sr)] # TODO: in secs i.e., until 51:45 [note annotatiosn are until 52:00, last segment to be omitted everywhere]
        if self.interactionId == "group5_session3":
            audio = audio[:(2685*self.audio_sr)]  # TODO: in secs i.e., until 51:45 [note annotatiosn are until 52:00, last segment to be omitted everywhere]
        ############## HARDCODED BUG FIXES ###################################################
        return audio
    
    def get_video(self, from_store=True):
        # if from_store:
        video = cv2.VideoCapture(self.video_path) #np.load(self.video_store_path)
        # else:
        #     video = data_util.read_video_as_numpy(self.video_path)
        return video

    def get_feature(self, ftype):
        if getattr(self, ftype) is None:
            fpath = self.get_features_path(get_modality(ftype), ftype, get_ext(ftype))
            participant_feats = data_util.read_feats(fpath, get_ext(ftype))
            if get_modality(ftype) == "video":
                feat_sr = c.MEMO_ANNOTATION_SEG_SIZE
                if ftype == "resnet50":
                     ############## HARDCODED BUG FIXES ###################################################
                    # Wrong Video cuts fix for interactions 11_2
                    if self.interactionId == "group11_session2":
                        participant_feats = participant_feats[:77625, :] # TODO: in secs i.e., until 51:45 [note annotatiosn are until 52:00, last segment to be omitted everywhere]
                    ############## HARDCODED BUG FIXES ###################################################
                    participant_feats = pd.DataFrame(np.mean(participant_feats, axis=-1))
                    feat_sr = c.MEMO_ANNOTATION_SEG_SIZE * c.MEMO_VIDEO_SR
                grouped = participant_feats.groupby(participant_feats.index // feat_sr)
                participant_feats = np.squeeze(data_util.expand_dims_grouped_df(grouped))
        else:
            participant_feats = getattr(self, ftype)
        return participant_feats
    
    def set_feature(self, ftype, feat_df):
        # print(type(feat_df))
        # print(feat_df.shape)
        setattr(self, ftype, feat_df)
    
    def calculate_segment_feats(self, feat, temp_agg_type=np.mean, modality="audio"):   
        seg_feats = feat
        if modality == "audio":
            seg_feats = temp_agg_type(seg_feats, axis=1)
        elif modality == "video":
            seg_feats = temp_agg_type(seg_feats, axis=1)
            # seg_feats = pd.DataFrame(seg_feats)
            # seg_feats = seg_feats.groupby(seg_feats.index // c.MEMO_ANNOTATION_SEG_SIZE).mean()
            # print(type(seg_feats))
        return seg_feats
    
        
class Interactions:
    def __init__(self, groupId, sessId, audios_path, videos_path, features_path, configs_path, annot_path, load_data=False):
        
        self.id = "group"+groupId+"_session"+sessId
        self.group = groupId
        self.session = sessId
        self.config = self.get_interaction_config(configs_path)
        self.audios_path = audios_path
        self.videos_path = videos_path
        self.features_path = features_path
        self.labels_path = annot_path
        
        # Group-level Information
        self.audio_sr = c.MEMO_AUDIO_SR
        self.group_audio_path = self.get_group_audio_path()
        
        self.video_sr = c.MEMO_VIDEO_SR
        self.group_video_path = self.get_group_audio_path()
        
        if load_data:
            self.group_audio = self.get_group_audio()
            self.group_video = self.get_group_video(from_store=True)
        
        # Individual-level Information
        self.participantIds = self.get_participantIds()
        self.participants = self.get_group_participants()
        
        # Group-Affect annotations holder
        self.valence = None
        self.mean_valence_gt = None
        self.std_valence_gt = None
        self.var_valence_gt = None
        
        self.arousal = None
        self.mean_arousal_gt = None
        self.std_arousal_gt = None
        self.var_arousal_gt = None
        
        
        self.annotators = None
        self.num_annotators = None
        
        
        # Features
        # Audio
        self.afeats = []
        self.audio_feats = {}
        
        # Video
        self.vfeats = c.VIDEO_FEATURES
        self.video_feats = {}
        
        
    def get_interaction_config(self, base_path):
        config_path = os.path.join(base_path, self.id+".json")
        config = dict_from_json(config_path)
        return config
    
    def get_participantIds(self):
        ids = []
        participants_info = self.config["participant_alias"]
        for key, value in participants_info.items():
            if "TechSupport" not in value:
                ids.append(key+"_"+value)
        return ids
    
    def get_group_participants(self):
        participants = []
        for participantId in self.participantIds:
            pid = participantId.split("_")[0]
            pname = participantId.split("_")[1]
            participantObj = Participant(self.id, pid, pname, self.audios_path, self.videos_path, self.features_path)
            participants.append(participantObj)
        return participants
    
    def get_group_audio_path(self):
        audio_path = os.path.join(self.audios_path, self.id+".mp3")
        return audio_path
    
    def get_group_audio(self):
        audio, _ = librosa.load(self.group_audio_path, sr=self.audio_sr, mono=False) 
        return audio

    def get_group_video_path(self):
        video_path = os.path.join(self.videos_path, self.id+".mp4")
        return video_path
    
    def get_group_features_path(self, modality="audio", ftype="pitch", ext="npy", synconv=False):
        ftypes_path = "/".join(ftype.split('_')) if synconv else ftype # to account for nested synchrony convergence features also
        path = os.path.join(self.features_path, modality, ftypes_path, self.id + "." + ext)
        return path
    
    def get_group_video(self, from_store=True):
        if from_store:
            video = np.load(self.group_video_store_path)
        else:
            video = data_util.read_video_as_numpy(self.group_video_path)
        return video

    ################## FEATURE Get and Set  ##################
    def group_to_participant_feats(self, group_feats):
        # Funciton to convert the features extracted from a group-levle video
        participant_feats = group_feats
        # Sanity check for number of faces
        frame_index = participant_feats.iloc[:,0].values
        frames, counts = np.unique(frame_index, return_counts=True)
        return participant_feats
    
    def load_participant_feature(self, ftype):
        if is_extracted_at_group_level(ftype):
            feats_path = self.get_group_features_path(get_modality(ftype), ftype, get_ext(ftype))
            group_feats = data_util.read_feats(feats_path, get_ext(ftype))
            participant_feats = self.group_to_participant_feats(group_feats)
            for i, (participant, feat) in enumerate(zip(self.participants, participant_feats)):
                participant.set_feature(ftype, feat)
        else:
            for participant in self.participants:
                feat = participant.get_feature(ftype)
                participant.set_feature(ftype, feat)
    
    def get_ftype_modality(self, ftype, feature=''):
        if feature != '':
            return c.FEATURES_INFO[feature]["modality"]
        return c.FEATURES_INFO[ftype]["modality"]
    
    def get_feature(self, ftype):
        feats_dict = self.audio_feats if self.get_ftype_modality(ftype) == "audio" else self.video_feats
        return feats_dict[ftype] #getattr(self, ftype)
    
    def set_feature(self, ftype, featureset):
        feats_dict = self.audio_feats if self.get_ftype_modality(ftype) == "audio" else self.video_feats
        feats_dict[ftype] = featureset # setattr(self, ftype, feature)
    ##########################################################
    
    ################## Group-Affect Annotaitons/Ground-truth Get and Set  ##################
    def get_labels_path(self):
        annot_path = os.path.join(self.labels_path, self.id.split("_")[0], self.id.split("_")[1]+".csv")
        return annot_path
    
    def load_interaction_annotators(self):
        assert self.arousal is not None, "Labels not loaded. Load them first."
        avail_annotations = self.arousal.dropna(axis='columns')
        annotators = [annot_info.split("_")[-1] for annot_info in avail_annotations.columns.values]
        self.annotators = annotators
        self.num_annotators = len(annotators)
        
    def get_group_affect(self):
        if self.arousal is None or self.valence is None:
            annot_df = pd.read_csv(self.get_labels_path())
            arousal_df = annot_df.filter(like='Arousal', axis=1)
            valence_df = annot_df.filter(like='Valence', axis=1)
            return arousal_df, valence_df          
        else:
            return self.arousal, self.valence
    
    def set_groundtruth_group_affect(self, df, dim="arousal"):
        df = df.dropna(axis='columns').values
        if dim == "arousal":
            self.mean_arousal_gt = np.mean(df, axis=-1)# Mean set
            self.std_arousal_gt  = np.std(df, axis=-1) # STD set
            self.var_arousal_gt  = np.var(df, axis=-1) # Variance set
        else:
            self.mean_valence_gt = np.mean(df, axis=-1)# Mean set
            self.std_valence_gt  = np.std(df, axis=-1) # STD set
            self.var_valence_gt  = np.var(df, axis=-1) # Variance set

    def set_group_affect(self, arousal_df, valence_df):
        # Setting All annotations
        self.arousal = arousal_df
        self.valence = valence_df
        # Setting GT averaged annotations
        self.set_groundtruth_group_affect(arousal_df, "arousal")
        self.set_groundtruth_group_affect(valence_df, "valence")
        
    def load_group_affect(self):
        arousal_df, valence_df = self.get_group_affect()
        self.set_group_affect(arousal_df, valence_df)
    ##########################################################

    # Group-level Features Calculation
    ## Aggregation Based
    def extract_aggregate_group_feats(self, ftype, agg_type=np.mean, temporal_agg=True, normalize=False):
        assert getattr(self.participants[0], ftype) is not None, "Feature Type not yet loaded for the interaction."
        stacked_participants_feats = None
        for participant in self.participants:
            part_feats = participant.get_feature(ftype)
            num_segments, feat_dims = part_feats.shape[0], part_feats.shape[1]
            if temporal_agg: 
                part_feats = participant.calculate_segment_feats(part_feats, modality=self.get_ftype_modality(ftype))
            if normalize:
                part_feats = (part_feats - np.mean(part_feats))/np.std(part_feats)
            part_feats = np.expand_dims(part_feats, axis=0)
            stacked_participants_feats = data_util.concat_np(stacked_participants_feats, part_feats)

        if agg_type.__name__ != "concatenate":
            agg_feats = agg_type(stacked_participants_feats, axis=0)
            return agg_feats
        else:
            return stacked_participants_feats
    
    def get_aggregate_group_feats(self, ftype, agg_type="mean"):
        feats_dict = self.audio_feats if self.get_ftype_modality(ftype) == "audio" else self.video_feats
        return feats_dict[agg_type+"_"+ftype] #getattr(self, agg_type+"_"+ftype)
    
    def set_aggregate_group_feats(self, feats, ftype, agg_type="mean", sub_feat=None):
        feats_dict = self.audio_feats if self.get_ftype_modality(ftype) == "audio" else self.video_feats
        if sub_feat is None:
            feats_dict[agg_type+"_"+ftype] = feats
        else:
            feats_dict[agg_type+"_"+ftype+"_"+sub_feat] = feats
    
    def get_group_feats(self, ftype, feature, sub_feature, modality):
        feature_key = ftype+"_"+feature+"_"+sub_feature if sub_feature != '' else ftype+"_"+feature
        feats_dict = self.audio_feats if modality == "audio" else self.video_feats
        if feature_key in feats_dict.keys():
            return feats_dict[feature_key] #getattr(self, feature_key)
        else:
            feats_path = self.get_group_features_path(modality, ftype=feature_key, ext="csv", synconv=True)
            feats = data_util.read_feats(feats_path, "csv")
            return feats
        
    def set_group_feats(self, feats, feature_key, modality):
        feats_dict = self.audio_feats if modality == "audio" else self.video_feats
        feats_dict[feature_key] = feats # setattr(self, ftype, feature)
        