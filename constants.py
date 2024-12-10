
import numpy as np
import utilities.calculation as calc

MEMO_AUDIO_SR = 22050 #48000 # in Hz
MEMO_AUDIO_FEATEXTRACT_SR = 16000 # in Hz

MEMO_VIDEO_SR = 25 # in fps
MEMO_ANNOTATION_SEG_SIZE = 15 # in secs

GROUP_VIDEO_RESOLUTION = (1280, 720)

FEATURES_INFO = {
    "pitch": {
        "modality": "audio",
        "ext": "npy",
        "extracted_at_group": False
    },
    "vggish": {
        "modality": "audio",
        "ext": "npy",
        "extracted_at_group": False
    },
    "mfcc": {
        "modality": "audio",
        "ext": "npy",
        "extracted_at_group": False
    },
    "intensity": {
        "modality": "audio",
        "ext": "npy",
        "extracted_at_group": False
    },
    "resnet50": {
        "modality": "video",
        "ext": "npy",
        "extracted_at_group": False
    },
    "aucs": {
        "modality": "video",
        "ext": "csv",
        "extracted_at_group": False
    },
    "emotions": {
        "modality": "video",
        "ext": "csv",
        "extracted_at_group": False
    },
    "facepose": {
        "modality": "video",
        "ext": "csv",
        "extracted_at_group": False
    },
}

MFCC_FEATURES = ["0","1","2","3","4"]
AUCS_FEATURES = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU11","AU12","AU14","AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43"]
FACEPOSE_FEATURES = ["headpitch", "headroll", "headyaw"]
EMOTIONS_FEATURES = ["emo_anger", "emo_disgust", "emo_fear", "emo_happiness", "emo_sadness", "emo_surprise", "emo_neutral"]

VIDEO_FEATURES = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU11","AU12","AU14","AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43",\
                    "head_pitch", "head_roll", "head_yaw",\
                    "emo_anger", "emo_disgust", "emo_fear", "emo_happiness", "emo_sadness", "emo_surprise", "emo_neutral"]

SYNCHRONY_FEATURES = ['synchrony_corrcoeff', 'synchrony_maxcorr', 'synchrony_tmax']
CONVERGENCE_FEATURES = ['convergence_global', 'convergence_symmetric', 'convergence_asymmetric']
GROUP_AGGS = ['mean', 'std', 'min', 'max', 'grad', 'median']

AUDIO_FEATURES = []

groupfeat_aggs = {
                    "mean": np.mean, 
                    "std": np.std, 
                    "min": np.amin, 
                    "max": np.amax, 
                    "grad": calc.gradient,
                    "median": np.median
                }

SUBFEATS = {
                'mfcc':MFCC_FEATURES,
                'aucs':AUCS_FEATURES,
                'facepose':FACEPOSE_FEATURES
            }
