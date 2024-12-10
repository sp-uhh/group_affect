import os
import cv2
import time
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from feat import Detector

from utilities.dataset import get_MEMO_dataset
import utilities.data as data_utils
import constants as c

# FEATURE_TYPE = 'aucs'
# BATCH_SIZE = 1 # in secs
# SKIP_FRAMES = 25 #in nframes


AUC_HEADER = ["AU"+auc for auc in ["01","02","04","05","06","07","09","10","11","12","14","15","17","20","23","24","25","26","28","43"]]
POSE_HEADER = ["head_pitch", "head_roll", "head_yaw"]
EMOTION_HEADER = ["emo_anger", "emo_disgust", "emo_fear", "emo_happiness", "emo_sadness", "emo_surprise", "emo_neutral"]



def get_faces_landmarks(detector: Detector, image):
    detected_face = detector.detect_faces(image)
    # assert len(detected_face[0]) == 1, "Detected wrong number of faces. !!!!! Warning !!!!!" + " No.of Faces = " + str(len(detected_face[0]))\
    detected_landmark = detector.detect_landmarks(image, detected_face)
    return detected_face, detected_landmark

def handle_multiple_faces(faces, image_feats):
    if len(faces[0]) > 1:
        # image_feats = image_feats.iloc[0] #take 1st face's feats
        image_feats = image_feats.head(1)
    return image_feats

def extract_facial_expressions(detector: Detector, image, landmarks, faces, findex, ftype='aucs'):
    if ftype == "aucs":
        feats = detector.detect_aus(image, landmarks)[0]
        feats_df = pd.DataFrame(data=feats,
                                columns = AUC_HEADER,
                                index=[findex for i in range(len(feats))])
    elif ftype == "emotions":
        feats = detector.detect_emotions(image, faces, landmarks)[0]
        feats_df = pd.DataFrame(data=feats,
                                columns = EMOTION_HEADER,
                                index=[findex for i in range(feats.shape[0])])
    elif ftype == "facepose":
        feats = np.array(detector.detect_facepose(image)["poses"][0])
        feats_df = pd.DataFrame(data=feats,
                                columns = POSE_HEADER,
                                index=[findex for i in range(feats.shape[0])])
    
    feats_df = handle_multiple_faces(faces, feats_df)
    return feats_df


def extractor(feature_type, skip_frames, batch_size):
    
    no_faces = [] 
    
    memoObj = get_MEMO_dataset("raw")
    memoInteractions = memoObj.interactions_df
    detector = Detector(face_model="img2pose",
                        landmark_model="mobilefacenet",
                        au_model="xgb",
                        facepose_model="img2pose",
                        emotion_model="resmasknet")

    for interaction in memoInteractions:
        for participant in interaction.participants:
            # Folder Orga.
            outfile = participant.get_features_path("video", feature_type, ext="csv")
            print("Output File = ", outfile)
            os.makedirs("/".join(outfile.split("/")[:-1]), exist_ok=True)
            if not os.path.exists(outfile):
                print("Extracting ", feature_type, " for Interaction: ", interaction.id + " and Participant: ", participant.id) 
                video_path = participant.get_video_path()     
                print(video_path)
                video = cv2.VideoCapture(video_path)
                length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)
                print("Number of video frames = ", length)
                print("Frame Rate = ", fps, "fps")
                
                count=0
                video_feats = None
                while video.isOpened():
                    success, image = video.read()
                    if success:
                        if count !=0 and (count+1)%skip_frames == 0:
                            faces, landmark = get_faces_landmarks(detector, image)
                            print(count, "/", length, ", #faces = ", len(faces[0]))
                            if len(faces[0]) != 0:  # No faces? Use previous face features
                                image_feats = extract_facial_expressions(detector,
                                                                            image, landmark, faces,
                                                                            count+1,
                                                                            ftype=feature_type)
                            else:
                                print("No Faces detected !!!!!!!!!!!!!!!!!")
                                no_faces.append(interaction.id+"_"+participant.id+":"+str(count))
                                # image_feats.set_index([count+1],inplace=True)
                            video_feats = data_utils.concat_pd(video_feats, image_feats)
                        count = count + 1
                    else:
                        break
                    
                cv2.destroyAllWindows()
                video.release()
                
                print("Final embedding Shape of Video : ", video_feats.shape)
                video_feats.to_csv(outfile)
            
def main():
    
    # feature_type, skip_frames, batch_size
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_type', default="aucs", type=str)
    parser.add_argument('--skip_frames', default=25, type=int)
    parser.add_argument('--batch_size', default=1, type=int)

    a = parser.parse_args()
    feature_type = a.feature_type
    skip_frames = int(a.skip_frames)
    batch_size = int(a.batch_size)
    
    print('Initializing ', feature_type ,' extraction Process..')
    
    extractor(feature_type, skip_frames, batch_size)
    
if __name__ == '__main__':
    main()