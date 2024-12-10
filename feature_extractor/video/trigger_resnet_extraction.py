import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm

from feature_extractor.video import extract_resnet as resnet
from utilities.dataset import get_MEMO_dataset
import utilities.data as data_utils
import constants as c

FEATURE_TYPE = 'resnet50'
BATCH_SIZE = 1 # in secs

def extractor():
    
    memoObj = get_MEMO_dataset("raw")
    memoInteractions = memoObj.interactions_df

    model, feat_extractor = resnet.instantiate_resnet(variant=FEATURE_TYPE)

    for interaction in memoInteractions:
        for participant in interaction.participants:
            # Folder Orga.
            outfile = participant.get_features_path("video", FEATURE_TYPE)
            os.makedirs("/".join(outfile.split("/")[:-1]), exist_ok=True)
            if not os.path.exists(outfile):
                print("Extracting ", FEATURE_TYPE, " for Interaction: ", interaction.id + " and Participant: ", participant.id) 
                video_path = participant.get_video_path()     

                video = cv2.VideoCapture(video_path)
                length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)
                print("Number of video frames = ", length)
                print("Frame Rate = ", fps, "fps")
                print("I/P Batch SIZE = ", int(fps*BATCH_SIZE))
                
                count = 0
                batched_image = None
                video_embedding = None
                while video.isOpened():
                    # print(count, "/", length)
                    success, image = video.read()
                    if success:
                        start_time = time.time()
                        processed_image = resnet.preproc_image_for_resnet(torch.permute(torch.from_numpy(image), (2, 0, 1)))
                        batched_image = data_utils.concat_tensor(batched_image, processed_image, axis=0)
                        if batched_image.shape[0] == int(fps*BATCH_SIZE): # Extract Features
                            # print("resnet input = ", batched_image.shape)
                            batched_embed = resnet.extract_resnet_feats(feat_extractor, batched_image).detach().numpy()
                            video_embedding = data_utils.concat_np(video_embedding, batched_embed, axis=0)
                            batched_image = None
                            print("Video Emnbedding: ", video_embedding.shape)  
                            print("Time Taiken for 1 batch = ", time.time() - start_time)  
                            start_time = time.time()                    
                    else:
                        break
                    count += 1
                
                cv2.destroyAllWindows()
                video.release()
                
                print("Final embedding Shape of Video : ", video_embedding.shape)
                np.save(outfile, video_embedding)
            
def main():
    print('Initializing ResNet extraction Process..')
    extractor()
    
if __name__ == '__main__':
    main()