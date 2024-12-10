import cv2
import os

import constants as c
from readers.utils.audio_utils import addAudioToVideo

VIDEO_RESOLUTION=c.GROUP_VIDEO_RESOLUTION
VIDEO_FPS=c.MEMO_VIDEO_SR

def get_video_proc(video_path):
    input_video = cv2.VideoCapture(video_path)
    # get vcap property 
    # width  = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    # height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # print('width:', width, ", height: ", height)  # float `fps`
    return input_video

def get_video_bbox(config, input_video, participant_id):
    
    layout = config['input_layout']
    participant_loc = config['participant_loc'][participant_id]
    
    width = VIDEO_RESOLUTION[0] #input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = VIDEO_RESOLUTION[1] #input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    row, col = participant_loc[0], participant_loc[1]
    
    window_width = width / int(layout[0])
    window_height = height / int(layout[1])

    x_min, y_min = int(col * window_width), int(row * window_height)
    x_max, y_max = int(x_min + window_width), int(y_min + window_height)
    # Correction for Layout "32"
    if layout == "32":
        if row == 0: # 1st Row Participant
            y_min = y_min + 120
        elif row == 1: # 2nd Row Participant
            y_max = y_max - 120
            
    return (x_min, y_min, x_max, y_max)

def crop_participant(frame, box):
    frame_copy = frame.copy()
    return frame_copy[box[1]:box[3], box[0]:box[2], :] 

def get_participants(config):
    return config["participant_loc"].keys()

def get_boxes(config, input_video, participants):
    boxes = {}
    for participant in participants:
        video_bbox = get_video_bbox(config, input_video, participant)
        boxes[participant] = video_bbox
    return boxes

import time

def get_storage_fpath(config, base_store, group, session, participant):
    store = base_store + "group" + group + "_session" + session + "/"
    os.makedirs(store, exist_ok=True)
    store = store+participant+"_"+config["participant_alias"][participant].lower().strip()+".avi"
    return store
    
def get_writers(config, participants, boxes, base_store, group, session):
    writers = {}
    
    for participant in participants:
        participant_box = boxes[participant]
        
        crop_videoWidth = int(participant_box[2]-participant_box[0])# x_min, y_min, x_max, y_max
        crop_videoHeight = int(participant_box[3]-participant_box[1])# x_min, y_min, x_max, y_max
        store = get_storage_fpath(config, base_store, group, session, participant)
        writers[participant] = cv2.VideoWriter(store, cv2.VideoWriter_fourcc(*'XVID'), VIDEO_FPS, (crop_videoWidth, crop_videoHeight))
        time.sleep(1)
    return writers


def write_participants_cropped_videos(config, video_path, base_store, groupId, sessId):

    cv2.destroyAllWindows()
    participants = get_participants(config)
    
    input_video = get_video_proc(video_path)
    boxes = get_boxes(config, input_video, participants)
    writers = get_writers(config, participants, boxes, base_store, groupId, sessId)
    # print("Boxes: ", boxes)
    length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    nframes = 0
    while(input_video.isOpened()):
        ret,frame = input_video.read()
        if ret == True:
            for participant in participants:
                box = boxes[participant]
                writer = writers[participant]                
                participant_frame = crop_participant(frame, box)    
                writer.write(participant_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
        nframes = nframes + 1
        print(str(nframes)+"/"+str(length))
        
    input_video.release()
    for participant in participants:
        writers[participant].release()
        
    cv2.destroyAllWindows()
    
    return True