import os
import torch
import numpy as np
from tqdm import tqdm
from torchvggish import vggish, vggish_input

import feature_extractor.audio.utils as utils
from utilities.dataset import get_MEMO_dataset
import constants as c


def instantiate_vggish():
    # Initialise model and download weights
    model = vggish()
    model.eval()
    return model

def extract_vggish_feats(model, wav_segment):
    embeddings = model.forward(wav_segment).detach().numpy()
    embeddings = np.mean(embeddings, axis=0)
    return embeddings


def extractor():
    
    memoObj = get_MEMO_dataset("raw")
    memoInteractions = memoObj.interactions_df
    vgg_model = instantiate_vggish() 

    for interaction in memoInteractions:
        for participant in interaction.participants:
            # Folder Orga.
            outfile = participant.get_features_path("audio_v2", "vggish")
            os.makedirs("/".join(outfile.split("/")[:-1]), exist_ok=True)
            if not os.path.exists(outfile):
                print("Extracting VGGish for Interaction: ", interaction.id + " and Participant: ", participant.id) 
                wav_path = participant.get_audio_path()     
                           
                full_waveform = vggish_input.wavfile_to_waveform(wav_path)
                ############## HARDCODED BUG FIXES ###################################################
                # Wrong audio cuts fix for interactions 11_2 and 5_3
                if interaction.id == "group11_session2":
                    full_waveform = full_waveform[:(3105*c.MEMO_AUDIO_SR)] # TODO: in secs i.e., until 51:45 [note annotatiosn are until 52:00, last segment to be omitted everywhere]
                if interaction.id == "group5_session3":
                    full_waveform = full_waveform[:(2685*c.MEMO_AUDIO_SR)]  # TODO: in secs i.e., until 51:45 [note annotatiosn are until 52:00, last segment to be omitted everywhere]
                ############## HARDCODED BUG FIXES ###################################################
                segmented_waveforms = utils.batch_as_segments(full_waveform, c.MEMO_ANNOTATION_SEG_SIZE, c.MEMO_AUDIO_SR, "audio")
                segmented_features = []
                for i, segment in tqdm(enumerate(segmented_waveforms)):
                    segmented_example = vggish_input.waveform_to_examples(segment, c.MEMO_AUDIO_SR, return_tensor=True)
                    seg_feat = extract_vggish_feats(vgg_model, segmented_example)
                    segmented_features.append(seg_feat)

                print("no.of.segments = ", len(segmented_features), ", VGGish Shape - ", seg_feat.shape)
                segmented_features = np.stack(segmented_features, axis=0 )
                
                np.save(outfile, segmented_features)
            
def main():
    print('Initializing VGGish extraction Process..')
    extractor()
    
if __name__ == '__main__':
    main()