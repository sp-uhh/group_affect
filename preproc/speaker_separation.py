from pathlib import Path
import torchaudio
import argparse
import glob
import sys
import os
sys.path.insert(0, os.getcwd()+'/../groupsync/')

from readers.memo_ga import MEMOGroupAff

from speechsep import sepformer as sp
from speechsep import mossformer as ms

model_type = 'sepformer' # or 'sepformer'
sepformer_data = 'whamr' # or 'wsj02mix'
device = 'cuda'

team_prefix = 'CfC_Team'
# teams = ['01', '02', '03', '04', '05'] #['06', '07', '08', '09', '10']

def inst_model(model):
    if model == 'sepformer':
        return instantiate_sepformer()
    elif model == 'mossformer':
        return instantiate_mossformer()
    return None

def instantiate_sepformer():
    sep_model = sp.SepFormer(model_type=model_type, pretrain_dataset=sepformer_data, device=device)
    return sep_model

def instantiate_mossformer():
    sep_model = ms.MossFormer()
    return sep_model

def separate_spearkers_for_dataset(teams, dataset_folder, sr=16000):
    sep_model = inst_model(model_type) #instantiate_sepformer()
    # for dir in os.listdir(dataset_folder):
    for team in teams:
        wav_path = dataset_folder+team_prefix+team+'/wav_'+str(sr)+'/cut_wav/'
        sepwav_path = os.path.join(wav_path, "separated")
        Path(sepwav_path).mkdir(parents=True, exist_ok=True)
        for wav in glob.glob(os.path.join(wav_path, '*.wav')):
            if "Dome" not in wav:
                fname = wav.split("/")[-1].split('.')[0]         
                print("src wav - ", wav)
                sep_audio, num_spkrs = sep_model.separate_audio(wav)
                for spkr in range(num_spkrs): 
                    print("Savinfg @", sepwav_path+"/"+fname+"_"+str(spkr)+".wav")          
                    torchaudio.save(sepwav_path+"/"+fname+"_"+str(spkr)+".wav", sep_audio[:, :, spkr].detach().cpu(), 8000)


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--teams', default=True, type=str)
    a = parser.parse_args()
    
    memodb = MEMOGroupAff()
    audio_path = memodb.audios_folder
    teams = [str(item) for item in a.teams.split(',')]
    separate_spearkers_for_dataset(teams, audio_path, sr=8000)

    
    
if __name__ == '__main__':
    main()