"""
.. moduleauthor:: Navin Raj Prabhu
"""

from speechbrain.pretrained import SepformerSeparation as separator

import torchaudio
import torch

class SepFormer():
    
    def __init__(self, model_type='sepformer', pretrain_dataset='wsj02mix', device='cuda'):
        # pretrain_dataset - libri3mix or wsj02mix
        
        
        self.device = {"device":device,}
                    #    'distributed_launch':True,
                    #    'data_parallel_count':-1,
                    #    'data_parallel_backend':True} 
        self.pretrain_dataset = "speechbrain/"+model_type+"-" + pretrain_dataset
        self.model = separator.from_hparams(source=self.pretrain_dataset,
                                            savedir='pretrained_models/'+model_type+'-'+pretrain_dataset,
                                            run_opts=self.device)
        self.req_sr = 8000
        self.num_wavsplits = 16
        

    def _resample_audio(self, wav, orig_sample_rate, req_sample_rate):
        # print("Resampling ....")
        transform = torchaudio.transforms.Resample(orig_sample_rate, req_sample_rate)
        waveform = transform(wav)
        return waveform
    
    def separate_audio(self, wav_file):
        waveform, sample_rate = torchaudio.load(wav_file, normalize=True)
        if waveform.shape[0]>1: # If True, audio is Stereo
            waveform = torch.mean(waveform, dim=0).unsqueeze(0) # Stereo to Mono
        if sample_rate != self.req_sr:
            waveform = self._resample_audio(waveform, sample_rate, self.req_sr)
        print("Audio Duration - ", int((waveform.shape[-1]/self.req_sr)/60), " mins",\
             int((waveform.shape[-1]/self.req_sr)%60), " secs")
        # print("Separtion In progress .... of audio shape ", waveform.shape)
        waveform = torch.stack(list(torch.tensor_split(waveform, self.num_wavsplits, dim=-1)), dim=0)
        print("After Split  - ", waveform.shape)
        sep_audio = None
        for i in range(waveform.shape[0]):
            curr_audio = waveform[i, :, :]
            # print("e.g. Sepformer ip - ", curr_audio.shape)
            curr_sep_audio = self.model.separate_batch(mix=curr_audio) 
            sep_audio = curr_sep_audio if sep_audio is None else torch.cat((sep_audio, curr_sep_audio), 1)
        # sep_audio = self.model.separate_batch(mix=waveform) 
        num_audio = sep_audio.shape[-1] # Number of est. speakers
        print("Sep Audio - ", sep_audio.shape)
        print("Num of Est. Speakers - ", num_audio)
        print("Num of Audio Segments- ", sep_audio.shape[0])

        return sep_audio, num_audio
        