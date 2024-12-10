"""
.. moduleauthor:: Navin Raj Prabhu
"""

import numpy
import torch
import torchaudio
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

print(torch.__version__)

class MossFormer():
    # Note : Not working for large audio (not able to send loaded wav as input)
    # Workaround - split audio - save separately - then separte and the load and cocate- TOO MUCH Work!!
    def __init__(self):
        
        self.model = pipeline(Tasks.speech_separation,
                              model='damo/speech_mossformer_separation_temporal_8k')
        self.req_sr = 8000
        self.num_wavsplits = 16
        

    def _resample_audio(self, wav, orig_sample_rate, req_sample_rate):
        print("Resampling ....")
        transform = torchaudio.transforms.Resample(orig_sample_rate, req_sample_rate)
        waveform = transform(wav)
        return waveform
    
    def separate_audio(self, wav_file):
        # waveform, sample_rate = torchaudio.load(wav_file, normalize=True)
        # if waveform.shape[0]>1: # If True, audio is Stereo
        #     waveform = torch.mean(waveform, dim=0)#.unsqueeze(0) # Stereo to Mono
        # print("Initial waveform shape - ", waveform.shape)
        # if sample_rate != self.req_sr:
        #     waveform = self._resample_audio(waveform, sample_rate, self.req_sr)
        # print("Audio Duration - ", int((waveform.shape[-1]/self.req_sr)/60), " mins",\
        #      int((waveform.shape[-1]/self.req_sr)%60), " secs")
        # print("Separtion In progress .... of audio shape ", waveform.shape)
        # waveform = torch.stack(list(torch.tensor_split(waveform, self.num_wavsplits, dim=0)), dim=0)
        # print("After Split  - ", waveform.shape)
        # sep_audio = None
        sep_audio = self.model(wav_file)
        num_audio = len(sep_audio['output_pcm_list']) # Number of est. speakers
        print("Num of Est. Speakers - ", num_audio)
        
        for i, signal in enumerate(sep_audio['output_pcm_list']):
            print('Speaker = ', i)
            print(signal.shape)
            # save_file = f'output_spk{i}.wav'
            # sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
        # for i in range(waveform.shape[0]):
        #     curr_audio = waveform[i, :].unsqueeze(0).numpy()
        #     print("e.g. MOSS Former ip - ", curr_audio.shape)
        #     # TODO - Check model output and format make it compaticle with return all_audio, num_audio
        #     curr_sep_audio = self.model(curr_audio) 
        #     print("curr_sep_audio Shape  - ", curr_sep_audio.shape)
        #     sep_audio = curr_sep_audio if sep_audio is None else torch.cat((sep_audio, curr_sep_audio), 1)
        
        # # TODO - Check model output and format make it compaticle with return all_audio, num_audio
        # all_audio = None
        # for i, signal in enumerate(sep_audio['output_pcm_list']):
        #     all_audio = signal if all_audio is None else torch.cat((all_audio, signal), axis=0)

        return signal, num_audio
