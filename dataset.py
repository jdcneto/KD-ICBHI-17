import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset
from torch.distributions.beta import Beta


class SoundDataset(TorchDataset):
    def __init__(self, df,  audiopath, mixup=False, beta=2, rate=0.5):
        """
        Reads the wave file from and returns a array and It's respective class
        """
        self.df = df
        self.audiopath = audiopath
        self.mixup = mixup
        self.beta = beta
        self.rate = rate

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Returns:
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]
        wave_path = os.path.join(self.audiopath, row.filename)
        waveform, _ = torchaudio.load(wave_path)
        target = row.id

        if self.mixup and (torch.rand(1) < self.rate):
            idx = torch.randint(len(self.df), (1,)).item()
            row = self.df.iloc[idx]
            wave_path2 = os.path.join(self.audiopath, row.filename)
            waveform2, _ = torchaudio.load(wave_path2)
            target2 = row.id
            b_sample = Beta(self.beta, self.beta)
            l = b_sample.sample().item()
            l = max(l, 1. - l)
            waveform = (waveform * l + waveform2 * (1. - l))
            target = target * l + target2 * (1. - l)

        return waveform.squeeze(0), target