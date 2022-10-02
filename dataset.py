import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset
from torch.distributions.beta import Beta


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, y2 = self.dataset[idx2]
            b_sample = Beta(self.beta, self.beta)
            l = b_sample.sample().item()
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, (y1 * l + y2 * (1. - l))
            
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SoundDataset(TorchDataset):
    def __init__(self, df,  audiopath):
        """
        Reads the wave file from and returns a array and It's respective class
        """
        self.df = df
        self.audiopath = audiopath
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Returns:
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]

        #waveform
        wave_path = os.path.join(self.audiopath, row.filename)
        waveform, _ = torchaudio.load(wave_path)
        target = row.id
        return waveform.squeeze(0), target


def Loader(df, audio_path, mix_up=True):
    ds = SoundDataset(df, audio_path)
    if mix_up:
        ds = MixupDataset(ds)
    return ds