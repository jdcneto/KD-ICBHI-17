import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset
from torch.distributions.beta import Beta


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
        wave_path = os.path.join(self.audiopath, row.filename)
        waveform, _ = torchaudio.load(wave_path)
        target = row.id

        return waveform.squeeze(0), target