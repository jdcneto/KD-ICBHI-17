import io
import os
import pathlib
import random

import librosa
import torchaudio
from torch.utils.data import Dataset as TorchDataset

import torch
from ba3l.ingredients.datasets import Dataset
import pandas as pd
from sacred.config import DynamicIngredient, CMD
from scipy.signal import convolve
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import h5py
from helpers.audiodatasets import  PreprocessDataset


# LMODE = os.environ.get("LMODE", False)

# dataset = Dataset('Esc50')


# @dataset.config
# def default_config():
    # name = 'esc50'  # dataset name
    # normalize = False  # normalize dataset
    # subsample = False  # subsample squares from the dataset
    # roll = True  # apply roll augmentation
    # fold = 1
    # base_dir = "/share/rk7/shared/ESC-50-master/"  # base directory of the dataset as downloaded
    # if LMODE:
        # base_dir = "/system/user/publicdata/CP/audioset/audioset_hdf5s/"
    # meta_csv = base_dir + "meta/esc50.csv"
    # audio_path = base_dir + "audio_32k/"
    # ir_path = base_dir + "irs/"
    # num_of_classes = 50


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, y2 = self.dataset[idx2]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            return x, (y1 * l + y2 * (1. - l))
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class AudioSetDataset(TorchDataset):
    def __init__(self, meta_csv,  audiopath, train=False, sample_rate=32000, classes_num=527):
        """
        Reads the mp3 bytes from HDF file decodes using av and returns a fixed length audio wav
        """
        self.sample_rate = sample_rate
        self.meta_csv = meta_csv
        self.df = pd.read_csv(meta_csv)
        self.sr = sample_rate
        self.classes_num = classes_num
        self.audiopath = audiopath
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.

        Args:
          meta: {
            'hdf5_path': str,
            'index_in_hdf5': int}
        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        row = self.df.iloc[index]

        #waveform = decode_mp3(np.fromfile(self.audiopath + row.filename, dtype='uint8'))
        waveform, _ = torchaudio.load(self.audiopath + row.filename)
        target = row.id
        return waveform, target


@dataset.command
def get_base_training_set(meta_csv, audio_path):
    ds = AudioSetDataset(meta_csv, audio_path,  train=True)
    return ds


@dataset.command
def get_base_test_set(meta_csv, audio_path):
    ds = AudioSetDataset(meta_csv, audio_path, train=False)
    return ds


@dataset.command
def get_training_set(normalize, roll, wavmix=False):
    ds = get_base_training_set()
    get_ir_sample()
    if normalize:
        print("normalized train!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)
    if wavmix:
        ds = MixupDataset(ds)

    return ds


@dataset.command
def get_test_set(normalize):
    ds = get_base_test_set()
    if normalize:
        print("normalized test!")
        fill_norms()
        ds = PreprocessDataset(ds, norm_func)
    return ds
