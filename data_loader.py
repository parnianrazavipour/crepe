# data_loader.py

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from typing import Optional

class MDBStemSynth(Dataset):
    """
    MDB-stem-synth dataset class for audio and corresponding pitch (F0) annotation loading.
    This dataset consists of monophonic audio tracks with accurate pitch annotations.
    """

    def __init__(self, root: str, split: str = "train", step_size: int = 10, transform: Optional[callable] = None):
        self.root = root
        self.split = split
        self.sr = 16000  
        self.frame_length = 1024  
        self.step_size = step_size 
        self.hop_length = int(self.sr * self.step_size / 1000) 
        self.transform = transform
        self.fref = 10.0  
        self.cent_bins = self.get_cent_bins() 

        self.meta = self.load_metadata()

        self.frame_indices = self.calculate_frame_indices()

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.frame_indices[idx]

        audio_name = self.meta['audio_name'][file_idx]
        f0_name = self.meta['f0_name'][file_idx]

        audio_path = os.path.join(self.root, 'audio_stems', audio_name)
        f0_path = os.path.join(self.root, 'annotation_stems', f0_name)

        frames, frame_times = self.load_audio_frames(audio_path)
        targets = self.load_f0_targets(f0_path, frame_times)

        audio_frame = frames[frame_idx]
        target = targets[frame_idx]

        if self.transform:
            audio_frame = self.transform(audio_frame)

        audio_frame = torch.tensor(audio_frame, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return {'audio': audio_frame, 'f0_values': target}

    def load_metadata(self):
        """
        Load metadata for the MDB-stem-synth dataset, such as file names.
        """
        meta = {"audio_name": [], "f0_name": []}
        audio_dir = os.path.join(self.root, 'audio_stems')
        for file in os.listdir(audio_dir):
            if file.endswith(".wav"):
                audio_file = file
                f0_file = file.replace(".wav", ".csv")  
                meta["audio_name"].append(audio_file)
                meta["f0_name"].append(f0_file)
        return meta

    def calculate_frame_indices(self):
        """
        Calculate the frame indices for all audio files.
        Returns a list of tuples (file_idx, frame_idx).
        """
        frame_indices = []
        for file_idx, audio_name in enumerate(self.meta['audio_name']):
            audio_path = os.path.join(self.root, 'audio_stems', audio_name)
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            audio_length_samples = len(audio)
            padded_length = audio_length_samples + self.frame_length
            n_frames = 1 + int((padded_length - self.frame_length) / self.hop_length)
            for frame_idx in range(n_frames):
                frame_indices.append((file_idx, frame_idx))
        return frame_indices
    

    def load_audio_frames(self, path):
        """
        Load audio, pad it, extract overlapping frames, and normalize.
        """
        audio, sr = librosa.load(path, sr=self.sr, mono=True)

        audio = np.pad(audio, self.frame_length // 2, mode='constant')
        n_frames = 1 + int((len(audio) - self.frame_length) / self.hop_length)

        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length).T.copy()  # Add .copy() here

        frames -= np.mean(frames, axis=1, keepdims=True)
        frames /= np.clip(np.std(frames, axis=1, keepdims=True), 1e-8, None)

        frame_times = (np.arange(n_frames) * self.hop_length) / self.sr

        return frames, frame_times


    def load_f0_targets(self, f0_path, frame_times):
        """
        Load F0 annotations and create one-hot target vectors aligned with frame times.
        """
        f0_data = pd.read_csv(f0_path, header=None)
        f0_times = f0_data.iloc[:, 0].values
        f0_values = f0_data.iloc[:, 1].values

        f0_interp = np.interp(frame_times, f0_times, f0_values, left=0.0, right=0.0)

        f0_cents = hz_to_cents(f0_interp)

        bin_indices = np.digitize(f0_cents, self.cent_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.cent_bins) - 1)

        targets = np.zeros((len(frame_times), len(self.cent_bins)), dtype=np.float32)
        for i, (f0, bin_idx) in enumerate(zip(f0_interp, bin_indices)):
            if f0 > 0:  
                targets[i, bin_idx] = 1.0
            else:
                pass  

        return targets 
    def get_cent_bins(self):
        """
        Generate 360-cent bins for the pitch range from C1 (32.70 Hz) to B7 (1975.5 Hz).
        """
        f_min = 32.70  # C1
        f_max = 1975.5  # B7
        cent_min = 1200 * np.log2(f_min / self.fref)
        cent_max = 1200 * np.log2(f_max / self.fref)
        return np.linspace(cent_min, cent_max, 360)

def hz_to_cents(f0_hz, fref=10.0):
    """Convert frequency in Hz to cents, relative to a reference frequency."""
    f0_hz = np.maximum(f0_hz, 1e-6)
    return 1200 * np.log2(f0_hz / fref)
