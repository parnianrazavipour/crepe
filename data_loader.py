import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json

class MDBStemSynth(Dataset):
    def __init__(self, root: str, split: str = "train", step_size: int = 10, 
                 transform: Optional[callable] = None, 
                 use_interpolation: str = 'none',
                 segment_length: int = 100,
                 min_non_empty_frames: int = 50,
                 fmin = 32.70, 
                 fmax = 1975.5, 
                 num_of_bins = 360, 
                 fref = 10.0):
        assert use_interpolation in ['none', 'gaussian', 'sinc'], f"Invalid interpolation method: {use_interpolation}"
        self.root = root
        self.split = split
        self.sr = 16000  
        self.frame_length = 1024  
        self.step_size = step_size 
        self.hop_length = int(self.sr * self.step_size / 1000) 
        self.transform = transform
        self.fref = fref
        self.fmin = fmin
        self.fmax = fmax
        self.cent_min = 1200 * np.log2(self.fmin / self.fref)
        self.cent_max = 1200 * np.log2(self.fmax / self.fref)
        self.cent_bins = np.linspace(self.cent_min, self.cent_max, num_of_bins)
        print(f"Cent Resolution: {self.cent_bins[1] - self.cent_bins[0]:.2f} cents") 
        
        self.use_interpolation = use_interpolation
        self.segment_length = segment_length
        self.min_non_empty_frames = min_non_empty_frames

        self.meta = self.load_metadata()
        self.non_empty_indices = self.load_or_create_non_empty_indices()

    def load_or_create_non_empty_indices(self):
        """We attempt to create a list of non empty indices, saved at the root of the dataset."""
        indices_file = os.path.join(self.root, f'non_empty_indices_{self.split}.json')
        if os.path.exists(indices_file):
            with open(indices_file, 'r') as f:
                return json.load(f)
        else:
            indices = self.create_non_empty_indices()
            with open(indices_file, 'w') as f:
                json.dump(indices, f)
            with open(indices_file, 'r') as f:
                return json.load(f)
            return indices

    def create_non_empty_indices(self):
        indices = {}
        print("Indexing the dataset for non-empty segments. This will only run once.")
        for idx, audio_name in tqdm(enumerate(self.meta['audio_name']), desc="Creating non-empty indices", total=len(self.meta['audio_name'])):
            audio_path = os.path.join(self.root, 'audio_stems', audio_name)
            frames, _ = self.load_audio_frames(audio_path)
            non_empty = torch.any(frames != 0, dim=1)
            non_empty_ranges = self.get_continuous_ranges(non_empty)
            valid_ranges = [r for r in non_empty_ranges if r[1] - r[0] >= self.min_non_empty_frames]
            indices[idx] = valid_ranges
        return indices

    @staticmethod
    def get_continuous_ranges(bool_tensor):
        if not isinstance(bool_tensor, torch.Tensor):
            bool_tensor = torch.tensor(bool_tensor)
        
        ranges = []
        change_points = []
        
        for i in range(0, len(bool_tensor)):
            if bool_tensor[i] != bool_tensor[i - 1] and bool_tensor[i]: # we have a change point that's a valid start
                change_points.append(i)
                if len(change_points) == 2: # we have a valid range
                    ranges.append(change_points)
                    change_points = []

        return ranges

    def __len__(self):
        return len(self.meta['audio_name'])

    def __getitem__(self, idx):
        audio_name = self.meta['audio_name'][idx]
        f0_name = self.meta['f0_name'][idx]

        audio_path = os.path.join(self.root, 'audio_stems', audio_name)
        f0_path = os.path.join(self.root, 'annotation_stems', f0_name)
        
        start_time, duration = self.get_segment_time(idx)

        frames, frame_times = self.load_audio_frames(audio_path, start_time=start_time, duration=duration)
        targets, ground_truth_pitch = self.load_f0_targets(f0_path, frame_times)

        frames, targets = self.random_crop_or_pad(frames, targets)
        ground_truth_pitch = F.pad(ground_truth_pitch, (0, targets.shape[0] - ground_truth_pitch.shape[0]))

        if self.transform:
            frames = self.transform(frames)

        return {'audio': frames, 'f0_values': targets, 'ground_truth_pitch': ground_truth_pitch}

    def get_segment_time(self, idx):
        valid_ranges = self.non_empty_indices[str(idx)]
        if not valid_ranges or len(valid_ranges) == 0:
            return None, None

        start, end = random.choice(valid_ranges)
        segment_duration = self.segment_length * self.hop_length / self.sr

        if end - start < self.segment_length:
            pad_left = random.randint(0, self.segment_length - (end - start))
            start = max(0, start - pad_left)

        if end - start > self.segment_length:
            start = random.randint(start, end - self.segment_length)
        
        start_time = start * self.hop_length / self.sr
        return start_time, segment_duration

    def load_f0_targets(self, f0_path, frame_times):
        f0_data = pd.read_csv(f0_path, header=None)
        f0_times = f0_data.iloc[:, 0].values
        f0_values = f0_data.iloc[:, 1].values

        f0_interp = np.interp(frame_times, f0_times, f0_values, left=0.0, right=0.0)
        f0_cents = hz_to_cents(f0_interp)

        if self.use_interpolation == 'none':
            targets = self.create_one_hot_targets(f0_cents)
        elif self.use_interpolation == 'gaussian':
            targets = self.create_gaussian_blurred_targets(f0_cents)
        elif self.use_interpolation == 'sinc':
            targets = self.create_sinc_interpolated_targets(f0_cents)
        else:
            raise ValueError(f"Invalid interpolation method: {self.use_interpolation}")

        return targets, torch.tensor(f0_interp, dtype=torch.float32)

    def sample_non_empty_segment(self, idx, frames, targets, ground_truth_pitch):
        valid_ranges = self.non_empty_indices[str(idx)]
        if not valid_ranges or len(valid_ranges) == 0:
            if frames.shape[0] <= self.segment_length:
                return frames, targets, ground_truth_pitch
            start = random.randint(0, frames.shape[0] - self.segment_length)
            end = start + self.segment_length
            return frames[start:end], targets[start:end], ground_truth_pitch[start:end]

        start, end = random.choice(valid_ranges)
        segment_length = min(self.segment_length, end - start)

        if segment_length < self.segment_length:
            pad_left = random.randint(0, self.segment_length - segment_length)
            pad_right = self.segment_length - segment_length - pad_left
            start = max(0, start - pad_left)
            end = min(frames.shape[0], end + pad_right)

        if end - start > self.segment_length:
            start = random.randint(start, end - self.segment_length)
        end = start + self.segment_length

        return frames[start:end], targets[start:end], ground_truth_pitch[start:end]

    def random_crop_or_pad(self, frames, targets):
        if frames.shape[0] > self.segment_length:
            start = torch.randint(0, frames.shape[0] - self.segment_length + 1, (1,)).item()
            end = start + self.segment_length
            return frames[start:end], targets[start:end]
        elif frames.shape[0] < self.segment_length:
            pad_length = self.segment_length - frames.shape[0]
            pad_start = torch.randint(0, pad_length + 1, (1,)).item()
            pad_end = pad_length - pad_start

            padded_frames = F.pad(frames, (0, 0, pad_start, pad_end))
            padded_targets = F.pad(targets, (0, 0, pad_start, pad_end))

            return padded_frames, padded_targets
        else:
            return frames, targets

    def load_metadata(self):
        """
        Load metadata for the MDBStemSynth dataset, such as file names.
        """
        meta = {"audio_name": [], "f0_name": []}
        
        annotation_dir = os.path.join(self.root, 'annotation_stems')
        for file in os.listdir(annotation_dir):
            if file.endswith(".csv") and not file.startswith("._"):
                meta["f0_name"].append(file)
                audio_file = file.replace(".csv", ".wav")
                meta["audio_name"].append(audio_file)
        return meta

    def load_audio_frames(self, path, start_time=0, duration=None):
        """
        Load a segment of audio, pad it, extract overlapping frames, and normalize.
        """
        if duration is None and start_time is None:
            start_time = 0
            duration = 10
        audio, sr = librosa.load(path, sr=self.sr, mono=True, offset=start_time, duration=duration)

        audio = np.pad(audio, self.frame_length // 2, mode='constant')
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length).T
        frames = frames.copy()

        frames -= np.mean(frames, axis=1, keepdims=True)
        frames /= np.clip(np.std(frames, axis=1, keepdims=True), 1e-8, None)

        frame_times = (np.arange(frames.shape[0]) * self.hop_length) / self.sr

        return torch.tensor(frames, dtype=torch.float32), frame_times

    def create_one_hot_targets(self, f0_cents):
        """Create one-hot encoded targets."""
        bin_indices = np.digitize(f0_cents, self.cent_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.cent_bins) - 1)

        targets = torch.zeros((len(f0_cents), len(self.cent_bins)), dtype=torch.float32)
        for i, (f0, bin_idx) in enumerate(zip(f0_cents, bin_indices)):
            if f0 > 0:  
                targets[i, bin_idx] = 1.0
        return targets

    def create_gaussian_blurred_targets(self, f0_cents, std_dev_cents=25):
        """Create Gaussian-blurred targets using PyTorch."""
        one_hot_targets = self.create_one_hot_targets(f0_cents)
        
        cents_per_bin = self.cent_bins[1] - self.cent_bins[0]
        std_dev_bins = std_dev_cents / cents_per_bin
        
        kernel_size = int(6 * std_dev_bins)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        gaussian_kernel = torch.exp(-torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)**2 / (2 * std_dev_bins**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        blurred_targets = F.conv1d(
            one_hot_targets.unsqueeze(1),
            gaussian_kernel.view(1, 1, -1),
            padding=kernel_size//2
        ).squeeze(1)
        
        sum_values = blurred_targets.sum(dim=1, keepdim=True)
        blurred_targets = blurred_targets / torch.where(sum_values > 0, sum_values, torch.ones_like(sum_values))
        normalized_targets = blurred_targets / blurred_targets.sum(dim=1, keepdim=True)
        
        return blurred_targets

    def create_sinc_interpolated_targets(self, f0_cents, width_factor=5):
        """Create sinc-interpolated targets."""
        f0_cents = torch.tensor(f0_cents, dtype=torch.float32)
        cent_bins = torch.tensor(self.cent_bins, dtype=torch.float32)
        bin_width = cent_bins[1] - cent_bins[0]
        
        distances = (f0_cents.unsqueeze(1) - cent_bins.unsqueeze(0)) / bin_width
        
        x = torch.pi * distances * width_factor
        sinc_values = torch.where(x == 0, torch.tensor(1.0), torch.sin(x) / x)
        
        envelope = torch.exp(-0.5 * (distances / width_factor)**2)
        sinc_values *= envelope

        normalized_values = sinc_values / torch.sum(sinc_values, dim=1, keepdim=True)
        
        normalized_values = torch.where(normalized_values > 0.0, normalized_values, torch.zeros_like(normalized_values))
        
        return normalized_values
    
    def visualize_interpolations(self, idx):
        audio_name = self.meta['audio_name'][idx]
        f0_name = self.meta['f0_name'][idx]

        audio_path = os.path.join(self.root, 'audio_stems', audio_name)
        f0_path = os.path.join(self.root, 'annotation_stems', f0_name)

        frames, frame_times = self.load_audio_frames(audio_path)
        f0_cents = self.load_f0_cents(f0_path, frame_times)
        frames, f0_cents = self.sample_non_empty_segment(idx, frames, f0_cents)

        one_hot_targets = self.create_one_hot_targets(f0_cents)
        gaussian_targets = self.create_gaussian_blurred_targets(f0_cents)
        sinc_targets = self.create_sinc_interpolated_targets(f0_cents)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        im1 = ax1.imshow(one_hot_targets.T, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('One-Hot Encoding')
        ax1.set_xlabel('Time Frame')
        ax1.set_ylabel('Cent Bin')
        plt.colorbar(im1, ax=ax1, label='Probability')

        im2 = ax2.imshow(gaussian_targets.T, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Gaussian Interpolation')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Cent Bin')
        plt.colorbar(im2, ax=ax2, label='Probability')

        im3 = ax3.imshow(sinc_targets.T, aspect='auto', origin='lower', cmap='viridis')
        ax3.set_title('Sinc Interpolation')
        ax3.set_xlabel('Time Frame')
        ax3.set_ylabel('Cent Bin')
        plt.colorbar(im3, ax=ax3, label='Probability')

        plt.tight_layout()
        return fig
    
    def load_f0_cents(self, f0_path, frame_times):
        """
        Load F0 annotations and convert to cents aligned with frame times.
        """
        f0_data = pd.read_csv(f0_path, header=None)
        f0_times = f0_data.iloc[:, 0].values
        f0_values = f0_data.iloc[:, 1].values

        f0_interp = np.interp(frame_times, f0_times, f0_values, left=0.0, right=0.0)
        return hz_to_cents(f0_interp)
    
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

if __name__ == "__main__":
    dataset = MDBStemSynth(root="/root/MDB-stem-synth", split="train", step_size=10, segment_length=500, num_of_bins=128)
    
    print(f"Number of audio files: {len(dataset)}")
    random_idx = random.randint(0, len(dataset) - 1)
    fig = dataset.visualize_interpolations(random_idx)
    
    plt.savefig('interpolation_comparison.png')
    plt.close(fig)
    
    print(f"Visualization saved as 'interpolation_comparison.png'")







