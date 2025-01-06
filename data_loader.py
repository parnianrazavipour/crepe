import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def hz_to_cents(frequency_hz, fref=10.0):
    """
    Convert frequency in Hz to cents relative to a reference frequency.
    Handles unvoiced regions (frequency <= 0) by setting them to NaN.
    
    Args:
        frequency_hz (np.ndarray): Array of frequencies in Hz.
        fref (float): Reference frequency in Hz.
        
    Returns:
        np.ndarray: Array of frequencies in cents, with NaN for unvoiced frames.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        frequency_hz = np.where(frequency_hz > 0, frequency_hz, np.nan)
        return 1200 * np.log2(frequency_hz / fref)

class MDBStemSynth(Dataset):
    def __init__(self, root: str, split: str = "train", step_size: int = 10, 
                 transform: Optional[callable] = None, 
                 use_interpolation: str = 'none',
                 segment_length: int = 1,  # Set to 1 for frame-based approach
                 fmin = 32.70, 
                 fmax = 1975.5, 
                 num_of_bins = 360, 
                 fref = 10.0):
        """
        Initialize the MDBStemSynth Dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split ('train', 'validation', 'test').
            step_size (int): Hop size in milliseconds.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_interpolation (str): Target interpolation method ('none', 'gaussian', 'sinc').
            segment_length (int): Number of frames per segment (set to 1 for frame-based).
            fmin (float): Minimum frequency in Hz.
            fmax (float): Maximum frequency in Hz.
            num_of_bins (int): Number of pitch bins.
            fref (float): Reference frequency in Hz for cents calculation.
        """
        assert use_interpolation in ['none', 'gaussian', 'sinc'], f"Invalid interpolation method: {use_interpolation}"
        self.root = root
        self.split = split
        self.sr = 16000  
        self.frame_length = 1024  
        self.step_size = step_size 
        self.hop_length = int(self.sr * self.step_size / 1000)  # e.g., 10ms -> 160 samples
        self.transform = transform
        self.fref = fref
        self.fmin = fmin
        self.fmax = fmax
        self.cent_min = 1200 * np.log2(self.fmin / self.fref)
        self.cent_max = 1200 * np.log2(self.fmax / self.fref)
        self.cent_bins = np.linspace(self.cent_min, self.cent_max, num_of_bins)
        print(f"Cent Resolution: {self.cent_bins[1] - self.cent_bins[0]:.2f} cents") 
        
        self.use_interpolation = use_interpolation
        self.segment_length = segment_length  # Should be 1 for frame-based
        self.preloaded_audio = []
        self.preloaded_f0_cents = []        
        self.meta = self.load_metadata()
        print("Pre-loading audio frames and F0 cents...")
        for audio_idx, audio_name in tqdm(enumerate(self.meta['audio_name']), desc="Pre-loading Data", total=len(self.meta['audio_name'])):
            audio_path = os.path.join(self.root, 'audio_stems', audio_name)
            f0_path = os.path.join(self.root, 'annotation_stems', self.meta['f0_name'][audio_idx])

            frames, frame_times = self.load_audio_frames(audio_path)
            f0_cents = self.load_f0_cents(f0_path, frame_times)

            self.preloaded_audio.append(frames)        # List of tensors
            self.preloaded_f0_cents.append(f0_cents)   # List of numpy arrays

        # Create frame_map using pre-loaded data
        self.frame_map = self.create_frame_map()



    def create_frame_map(self):
        frame_map = []
        print("Creating frame map by filtering out unvoiced frames...")
        for audio_idx, f0_cents in enumerate(self.preloaded_f0_cents):
            voiced_indices = np.where((~np.isnan(f0_cents)) & (f0_cents > 0))[0]
            for frame_idx in voiced_indices:
                frame_map.append((audio_idx, frame_idx))
        print(f"Total voiced frames: {len(frame_map)}")
        return frame_map

    def __getitem__(self, idx):
        audio_idx, frame_idx = self.frame_map[idx]

        # Retrieve pre-loaded frame and f0_cents
        frame = self.preloaded_audio[audio_idx][frame_idx]
        f0_cent = self.preloaded_f0_cents[audio_idx][frame_idx]

        # Convert to tensor if not already
        if not isinstance(frame, torch.Tensor):
            frame = torch.tensor(frame, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            frame = self.transform(frame)

        # Determine the bin index
        bin_idx = np.digitize(f0_cent, self.cent_bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(self.cent_bins) - 1)

        # Ground truth pitch in cents
        ground_truth_pitch = torch.tensor(f0_cent, dtype=torch.float32)

        return {'audio': frame, 'label': bin_idx, 'ground_truth_pitch': ground_truth_pitch}



    def __len__(self):
        return len(self.frame_map)


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
        audio, sr = librosa.load(path, sr=self.sr, mono=True, offset=start_time, duration=duration)

        # Pad audio to ensure centering of frames
        pad_width = self.frame_length // 2
        audio = np.pad(audio, pad_width, mode='constant')
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=self.hop_length).T
        frames = frames.copy()

        # Normalize frames
        frames -= np.mean(frames, axis=1, keepdims=True)
        frames /= np.clip(np.std(frames, axis=1, keepdims=True), 1e-8, None)

        frame_times = (np.arange(frames.shape[0]) * self.hop_length) / self.sr

        return torch.tensor(frames, dtype=torch.float32), frame_times
    
    def load_f0_cents(self, f0_path, frame_times):
        """
        Load F0 annotations and convert to cents aligned with frame times.
        """
        f0_data = pd.read_csv(f0_path, header=None)
        f0_times = f0_data.iloc[:, 0].values
        f0_values = f0_data.iloc[:, 1].values

        f0_interp = np.interp(frame_times, f0_times, f0_values, left=0.0, right=0.0)
        return hz_to_cents(f0_interp, self.fref)
    
    def create_one_hot_targets(self, f0_cents):
        """Create one-hot encoded targets."""
        bin_indices = np.digitize(f0_cents, self.cent_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.cent_bins) - 1)

        targets = torch.zeros((len(f0_cents), len(self.cent_bins)), dtype=torch.float32)
        for i, bin_idx in enumerate(bin_indices):
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
        
        return blurred_targets

    def create_sinc_interpolated_targets(self, f0_cents, width_factor=5):
        """Create sinc-interpolated targets."""
        f0_cents = torch.tensor(f0_cents, dtype=torch.float32)
        cent_bins = torch.tensor(self.cent_bins, dtype=torch.float32)
        bin_width = cent_bins[1] - cent_bins[0]
        
        distances = (f0_cents.unsqueeze(1) - cent_bins.unsqueeze(0)) / bin_width
        
        x = torch.pi * distances * width_factor
        sinc_values = torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)
        
        envelope = torch.exp(-0.5 * (distances / width_factor)**2)
        sinc_values *= envelope

        normalized_values = sinc_values / torch.sum(sinc_values, dim=1, keepdim=True)
        
        normalized_values = torch.where(normalized_values > 0.0, normalized_values, torch.zeros_like(normalized_values))
        
        return normalized_values

    def visualize_interpolations(self, idx):
        """
        Visualize target interpolations for a specific frame.
        Note: This method is less meaningful in frame-based approach but retained for completeness.
        """
        sample = self[idx]
        frame = sample['audio'].numpy()
        target = sample['f0_values'].numpy()
        ground_truth_pitch = sample['ground_truth_pitch'].item()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        ax1.bar(np.arange(len(target)), target, color='blue', label='Target (f0_values)')
        ax1.set_title(f'True One-Hot Encoded Pitch (Ground Truth: {ground_truth_pitch:.2f} cents)')
        ax1.set_xlabel('Pitch Bins')
        ax1.set_ylabel('Probability')
        ax1.legend()
        
        ax2.bar(np.arange(len(frame)), frame, color='red', label='Audio Frame')
        ax2.set_title('Audio Frame Amplitude')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    dataset = MDBStemSynth(
        root="/root/MDB-stem-synth", 
        split="train", 
        step_size=10, 
        segment_length=1,  # Frame-based approach
        num_of_bins=360,
        use_interpolation='gaussian' 
    )
    
    print(f"Number of voiced frames: {len(dataset)}")
    
    random_idx = random.randint(0, len(dataset) - 1)
    fig = dataset.visualize_interpolations(random_idx)
    
    plt.savefig('interpolation_comparison.png')
    plt.close(fig)
    
    print(f"Visualization saved as 'interpolation_comparison.png'")
