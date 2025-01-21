import torch
import matplotlib.pyplot as plt
import numpy as np
from model import CREPEModel
from data_loader import MDBStemSynth
from torch.utils.data import DataLoader, random_split
import os
import librosa
import librosa.display
import soundfile as sf

SAVE_DIR = "output_signals"

fref = 10.0
fmin = 32.70
fmax = 1975.5
num_of_bins = 360

cent_min = 1200 * np.log2(fmin / fref)
cent_max = 1200 * np.log2(fmax / fref)

BATCH_SIZE = 500
EPOCHS = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CENTS_PER_BIN = 20
PITCH_BINS = np.linspace(cent_min, cent_max, num_of_bins)
pitch_bins = torch.tensor(PITCH_BINS, dtype=torch.float32, device=DEVICE)

scheduler_step = 50
scheduler_gamma = 0.1

root = '/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth'
dataset = MDBStemSynth(root=root, split="train")
_, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CREPEModel(capacity='full').to(DEVICE)
MODEL_PATH = "/home/ParnianRazavipour/crepe/multi_pitch/(**Threshold=0.5)best_model2_epoch_31.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

audio_indices_1 = dataset.names_idx["MusicDelta_InTheHalloftheMountainKing_STEM_03.RESYN.wav"]
audio_indices_2 = dataset.names_idx["MusicDelta_Beatles_STEM_07.RESYN.wav"]

def extract_frames(dataset, audio_idx):
    frames = dataset.preloaded_audio[audio_idx]
    ground_truth_pitches = dataset.preloaded_f0_cents[audio_idx]
    return frames.numpy(), ground_truth_pitches

audio_1, ground_truth_pitch_1 = extract_frames(dataset, audio_indices_1)
audio_2, ground_truth_pitch_2 = extract_frames(dataset, audio_indices_2)

min_frames = min(audio_1.shape[0], audio_2.shape[0])

audio_1 = audio_1[:min_frames]
audio_2 = audio_2[:min_frames]

ground_truth_pitch_1 = ground_truth_pitch_1[:min_frames]
ground_truth_pitch_2 = ground_truth_pitch_2[:min_frames]

audio_1_combined = audio_1.flatten()
audio_2_combined = audio_2.flatten()

combined_audio = (audio_1_combined + audio_2_combined)

sf.write(os.path.join(SAVE_DIR, "signal_1.wav"), audio_1_combined, samplerate=16000)
sf.write(os.path.join(SAVE_DIR, "signal_2.wav"), audio_2_combined, samplerate=16000)
sf.write(os.path.join(SAVE_DIR, "combined_signal.wav"), combined_audio, samplerate=16000)

duration_1 = len(audio_1_combined) / 16000
duration_2 = len(audio_2_combined) / 16000
print(f"signal_1.wav duration: {duration_1:.2f} seconds, frames: {len(audio_1_combined)}")
print(f"signal_2.wav duration: {duration_2:.2f} seconds, frames: {len(audio_2_combined)}")
print(f"Saved signal_1.wav, signal_2.wav, and combined_signal.wav to {SAVE_DIR}")

frames1 = audio_1
frames2 = audio_2

combined_frames = frames1 + frames2
combined_frames_tensor = torch.tensor(combined_frames, dtype=torch.float32).to(DEVICE)

combined_signal_pitch = (ground_truth_pitch_1 + ground_truth_pitch_2)

logits = model(combined_frames_tensor)

probs = torch.sigmoid(logits).cpu().detach().numpy()
predicted_classes = (probs > 0.5).astype(float)

predicted_pitches = [
    PITCH_BINS[np.where(pred == 1)[0]].tolist() for pred in predicted_classes
]

def visualize_pitch_predictions(
    audio, ground_truth_1, ground_truth_2, predicted_pitches, 
    combined_signal=None, sample_rate=16000, hop_length=160, save_path=None):
    audio_continuous = audio.flatten()
    time_audio = np.linspace(0, len(audio_continuous) / sample_rate, len(audio_continuous))
    num_frames = len(ground_truth_1)
    time_frames = np.arange(num_frames) * hop_length / sample_rate

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 1, 1)
    plt.plot(time_audio, audio_continuous, label="Mixed Waveform")
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_frames, ground_truth_1, label="Ground Truth 1 (Cents)", color='blue')
    plt.plot(time_frames, ground_truth_2, label="Ground Truth 2 (Cents)", color='green')
    if combined_signal is not None:
        plt.plot(time_frames, combined_signal, label="Combined Signal (Cents)", color='orange')

    for frame_idx, pitches in enumerate(predicted_pitches):
        for pitch in pitches:
            plt.scatter(time_frames[frame_idx], pitch, color='red', s=5)

    plt.title("Multi-Pitch Predictions with Threshold > 0.5")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Cents)")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()

visualize_pitch_predictions(
    combined_frames.flatten(),
    ground_truth_pitch_1.flatten(),
    ground_truth_pitch_2.flatten(),
    predicted_pitches,
    None,
    save_path="plot.png"
)
