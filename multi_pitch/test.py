import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import CREPEModel  # Your model class
from data_loader import MDBStemSynth  # Your dataset class
import mir_eval
import os

# Constants
fref = 10.0  # Reference frequency in Hz
fmin = 32.70  # Minimum frequency in Hz
fmax = 1975.5  # Maximum frequency in Hz
num_of_bins = 360
cent_min = 1200 * np.log2(fmin / fref)  # ~2040.84 cents
cent_max = 1200 * np.log2(fmax / fref)  # ~9150.30 cents
sample_rate = 16000
hop_length = 160

BATCH_SIZE = 500
EPOCHS = 32
LEARNING_RATE = 1e-4
CENTS_PER_BIN = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PITCH_BINS = np.linspace(cent_min, cent_max, num_of_bins)
pitch_bins = torch.tensor(PITCH_BINS, dtype=torch.float32, device=DEVICE)

MODEL_PATH = "/home/ParnianRazavipour/crepe/multi_pitch/(ktop=2)best_model2_epoch_5.pth"

# Dataset and DataLoader
root = '/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth'
dataset = MDBStemSynth(root=root, split="train")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Load Model
model = CREPEModel(capacity='full').to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"Model loaded from {MODEL_PATH}")
model.eval()

# Loss Criterion
criterion = nn.BCEWithLogitsLoss()

# Evaluation with mir_eval
def evaluate_with_mir_eval(ground_truth_pitches, predicted_pitches, time_base):
    """
    Evaluates multi-pitch metrics using mir_eval.multipitch.

    Args:
        ground_truth_pitches: List of lists containing ground truth pitches (in cents).
        predicted_pitches: List of lists containing predicted pitches (in cents).
        time_base: 1D array of timestamps (in seconds) corresponding to each frame.

    Returns:
        precision: Precision score.
        accuracy_chroma: Chroma accuracy score.
    """
    ground_truth_hz = [
        np.clip(10 * 2 ** (np.array(frame_cents) / 1200), 20, None) for frame_cents in ground_truth_pitches
    ]

    predicted_hz = [
        np.clip(10 * 2 ** (np.array(frame_cents) / 1200), 20, None) for frame_cents in predicted_pitches
    ]

    ref_time = np.array(time_base)
    est_time = np.array(time_base)

    scores = mir_eval.multipitch.metrics(ref_time, ground_truth_hz, est_time, predicted_hz)

    precision = scores[0]
    recall = scores[1]
    accuracy = scores[2]
    precision_chroma = scores[7]
    recall_chroma = scores[8]
    accuracy_chroma = scores[9]

    return precision, accuracy

# Evaluation Loop
def evaluate(model, dataloader, criterion):
    """Evaluates the model on the validation set."""
    running_loss = 0.0
    total_precision = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].float().to(DEVICE)
            labels = batch['label'].long().to(DEVICE)
            ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)

            logits = model(audio)

            # Calculate loss
            loss = criterion(logits, labels)

            # Decode predictions
            probs = torch.sigmoid(logits)
            predicted_classes = (probs > 0.5).float().cpu().numpy()

            predicted_pitches = [
                PITCH_BINS[np.where(pred == 1)[0]].tolist() for pred in predicted_classes
            ]

            ground_truth_pitches = [
                frame.cpu().numpy().tolist() for frame in ground_truth_pitch
            ]

            num_frames = len(ground_truth_pitch)
            time_frames = np.arange(num_frames) * hop_length / sample_rate

            # Evaluate using mir_eval
            precision, accuracy = evaluate_with_mir_eval(ground_truth_pitches, predicted_pitches, time_frames)

            running_loss += loss.item()
            total_precision += precision
            total_accuracy += accuracy

    epoch_loss = running_loss / len(dataloader)
    epoch_precision = total_precision / len(dataloader)
    epoch_accuracy = total_accuracy / len(dataloader)

    print(f"Validation Loss: {epoch_loss:.4f}, Precision: {epoch_precision:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Run Evaluation
evaluate(model, val_dataloader, criterion)
