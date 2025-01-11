import torch
from model import CREPEModel  # Your model class
from data_loader import MDBStemSynth  # Your dataset class
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
from model import CREPEModel
from data_loader import MDBStemSynth
import mir_eval
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from model import CREPEModel
from data_loader import MDBStemSynth
import mir_eval
import os
import random
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

import numpy as np

fref = 10.0  # Reference frequency in Hz
fmin = 32.70  # Minimum frequency in Hz
fmax = 1975.5  # Maximum frequency in Hz
num_of_bins = 360

cent_min = 1200 * np.log2(fmin / fref)  # ~2040.84 cents
cent_max = 1200 * np.log2(fmax / fref)  # ~9150.30 cents


BATCH_SIZE = 500          
EPOCHS = 32              # Changed from 32 to 100 as per CREPE
LEARNING_RATE = 1e-4      # Changed from 0.0002 to 1e-4 as per CREPE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CENTS_PER_BIN = 20        # cents
PITCH_BINS = np.linspace(cent_min, cent_max, num_of_bins)
pitch_bins = torch.tensor(PITCH_BINS, dtype=torch.float32, device=DEVICE)

scheduler_step = 50
scheduler_gamma = 0.1



MODEL_PATH = "best_model_epoch_9.pth"  

# Load the dataset
root = '/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth'
dataset = MDBStemSynth(root=root, split="train")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = random_split(dataset, [train_size, val_size])

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

model = CREPEModel(capacity='full').to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print(f"Model loaded from {MODEL_PATH}")
model.eval()

criterion = torch.nn.CrossEntropyLoss()


def evaluate_with_mir_eval(ground_truth_cents, predicted_pitch, check = False):
    """Evaluates RPA and RCA using mir_eval and prints a transcription in cents."""
    ref_voicing = np.ones_like(ground_truth_cents)
    est_voicing = np.ones_like(predicted_pitch)

    epsilon = 1e-6  # Prevent log2(0)
    predicted_pitch = np.maximum(predicted_pitch, epsilon)

    est_cent = 1200 * np.log2(predicted_pitch / 10)
    if check:

        print("\n--- Transcription in Cents ---")
        for gt_cent, pred_cent in zip(ground_truth_cents[:10], est_cent[:10]):  # Print the first 10 samples
            print(f"GT (Cents): {gt_cent:.2f}, Predicted (Cents): {pred_cent:.2f}")
        print("---------------------------------\n")

    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ground_truth_cents, est_voicing, est_cent, cent_tolerance=20)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ground_truth_cents, est_voicing, est_cent, cent_tolerance=20)
    
    return rpa, rca


# Evaluation loop
def evaluate(model, dataloader, criterion):
    running_loss = 0.0
    rpa_total = 0.0
    rca_total = 0.0
    with torch.no_grad():  # Disable gradient computation
        for batch in dataloader:
            audio = batch['audio'].float().to(DEVICE)
            labels = batch['label'].long().to(DEVICE)
            ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)

            batch_size, seq_length = audio.shape
            audio = audio.view(-1, seq_length)

            logits = model(audio)

            loss = criterion(logits, labels)

            _, predicted_classes = torch.max(logits, 1)
            predicted_cents = PITCH_BINS[predicted_classes.cpu().numpy()]
            predicted_pitch = 10 * (2 ** (predicted_cents / 1200))

            ground_truth_cents = ground_truth_pitch.cpu().numpy()

            rpa, rca = evaluate_with_mir_eval(
                ground_truth_cents,
                predicted_pitch,
                False
            )

            running_loss += loss.item()
            rpa_total += rpa
            rca_total += rca

    epoch_loss = running_loss / len(dataloader)
    epoch_rpa = rpa_total / len(dataloader)
    epoch_rca = rca_total / len(dataloader)

    print(f"Validation Loss: {epoch_loss:.4f}, Validation RPA: {epoch_rpa:.4f}, Validation RCA: {epoch_rca:.4f}")

evaluate(model, val_dataloader, criterion)
