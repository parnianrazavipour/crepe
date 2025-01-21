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
sample_rate=16000
hop_length=160

BATCH_SIZE = 500          
EPOCHS = 32              
LEARNING_RATE = 1e-4     
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CENTS_PER_BIN = 20        # cents
PITCH_BINS = np.linspace(cent_min, cent_max, num_of_bins)
pitch_bins = torch.tensor(PITCH_BINS, dtype=torch.float32, device=DEVICE)

scheduler_step = 50
scheduler_gamma = 0.1


def decode_weighted_average(logits, pitch_bins):
    """
    Implements weighted average decoding using already sigmoid-activated probabilities.
    """
    probs = logits  # Model outputs are already sigmoid-activated
    pitch_estimates = torch.sum(probs * pitch_bins, dim=1) / torch.clamp(torch.sum(probs, dim=1), min=1e-6)
    return pitch_estimates

def bins_to_frequency(bins):
    return cents_to_frequency(bins_to_cents(bins))

def bins_to_cents(bins):
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    return cents

def cents_to_frequency(cents):
    return 10 * 2 ** (cents / 1200)

def train_step(model, batch, criterion, optimizer):
    """
    Train the model on a single batch for multi-pitch estimation.

    Args:
        model (nn.Module): The pitch estimation model.
        batch (dict): A batch of data containing 'audio', 'label', and 'ground_truth_pitch'.
        criterion (nn.Module): The loss function (e.g., BCEWithLogitsLoss).
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.

    Returns:
        tuple: (loss_value, PRECISION, ACC)
    """
    # Extract data
    audio = batch['audio'].float().to(DEVICE)              # Shape: [B, seq_length]
    labels = batch['label'].long().to(DEVICE)             # Shape: [B]
    ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)  # Shape: [B]

    batch_size, seq_length = audio.shape

    # Mix random samples within the batch
    shuffled_indices = torch.randperm(batch_size)
    audio_shuffled = audio[shuffled_indices]
    labels_shuffled = labels[shuffled_indices]
    ground_truth_pitch_shuffled = ground_truth_pitch[shuffled_indices]

    mixed_audio = (audio + audio_shuffled)/2.0

    # Multi-label targets
    multi_labels = torch.zeros(batch_size, len(PITCH_BINS)).to(DEVICE)
    multi_labels.scatter_(1, labels.unsqueeze(1), 1)
    multi_labels.scatter_(1, labels_shuffled.unsqueeze(1), 1)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(mixed_audio)  # Shape: [B, 360]
    loss = criterion(outputs, multi_labels)
    loss.backward()
    optimizer.step()

    # Decode predictions
    probs = torch.sigmoid(outputs)
    predicted_classes = (probs > 0.5).float().cpu().numpy()

    # Map predictions to pitches
    predicted_pitches = [
        PITCH_BINS[np.where(pred == 1)[0]].tolist() for pred in predicted_classes
    ]

    # Combine ground truth pitches
    ground_truth_pitches = [
        [gt1.item() for gt1 in [gt, gt_shuffled] if gt1 > 0]
        for gt, gt_shuffled in zip(ground_truth_pitch.cpu(), ground_truth_pitch_shuffled.cpu())
    ]

    num_frames = len(ground_truth_pitch)
    time_frames = np.arange(num_frames) * hop_length / sample_rate


    # Evaluate performance
    PRECISION, ACC = evaluate_with_mir_eval(ground_truth_pitches, predicted_pitches,time_frames)

    return loss.item(), PRECISION, ACC


import mir_eval
import numpy as np

def evaluate_with_mir_eval(ground_truth_pitches, predicted_pitches, time_base):
    """
    Evaluates multi-pitch metrics using mir_eval.multipitch.

    Args:
        ground_truth_pitches: List of lists, each containing ground truth pitches (in cents).
        predicted_pitches: List of lists, each containing predicted pitches (in cents).
        time_base: 1D array of timestamps (in seconds) corresponding to each frame.

    Returns:
        dict: Dictionary of scores including precision, recall, accuracy, and chroma-based metrics.
    """
    ground_truth_hz = [
        np.clip(10 * 2 ** (np.array(frame_cents) / 1200), 20, None) for frame_cents in ground_truth_pitches
    ]

    # Convert predicted pitches (cents) to Hz with a minimum threshold of 20 Hz
    predicted_hz = [
        np.clip(10 * 2 ** (np.array(frame_cents) / 1200), 20, None) for frame_cents in predicted_pitches
    ]

    est_time = np.array(time_base)
    est_freqs = predicted_hz

    ref_time = np.array(time_base)
    ref_freqs = ground_truth_hz


    scores = mir_eval.multipitch.metrics(ref_time, ref_freqs, est_time, est_freqs)


    precision = scores[0]
    recall = scores[1]
    accuracy = scores[2]
    precision_chroma = scores[7]
    recall_chroma = scores[8]
    accuracy_chroma = scores[9]

    return precision,accuracy





train_step.counter = 0

def visualize_predictions(f0_values, outputs, epoch, batch_idx, sample_idx=0):
    """Visualize and save a single example of predicted vs. true values for pitch bins."""
    if f0_values.size(0) == 0 or outputs.size(0) == 0:
        print(f"Skipping visualization for epoch {epoch}, batch {batch_idx}. No valid data.")
        return

    f0_values_np = f0_values[sample_idx].cpu().detach().numpy()
    outputs_np = outputs[sample_idx].cpu().detach().numpy()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    ax[0].bar(np.arange(len(f0_values_np)), f0_values_np, color='blue', label='Target (f0_values)')
    ax[0].set_title('True One-Hot Encoded Pitch (f0_values)')
    ax[0].set_xlabel('Pitch Bins')
    ax[0].set_ylabel('Probability')
    ax[0].legend()

    ax[1].bar(np.arange(len(outputs_np)), outputs_np, color='red', label='Predicted (outputs)')
    ax[1].set_title('Predicted Pitch Distribution')
    ax[1].set_xlabel('Pitch Bins')
    ax[1].set_ylabel('Probability')
    ax[1].legend()

    plt.tight_layout()

    output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{sample_idx}.png')
    plt.savefig(fig_path)
    plt.close(fig)


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    PRECISION_total = 0.0
    ACC_total = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        loss, PRECISION, ACC = train_step(model, batch, criterion, optimizer)
        running_loss += loss
        PRECISION_total += PRECISION
        ACC_total += ACC
        num_batches += 1

    epoch_loss = running_loss / num_batches
    epoch_PRECISION = PRECISION_total / num_batches
    epoch_ACC = ACC_total / num_batches
    return epoch_loss, epoch_PRECISION, epoch_ACC



    
def evaluate(model, dataloader, criterion):
    print("Evaluating...")
    model.eval()
    running_loss = 0.0
    PRECISION_total = 0.0
    ACC_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].float().to(DEVICE)  # Shape: [B, seq_length]
            labels = batch['label'].long().to(DEVICE)  # Shape: [B]
            ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)  # Shape: [B]

            batch_size, seq_length = audio.shape

            shuffled_indices = torch.randperm(batch_size)
            audio_shuffled = audio[shuffled_indices]
            labels_shuffled = labels[shuffled_indices]
            ground_truth_pitch_shuffled = ground_truth_pitch[shuffled_indices]

            #TODO: save the audio (for reference)
            mixed_audio = (audio + audio_shuffled) /2.0

            multi_labels = torch.zeros(batch_size, len(PITCH_BINS)).to(DEVICE)
            multi_labels.scatter_(1, labels.unsqueeze(1), 1)
            multi_labels.scatter_(1, labels_shuffled.unsqueeze(1), 1)

            #TODO: plot the sigmoid
            logits = model(mixed_audio)  # Shape: [B, 360]

            loss = criterion(logits, multi_labels)

            probs = torch.sigmoid(logits)
            predicted_classes = (probs > 0.5).float().cpu().numpy()

            predicted_pitches = [
                PITCH_BINS[np.where(pred == 1)[0]].tolist() for pred in predicted_classes
            ]

            # Combine ground truth pitches
            ground_truth_pitches = [
                [gt1.item() for gt1 in [gt, gt_shuffled] if gt1 > 0]
                for gt, gt_shuffled in zip(ground_truth_pitch.cpu(), ground_truth_pitch_shuffled.cpu())
            ]

            num_frames = len(ground_truth_pitch)
            time_frames = np.arange(num_frames) * hop_length / sample_rate


            PRECISION, ACC = evaluate_with_mir_eval(ground_truth_pitches, predicted_pitches,time_frames)

            # Accumulate metrics
            running_loss += loss.item()
            PRECISION_total += PRECISION
            ACC_total += ACC

    # Calculate averages
    epoch_loss = running_loss / len(dataloader)
    epoch_PRECISION = PRECISION_total / len(dataloader)
    epoch_ACC = ACC_total / len(dataloader)

    return epoch_loss, epoch_PRECISION, epoch_ACC



def main():
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Load Dataset
    root = '/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth'
    dataset = MDBStemSynth(root=root, split="train")

    print("Total dataset size (voiced frames):", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = CREPEModel(capacity='full').to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    best_val_ACC = -float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"\nStarting Epoch {epoch}/{EPOCHS}...")

        train_loss, train_PRECISION, train_ACC = train_epoch(model, train_dataloader, criterion, optimizer)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train PRECISION: {train_PRECISION:.4f}, Train ACC: {train_ACC:.4f}')

        val_loss, val_PRECISION, val_ACC = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch}, Val Loss: {val_loss:.4f}, Val PRECISION: {val_PRECISION:.4f}, Val ACC: {val_ACC:.4f}')

        scheduler.step()

        # Save the best model
        if val_ACC > best_val_ACC:
            best_val_ACC = val_ACC
            torch.save(model.state_dict(), f'(**Threshold=0.5)best_model2_epoch_{epoch}.pth')
            print(f"Model saved at epoch {epoch} with validation ACC {val_ACC:.4f}.")

    print("Training complete.")

if __name__ == '__main__':
    main()
