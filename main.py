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

# Learning Rate Scheduler: Reduce LR by a factor of 10 after 50 epochs
scheduler_step = 50
scheduler_gamma = 0.1

def evaluate_with_mir_eval(ground_truth_cents, predicted_pitch, check = False):
    """Evaluates RPA and RCA using mir_eval and prints a transcription in cents."""
    ref_voicing = np.ones_like(ground_truth_cents)
    est_voicing = np.ones_like(predicted_pitch)

    epsilon = 1e-6  # Prevent log2(0)
    predicted_pitch = np.maximum(predicted_pitch, epsilon)

    # Convert the predicted pitch to cents
    est_cent = 1200 * np.log2(predicted_pitch / 10)
    if check:

        # Print a transcription (sample)
        print("\n--- Transcription in Cents ---")
        for gt_cent, pred_cent in zip(ground_truth_cents[:10], est_cent[:10]):  # Print the first 10 samples
            print(f"GT (Cents): {gt_cent:.2f}, Predicted (Cents): {pred_cent:.2f}")
        print("---------------------------------\n")

    # Raw Pitch Accuracy (RPA) and Raw Chroma Accuracy (RCA)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ground_truth_cents, est_voicing, est_cent, cent_tolerance=20)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ground_truth_cents, est_voicing, est_cent, cent_tolerance=20)
    
    return rpa, rca


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
    audio = batch['audio'].float().to(DEVICE)
    labels = batch['label'].long().to(DEVICE)  # Integer class labels
    ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)

    # Reshape inputs for the model
    batch_size, seq_length = audio.shape  # segment_length=1
    audio = audio.view(-1, seq_length)

    optimizer.zero_grad()

    # Forward pass
    outputs = model(audio)  # Outputs are raw logits

    # Compute loss
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Decode predictions
    _, predicted_classes = torch.max(outputs, 1)
    predicted_cents = PITCH_BINS[predicted_classes.cpu().numpy()]
    predicted_pitch = 10 * (2 ** (predicted_cents / 1200))

    # Ground truth pitch in cents
    ground_truth_cents = ground_truth_pitch.detach().cpu().numpy()


    if train_step.counter < 1:
        print("\n--- Sample Predictions ---")
        for gt, pred in zip(ground_truth_cents[:5], predicted_cents[:5]):
            print(f"GT: {gt:.2f} Hz, Predicted: {pred:.2f} Hz")
        print("--------------------------\n")
        train_step.counter += 1


    # Evaluate with mir_eval
    rpa, rca = evaluate_with_mir_eval(
        ground_truth_cents,
        predicted_pitch, False
    )

    return loss.item(), rpa, rca

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
    rpa_total = 0.0
    rca_total = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        loss, rpa, rca = train_step(model, batch, criterion, optimizer)
        running_loss += loss
        rpa_total += rpa
        rca_total += rca
        num_batches += 1

    epoch_loss = running_loss / num_batches
    epoch_rpa = rpa_total / num_batches
    epoch_rca = rca_total / num_batches
    return epoch_loss, epoch_rpa, epoch_rca


def weighted_average_pitch(logits, pitch_bins):
    """
    Computes the pitch in Hz using a weighted average of pitch bins.

    Args:
        logits: Model output logits (before sigmoid), shape (batch, bins).
        pitch_bins: Tensor of pitch bin frequencies in Hz, shape (bins,).

    Returns:
        predicted_pitch: Tensor of predicted pitch in Hz, shape (batch,).
    """
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Compute the weighted average in cents
    weighted_cents = (probabilities * pitch_bins).sum(dim=1) / probabilities.sum(dim=1).clamp(min=1e-6)

    # Convert cents to Hz
    predicted_pitch = 10 * (2 ** (weighted_cents / 1200))
    return predicted_pitch

def evaluate(model, dataloader, criterion):
    print("evaluate!")
    model.eval()
    running_loss = 0.0
    rpa_total = 0.0
    rca_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].float().to(DEVICE)
            labels = batch['label'].long().to(DEVICE)
            ground_truth_pitch = batch['ground_truth_pitch'].float().to(DEVICE)

            # Reshape inputs for the model
            batch_size, seq_length = audio.shape  # segment_length=1
            audio = audio.view(-1, seq_length)

            # Forward pass
            logits = model(audio)  # Raw logits

            # Compute loss
            loss = criterion(logits, labels)

            # Decode predictions
            _, predicted_classes = torch.max(logits, 1)
            predicted_cents = PITCH_BINS[predicted_classes.cpu().numpy()]
            predicted_pitch = 10 * (2 ** (predicted_cents / 1200))

            # Ground truth pitch in cents
            ground_truth_cents = ground_truth_pitch.cpu().numpy()

            # Evaluate with mir_eval
            rpa, rca = evaluate_with_mir_eval(
                ground_truth_cents,
                predicted_pitch, False
            )

            # Accumulate metrics
            running_loss += loss.item()
            rpa_total += rpa
            rca_total += rca

    # Calculate averages
    epoch_loss = running_loss / len(dataloader)
    epoch_rpa = rpa_total / len(dataloader)
    epoch_rca = rca_total / len(dataloader)
    return epoch_loss, epoch_rpa, epoch_rca



def main():
    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Load Dataset
    root = '/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth'
    dataset = MDBStemSynth(root=root, split="train")

    print("Total dataset size (voiced frames):", len(dataset))

    # Split dataset into train and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = CREPEModel(capacity='full').to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    best_val_rca = -float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"\nStarting Epoch {epoch}/{EPOCHS}...")

        train_loss, train_rpa, train_rca = train_epoch(model, train_dataloader, criterion, optimizer)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train RPA: {train_rpa:.4f}, Train RCA: {train_rca:.4f}')

        val_loss, val_rpa, val_rca = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch}, Val Loss: {val_loss:.4f}, Val RPA: {val_rpa:.4f}, Val RCA: {val_rca:.4f}')

        scheduler.step()

        # Save the best model
        if val_rca > best_val_rca:
            best_val_rca = val_rca
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
            print(f"Model saved at epoch {epoch} with validation RCA {val_rca:.4f}.")

    print("Training complete.")

if __name__ == '__main__':
    main()
