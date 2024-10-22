import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from model import CREPEModel
from data_loader import MDBStemSynth
import mir_eval
import random

torch.autograd.set_detect_anomaly(True)

NUM_BATCHES_PER_EPOCH = 500
BATCH_SIZE = 32
NUM_SAMPLES_PER_EPOCH = NUM_BATCHES_PER_EPOCH * BATCH_SIZE
EPOCHS = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CENTS_PER_BIN = 20  # cents
PITCH_BINS = np.linspace(32.70, 1975.53, 360)
pitch_bins = torch.tensor(PITCH_BINS, dtype=torch.float32, device=DEVICE)

def evaluate_with_mir_eval(ground_truth_pitch, predicted_pitch):
    """Evaluates RPA and RCA using mir_eval, taking into account voiced/unvoiced frames."""
    # Prepare mir_eval inputs (remove NaNs and 0 values)
    valid = ground_truth_pitch > 0
    ref_voicing = valid.astype(float)  # 1 for voiced frames, 0 for unvoiced frames
    est_voicing = (predicted_pitch > 0).astype(float)
    
    epsilon = 1e-6  # Small value to prevent log2(0)
    predicted_pitch = np.maximum(predicted_pitch, epsilon)

    # Convert the predicted pitch to cents
    est_cent = 1200 * np.log2(predicted_pitch / 10)
    
    # Raw Pitch Accuracy (RPA) and Raw Chroma Accuracy (RCA)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_voicing, ground_truth_pitch, est_voicing, est_cent , cent_tolerance=20)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_voicing, ground_truth_pitch, est_voicing, est_cent , cent_tolerance=20 )
    
    return rpa, rca

def decode_weighted_average(logits, pitch_bins):
    """Implements weighted average decoding using sigmoid probabilities."""
    probs = logits
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
    f0_values = batch['f0_values'].float().to(DEVICE)

    batch_size, seq_length, frame_length = audio.shape
    audio = audio.view(-1, frame_length)
    f0_values = f0_values.view(-1, f0_values.shape[-1])

    optimizer.zero_grad()

    # Forward pass
    outputs = model(audio)

    # Use weighted average decoding
    predicted_pitch = decode_weighted_average(outputs, pitch_bins)

    bin_indices = torch.argmax(f0_values, dim=1)
    ground_truth_pitch = pitch_bins[bin_indices]

    # Handle unvoiced frames (silence)
    unvoiced_mask = (f0_values.sum(dim=1) == 0)
    ground_truth_pitch[unvoiced_mask] = torch.tensor(0.0, device=DEVICE)

    voiced_frames = ~unvoiced_mask
    if voiced_frames.sum() > 0:
        loss = criterion(outputs[voiced_frames], f0_values[voiced_frames])
        loss.backward()
        optimizer.step()

        rpa, rca = evaluate_with_mir_eval(
            ground_truth_pitch[voiced_frames].detach().cpu().numpy(), 
            predicted_pitch[voiced_frames].detach().cpu().numpy()
        )
    else:
        loss = torch.tensor(0.0, device=DEVICE)
        rpa, rca = 0.0, 0.0

    return loss.item(), rpa, rca

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    rpa_total = 0.0
    rca_total = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        if num_batches >= NUM_BATCHES_PER_EPOCH:
            break
        loss, rpa, rca = train_step(model, batch, criterion, optimizer)
        running_loss += loss
        rpa_total += rpa
        rca_total += rca
        num_batches += 1

    epoch_loss = running_loss / NUM_BATCHES_PER_EPOCH
    epoch_rpa = rpa_total / NUM_BATCHES_PER_EPOCH
    epoch_rca = rca_total / NUM_BATCHES_PER_EPOCH
    return epoch_loss, epoch_rpa, epoch_rca

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    rpa_total = 0.0
    rca_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].float().to(DEVICE)
            f0_values = batch['f0_values'].float().to(DEVICE)

            batch_size, seq_length, frame_length = audio.shape
            audio = audio.view(-1, frame_length)
            f0_values = f0_values.view(-1, f0_values.shape[-1])

            outputs = model(audio)

            predicted_pitch = decode_weighted_average(outputs, pitch_bins)

            bin_indices = torch.argmax(f0_values, dim=1)
            ground_truth_pitch = pitch_bins[bin_indices]

            unvoiced_mask = (f0_values.sum(dim=1) == 0)
            ground_truth_pitch[unvoiced_mask] = torch.tensor(0.0, device=DEVICE)

            voiced_frames = ~unvoiced_mask
            if voiced_frames.sum() > 0:
                loss = criterion(outputs[voiced_frames], f0_values[voiced_frames])
                rpa, rca = evaluate_with_mir_eval(ground_truth_pitch.cpu().numpy(), predicted_pitch.cpu().numpy())
            else:
                loss = torch.tensor(0.0, device=DEVICE)
                rpa, rca = 0.0, 0.0

            running_loss += loss.item()
            rpa_total += rpa
            rca_total += rca

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

    print("Total dataset size:", len(dataset))

    # Split dataset into train and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    model = CREPEModel(capacity='full').to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"Starting Epoch {epoch}...")

        train_loss, train_rpa, train_rca = train_epoch(model, train_dataloader, criterion, optimizer)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train RPA: {train_rpa:.4f}, Train RCA: {train_rca:.4f}')

        val_loss, val_rpa, val_rca = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch}, Val Loss: {val_loss:.4f}, Val RPA: {val_rpa:.4f}, Val RCA: {val_rca:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
            print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}.")

    print("Training complete.")

if __name__ == '__main__':
    main()
