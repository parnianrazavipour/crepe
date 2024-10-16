# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import CREPEModel
from data_loader import MDBStemSynth
from torch.utils.data import DataLoader, SubsetRandomSampler
import random


# Hyperparameters
NUM_BATCHES_PER_EPOCH = 500
BATCH_SIZE = 1
NUM_SAMPLES_PER_EPOCH = NUM_BATCHES_PER_EPOCH * BATCH_SIZE  # 500 * 32 = 16000
EPOCHS = 3
NUM_VAL_SAMPLES = 4000
LEARNING_RATE = 0.0002
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PITCH_BINS = np.linspace(32.70, 1975.53, 360)  



def create_sampler(dataset, num_samples, seed=None):
    indices = list(range(len(dataset)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    else:
        random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    sampler = SubsetRandomSampler(sampled_indices)
    return sampler



def to_local_average_cents(salience, center=None):
    """Find the weighted average cents near the argmax bin."""
    cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    else:
        return np.array([to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])])

def calculate_pitch(prediction):
    """Converts the 360-bin prediction (salience matrix) to a pitch estimate using local weighted average in cents."""
    prediction_probs = prediction.detach().cpu().numpy() 
    predicted_cents = np.array([to_local_average_cents(pred) for pred in prediction_probs])  

    predicted_pitch = 10 * 2 ** (predicted_cents / 1200)
    return torch.tensor(predicted_pitch).to(prediction.device)  

def calculate_rpa(predicted_pitch, ground_truth_pitch, threshold_cents=50):
    """Calculates Raw Pitch Accuracy (RPA) using a 50-cent threshold."""
    pitch_diff = 1200 * torch.log2(predicted_pitch / ground_truth_pitch)
    within_threshold = torch.abs(pitch_diff) <= threshold_cents
    rpa = torch.mean(within_threshold.float()).item()
    return rpa

def train_step(model, batch, criterion, optimizer):
    """Performs a single training step."""
    audio = batch['audio'].float().to(DEVICE)  # Shape: (batch_size, 1024)
    f0_values = batch['f0_values'].float().to(DEVICE)  # Shape: (batch_size, 360)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(audio)

    # Loss calculation
    loss = criterion(outputs, f0_values)
    loss.backward()

    optimizer.step()

    torch.cuda.empty_cache()


    predicted_pitch = calculate_pitch(outputs)

    bin_indices = torch.argmax(f0_values, dim=1)
    ground_truth_pitch = torch.tensor(PITCH_BINS)[bin_indices].to(DEVICE)

    unvoiced_mask = (f0_values.sum(dim=1) == 0)
    ground_truth_pitch[unvoiced_mask] = 1e-6
    rpa = calculate_rpa(predicted_pitch, ground_truth_pitch)


    del audio, f0_values, outputs

    return loss.item(), rpa

def train_epoch(model, dataloader, criterion, optimizer):
    """Performs one epoch of training."""
    model.train()
    running_loss = 0.0
    rpa_total = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training Epoch"):
        if num_batches >= NUM_BATCHES_PER_EPOCH:
            break
        loss, rpa = train_step(model, batch, criterion, optimizer)
        running_loss += loss
        rpa_total += rpa
        num_batches += 1

    epoch_loss = running_loss / NUM_BATCHES_PER_EPOCH
    epoch_rpa = rpa_total / NUM_BATCHES_PER_EPOCH
    return epoch_loss, epoch_rpa


def evaluate(model, dataloader, criterion):
    """Evaluates the model."""
    model.eval()
    running_loss = 0.0
    rpa_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].float().to(DEVICE)
            f0_values = batch['f0_values'].float().to(DEVICE)

            outputs = model(audio)

            loss = criterion(outputs, f0_values)

            predicted_pitch = calculate_pitch(outputs)

            bin_indices = torch.argmax(f0_values, dim=1)
            ground_truth_pitch = torch.tensor(PITCH_BINS)[bin_indices].to(DEVICE)

            unvoiced_mask = (f0_values.sum(dim=1) == 0)
            ground_truth_pitch[unvoiced_mask] = 1e-6

            rpa = calculate_rpa(predicted_pitch, ground_truth_pitch)

            running_loss += loss.item()
            rpa_total += rpa

    epoch_loss = running_loss / len(dataloader)
    epoch_rpa = rpa_total / len(dataloader)
    return epoch_loss, epoch_rpa

def main():
    # Load Dataset
    train_dataset = MDBStemSynth(
        root='/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth',
        split="train",
        step_size=10,
        transform=None
    )

    val_dataset = MDBStemSynth(
        root='/home/ParnianRazavipour/mdb_stem_synth/MDB-stem-synth',
        split="val",
        step_size=10,
        transform=None
    )

    train_sampler = create_sampler(train_dataset, NUM_SAMPLES_PER_EPOCH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)

    VAL_SAMPLER_SEED = 42  
    val_sampler = create_sampler(val_dataset, NUM_VAL_SAMPLES, seed=VAL_SAMPLER_SEED)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)

    print("Datasets loaded and prepared.")
    torch.cuda.empty_cache()

    model = CREPEModel(capacity='full').to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        print(f"Starting Epoch {epoch}...\n")

        train_loss, train_rpa = train_epoch(model, train_dataloader, criterion, optimizer)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train RPA: {train_rpa:.4f}')
        val_loss, val_rpa = evaluate(model, val_dataloader, criterion)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train RPA: {train_rpa:.4f}, Val Loss: {val_loss:.4f}, Val RPA: {val_rpa:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Model saved at epoch {epoch} with validation loss {val_loss:.4f}.\n")

    print("Training complete.")

if __name__ == '__main__':
    main()

