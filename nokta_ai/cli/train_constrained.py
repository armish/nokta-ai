#!/usr/bin/env python3
"""
Training script for constrained Turkish diacritics restoration model
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

from ..models.constrained import (
    ConstrainedDiacriticsModel,
    ConstrainedDiacriticsRestorer,
    create_constrained_training_data
)


class ConstrainedDiacriticsDataset(Dataset):
    """Dataset for constrained diacritics training"""

    def __init__(self, training_samples):
        self.samples = training_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        contexts = torch.tensor(sample['contexts'], dtype=torch.long)
        targets = torch.tensor(sample['targets'], dtype=torch.long)

        return contexts, targets, sample['labels']


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    contexts_batch = []
    targets_batch = []
    labels_batch = []

    for contexts, targets, labels in batch:
        contexts_batch.append(contexts)
        targets_batch.append(targets)
        labels_batch.append(labels)

    # Pad sequences to same length
    max_len = max(len(seq) for seq in contexts_batch)

    padded_contexts = []
    padded_targets = []

    for contexts, targets in zip(contexts_batch, targets_batch):
        seq_len = len(contexts)
        if seq_len < max_len:
            # Pad with spaces (ord(' ') = 32)
            pad_context = torch.full((max_len - seq_len, contexts.size(1)), 32, dtype=torch.long)
            pad_target = torch.full((max_len - seq_len,), 32, dtype=torch.long)

            contexts = torch.cat([contexts, pad_context], dim=0)
            targets = torch.cat([targets, pad_target], dim=0)

        padded_contexts.append(contexts)
        padded_targets.append(targets)

    return (torch.stack(padded_contexts),
            torch.stack(padded_targets),
            labels_batch)


def train_constrained_model(args):
    """Train the constrained diacritics model"""

    # Load training data
    print("Loading dataset...")
    with open(args.data_cache, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        train_texts = data.get('train', [])
        val_texts = data.get('validation', [])
    else:
        split_idx = int(len(data) * 0.8)
        train_texts = data[:split_idx]
        val_texts = data[split_idx:]

    print(f"Creating constrained training data from {len(train_texts)} texts...")

    # Create constrained training data
    context_size = args.context_size
    max_train_texts = getattr(args, 'max_train_texts', 10000)
    max_val_texts = getattr(args, 'max_val_texts', 1000)

    train_samples = create_constrained_training_data(train_texts[:max_train_texts], context_size=context_size)
    val_samples = create_constrained_training_data(val_texts[:max_val_texts], context_size=context_size)

    print(f"Created {len(train_samples)} training samples, {len(val_samples)} validation samples")

    if len(train_samples) == 0:
        print("No training samples created! Check your data.")
        return

    # Create datasets
    train_dataset = ConstrainedDiacriticsDataset(train_samples)
    val_dataset = ConstrainedDiacriticsDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    device = torch.device('mps' if torch.backends.mps.is_available()
                         else 'cuda' if torch.cuda.is_available()
                         else 'cpu')

    print(f"Using device: {device}")

    model = ConstrainedDiacriticsModel(context_size=context_size, hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    start_time = datetime.now()

    # Create work-in-progress checkpoint path
    wip_path = args.output + ".wip"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_samples = 0

        for batch_idx, (contexts, targets, labels_batch) in enumerate(train_loader):
            contexts = contexts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(contexts, targets)

            # Calculate loss for each character type
            batch_loss = 0
            batch_count = 0

            # Collect all labels for each character type across the batch
            for char_type, pred_data in predictions.items():
                if 'logits' in pred_data and 'mask' in pred_data:
                    all_logits = pred_data['logits']  # (num_total_matches, num_variants)
                    all_masks = pred_data['mask']     # (batch_size, seq_len)

                    # Collect labels for all matches of this character type
                    all_labels = []
                    logit_idx = 0

                    for sample_idx, labels in enumerate(labels_batch):
                        if char_type in labels:
                            sample_mask = all_masks[sample_idx]
                            num_matches_in_sample = sample_mask.sum().item()

                            # Get labels for this sample
                            sample_labels = [label_info['label'] for label_info in labels[char_type]]

                            # Verify we have the right number of labels
                            if len(sample_labels) == num_matches_in_sample:
                                all_labels.extend(sample_labels)
                                logit_idx += num_matches_in_sample
                            else:
                                # Skip this sample if label count doesn't match
                                logit_idx += num_matches_in_sample

                    # Compute loss if we have matching labels and predictions
                    if len(all_labels) > 0 and len(all_labels) <= all_logits.size(0):
                        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
                        loss = nn.CrossEntropyLoss()(all_logits[:len(all_labels)], labels_tensor)
                        batch_loss += loss
                        batch_count += 1

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                total_samples += 1  # Count batches, not individual predictions

            if batch_idx % 10 == 0:
                current_loss = batch_loss.item() if batch_count > 0 else 0
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {current_loss:.4f}")

        avg_loss = total_loss / max(total_samples, 1)  # Now dividing by number of batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save work-in-progress checkpoint after each epoch
        temp_restorer = ConstrainedDiacriticsRestorer(context_size=context_size)
        temp_restorer.model = model
        temp_restorer.save_model(wip_path)
        print(f"Checkpoint saved to {wip_path}")

    # Save final model
    restorer = ConstrainedDiacriticsRestorer(context_size=context_size)
    restorer.model = model
    restorer.save_model(args.output)

    # Clean up work-in-progress checkpoint
    try:
        Path(wip_path).unlink()
        print(f"Removed checkpoint file {wip_path}")
    except FileNotFoundError:
        pass

    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")
    print(f"Final model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Train constrained Turkish diacritics model')
    parser.add_argument('--data-cache', type=str, required=True,
                       help='Path to dataset cache file')
    parser.add_argument('--output', type=str, default='models/constrained_model.pth',
                       help='Output model path (default: models/constrained_model.pth)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--context-size', type=int, default=100,
                       help='Context window size (default: 100)')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden layer size (default: 128)')
    parser.add_argument('--max-train-texts', type=int, default=10000,
                       help='Maximum number of training texts to use (default: 10000)')
    parser.add_argument('--max-val-texts', type=int, default=1000,
                       help='Maximum number of validation texts to use (default: 1000)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    train_constrained_model(args)


if __name__ == "__main__":
    main()