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
    create_constrained_training_data,
    remove_diacritics_simple,
    detect_optimal_device
)
from collections import Counter
import random


def balance_training_samples(training_samples, target_samples_per_char=None):
    """Balance training samples to ensure equal representation of each character type"""

    if not training_samples:
        return training_samples

    print("Balancing training samples for equal character representation...")

    # Group samples by which character types they contain
    char_to_samples = {'c': [], 'g': [], 'i': [], 'o': [], 's': [], 'u': []}

    for sample in training_samples:
        labels = sample.get('labels', {})
        for char_type in labels.keys():
            base_char = char_type.lower()
            if base_char in char_to_samples:
                char_to_samples[base_char].append(sample)

    # Print current distribution
    print("Current sample distribution by character type:")
    for char, samples in char_to_samples.items():
        print(f"  {char}/{char.upper()}: {len(samples):,} samples")

    # Determine target number of samples per character
    if target_samples_per_char is None:
        # Use the minimum count to ensure we don't lose rare characters
        min_count = min(len(samples) for samples in char_to_samples.values() if len(samples) > 0)
        target_samples_per_char = min_count
        print(f"Auto-selected target: {target_samples_per_char:,} samples per character")
    else:
        print(f"Target: {target_samples_per_char:,} samples per character")

    # Balance by sampling from each character type
    balanced_samples = set()  # Use set to avoid duplicates

    for char, samples in char_to_samples.items():
        if len(samples) == 0:
            print(f"Warning: No samples found for character '{char}'")
            continue

        if len(samples) >= target_samples_per_char:
            # Randomly sample if we have enough
            selected = random.sample(samples, target_samples_per_char)
        else:
            # Use all samples if we don't have enough (with possible duplicates)
            selected = samples
            # Oversample to reach target
            while len(selected) < target_samples_per_char:
                selected.extend(random.choices(samples, k=min(len(samples), target_samples_per_char - len(selected))))

        # Add to balanced set (duplicates automatically removed)
        for sample in selected:
            balanced_samples.add(id(sample))  # Use id to track unique objects

    # Convert back to list of unique samples
    balanced_list = []
    sample_ids = set()

    for sample in training_samples:
        if id(sample) in balanced_samples and id(sample) not in sample_ids:
            balanced_list.append(sample)
            sample_ids.add(id(sample))

    print(f"Balanced dataset: {len(balanced_list):,} samples (was {len(training_samples):,})")
    return balanced_list


def calculate_character_frequencies(train_texts):
    """Calculate frequency weights for diacritic character types"""
    diacritic_chars = {'c', 'C', 'g', 'G', 'i', 'I', 'o', 'O', 's', 'S', 'u', 'U'}
    char_counts = Counter()

    print("Calculating character frequencies for loss weighting...")

    for text in train_texts:
        # Remove diacritics to get input text
        input_text = remove_diacritics_simple(text)
        target_text = text

        # Count characters that need diacritic restoration
        for input_char, target_char in zip(input_text, target_text):
            if input_char != target_char and input_char in diacritic_chars:
                # Map both cases to lowercase for consistency
                base_char = input_char.lower()
                char_counts[base_char] += 1

    # Calculate total diacritic characters
    total_diacritic_chars = sum(char_counts.values())

    if total_diacritic_chars == 0:
        print("Warning: No diacritic characters found in training data!")
        return {char: 1.0 for char in ['c', 'g', 'i', 'o', 's', 'u']}

    # Calculate frequencies (what portion each character represents)
    frequencies = {}
    for char in ['c', 'g', 'i', 'o', 's', 'u']:
        count = char_counts.get(char, 0)
        frequency = count / total_diacritic_chars
        frequencies[char] = frequency
        frequencies[char.upper()] = frequency  # Same weight for uppercase

    print("Character frequencies in training data:")
    for char in ['c', 'g', 'i', 'o', 's', 'u']:
        count = char_counts.get(char, 0)
        print(f"  {char}/{char.upper()}: {frequencies[char]:.3f} ({count:,} occurrences)")

    return frequencies


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

    # Calculate character frequencies for loss weighting
    selected_train_texts = train_texts[:max_train_texts]
    char_frequencies = calculate_character_frequencies(selected_train_texts)

    train_samples = create_constrained_training_data(selected_train_texts, context_size=context_size)
    val_samples = create_constrained_training_data(val_texts[:max_val_texts], context_size=context_size)

    print(f"Created {len(train_samples)} training samples, {len(val_samples)} validation samples")

    # Apply balanced sampling if requested
    if getattr(args, 'balanced_sampling', False):
        train_samples = balance_training_samples(train_samples,
                                               target_samples_per_char=getattr(args, 'samples_per_char', None))
        print("Applied balanced sampling to training data")

    if len(train_samples) == 0:
        print("No training samples created! Check your data.")
        return

    # Create datasets
    train_dataset = ConstrainedDiacriticsDataset(train_samples)
    val_dataset = ConstrainedDiacriticsDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model with enhanced device detection
    print("Detecting optimal device for training...")
    device = detect_optimal_device(verbose=True)

    # CUDA-specific optimizations
    if device.type == 'cuda':
        print(f"\nCUDA Optimization Tips:")
        if args.batch_size < 32:
            print(f"   - Consider increasing batch size (current: {args.batch_size}, try: 64-128)")
        if torch.cuda.get_device_properties(0).total_memory > 40e9:  # >40GB memory
            print(f"   - High-memory GPU detected - can handle large models")
            print(f"   - Try: --cuda-config for optimized settings")
        elif torch.cuda.get_device_properties(0).total_memory > 10e9:  # >10GB memory
            print(f"   - Mid-range GPU detected - good for medium models")
        print(f"   - GPU will be fully utilized during training")

    elif device.type == 'cpu':
        print(f"\nCPU Performance Warnings:")
        if args.batch_size > 8:
            print(f"   - Consider reducing batch size (current: {args.batch_size}, try: 4-8)")
        if args.context_size > 50:
            print(f"   - Consider reducing context size (current: {args.context_size}, try: 20-30)")
        if use_attention:
            print(f"   - Consider disabling attention (--no-attention) for faster training")

    # Get additional model parameters
    num_lstm_layers = getattr(args, 'num_lstm_layers', 2)
    use_attention = getattr(args, 'use_attention', True)

    print(f"Model configuration:")
    print(f"  Context size: {context_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  LSTM layers: {num_lstm_layers}")
    print(f"  Self-attention: {'Enabled' if use_attention else 'Disabled'}")

    model = ConstrainedDiacriticsModel(
        context_size=context_size,
        hidden_size=args.hidden_size,
        num_lstm_layers=num_lstm_layers,
        use_attention=use_attention
    ).to(device)

    # Expert recommends AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

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

            # Calculate frequency-weighted loss for each character type
            batch_loss = 0
            total_weight = 0

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

                    # Compute frequency-weighted loss if we have matching labels and predictions
                    if len(all_labels) > 0 and len(all_labels) <= all_logits.size(0):
                        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
                        char_loss = nn.CrossEntropyLoss()(all_logits[:len(all_labels)], labels_tensor)

                        # Apply frequency weight for this character type
                        char_weight = char_frequencies.get(char_type, 1.0)
                        weighted_loss = char_loss * char_weight

                        batch_loss += weighted_loss
                        total_weight += char_weight

            if total_weight > 0:
                # Normalize by total weight instead of count
                batch_loss = batch_loss / total_weight
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                total_samples += 1  # Count batches

            if batch_idx % 10 == 0:
                current_loss = batch_loss.item() if total_weight > 0 else 0
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {current_loss:.4f}")

        avg_loss = total_loss / max(total_samples, 1)  # Now dividing by number of batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save work-in-progress checkpoint after each epoch
        temp_restorer = ConstrainedDiacriticsRestorer(
            context_size=context_size,
            hidden_size=args.hidden_size,
            num_lstm_layers=num_lstm_layers,
            use_attention=use_attention
        )
        temp_restorer.model = model
        temp_restorer.save_model(wip_path)
        print(f"Checkpoint saved to {wip_path}")

    # Save final model
    restorer = ConstrainedDiacriticsRestorer(
        context_size=context_size,
        hidden_size=args.hidden_size,
        num_lstm_layers=num_lstm_layers,
        use_attention=use_attention
    )
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
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                       help='Learning rate (default: 0.0003, expert recommended 3e-4)')
    parser.add_argument('--context-size', type=int, default=96,
                       help='Context window size (default: 96, expert recommended)')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size (default: 256, expert recommended)')
    parser.add_argument('--num-lstm-layers', type=int, default=2,
                       help='Number of LSTM layers (default: 2, expert recommends 2-4)')
    parser.add_argument('--use-attention', action='store_true', default=True,
                       help='Use self-attention layer (default: True, expert recommended)')
    parser.add_argument('--no-attention', dest='use_attention', action='store_false',
                       help='Disable self-attention layer')
    parser.add_argument('--max-train-texts', type=int, default=15000,
                       help='Maximum number of training texts to use (default: 15000)')
    parser.add_argument('--max-val-texts', type=int, default=1000,
                       help='Maximum number of validation texts to use (default: 1000)')
    parser.add_argument('--balanced-sampling', action='store_true',
                       help='Apply balanced sampling to ensure equal character representation')
    parser.add_argument('--samples-per-char', type=int,
                       help='Target number of samples per character type (auto if not specified)')

    # Platform-specific configurations
    parser.add_argument('--cuda-config', action='store_true',
                       help='Use CUDA-optimized configuration (large batch, context, hidden size)')
    parser.add_argument('--mps-config', action='store_true',
                       help='Use MPS-optimized configuration (medium batch, context, hidden size)')
    parser.add_argument('--cpu-config', action='store_true',
                       help='Use CPU-optimized configuration (small batch, context, no attention)')

    args = parser.parse_args()

    # Apply pre-configured settings
    if args.cuda_config:
        print("🚀 Applying CUDA-optimized configuration...")
        args.batch_size = 128
        args.context_size = 128
        args.hidden_size = 512
        args.use_attention = True
        args.learning_rate = 0.001  # Slightly higher for larger batches
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - Context size: {args.context_size}")
        print(f"   - Hidden size: {args.hidden_size}")
        print(f"   - Learning rate: {args.learning_rate}")

    elif args.mps_config:
        print("🍎 Applying MPS-optimized configuration...")
        args.batch_size = 64
        args.context_size = 96
        args.hidden_size = 256
        args.use_attention = True
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - Context size: {args.context_size}")
        print(f"   - Hidden size: {args.hidden_size}")
        print(f"   - Attention: {args.use_attention}")

    elif args.cpu_config:
        print("💻 Applying CPU-optimized configuration...")
        args.batch_size = 4
        args.context_size = 20
        args.hidden_size = 128
        args.use_attention = False
        print(f"   - Batch size: {args.batch_size}")
        print(f"   - Context size: {args.context_size}")
        print(f"   - Hidden size: {args.hidden_size}")
        print(f"   - Attention: {args.use_attention}")

    # Create output directory
    Path(args.output).parent.mkdir(exist_ok=True)

    train_constrained_model(args)


if __name__ == "__main__":
    main()