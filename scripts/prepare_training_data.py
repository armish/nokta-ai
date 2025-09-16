#!/usr/bin/env python3
"""
Prepare Turkish corpus data for diacritics restoration training
Accepts one or more text files to combine into training data
"""

import os
import sys
import pickle
import random
import argparse
from pathlib import Path
from typing import List, Tuple


def load_corpus_file(file_path: str) -> List[str]:
    """Load and process a single corpus file"""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into paragraphs
    paragraphs = []
    current_paragraph = []

    for line in text.split('\n'):
        line = line.strip()
        if line:
            current_paragraph.append(line)
        elif current_paragraph:
            # Join lines into a paragraph
            paragraph = ' '.join(current_paragraph)
            if len(paragraph) > 100:  # Filter short paragraphs
                paragraphs.append(paragraph)
            current_paragraph = []

    # Add last paragraph if exists
    if current_paragraph:
        paragraph = ' '.join(current_paragraph)
        if len(paragraph) > 100:
            paragraphs.append(paragraph)

    print(f"  Loaded {len(paragraphs)} paragraphs from {file_path}")
    return paragraphs


def load_all_corpus_files(file_paths: List[str]) -> List[str]:
    """Load and combine all specified corpus files"""
    all_paragraphs = []

    for file_path in file_paths:
        if Path(file_path).exists():
            paragraphs = load_corpus_file(file_path)
            all_paragraphs.extend(paragraphs)
        else:
            print(f"Error: {file_path} not found!")
            sys.exit(1)

    print(f"Total paragraphs loaded: {len(all_paragraphs)}")
    return all_paragraphs


def create_training_samples(paragraphs: List[str],
                           window_size: int = 200,
                           stride: int = 100) -> List[str]:
    """Create overlapping training samples from paragraphs"""
    samples = []

    for paragraph in paragraphs:
        # If paragraph is short enough, use it as is
        if len(paragraph) <= window_size:
            samples.append(paragraph)
        else:
            # Create sliding windows
            for i in range(0, len(paragraph) - window_size + 1, stride):
                window = paragraph[i:i + window_size]
                samples.append(window)

    return samples


def prepare_dataset(file_paths: List[str],
                   output_file: str = "data/combined_cache.pkl",
                   train_ratio: float = 0.8):
    """Prepare combined corpus dataset for training"""
    print("Loading corpus files...")
    paragraphs = load_all_corpus_files(file_paths)

    if not paragraphs:
        print("Error: No paragraphs loaded from input files!")
        sys.exit(1)

    # Create training samples
    print("Creating training samples...")
    samples = create_training_samples(paragraphs)
    print(f"Created {len(samples)} training samples")

    # Shuffle samples
    random.shuffle(samples)

    # Split into train and validation
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save to cache
    cache_path = Path(output_file)
    cache_path.parent.mkdir(exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump({
            'train': train_samples,
            'validation': val_samples
        }, f)

    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Cache file: {cache_path}")

    # Show sample
    if train_samples:
        print("\nSample training text:")
        print(train_samples[0][:300] + "...")

    return str(cache_path)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Turkish corpus data for diacritics restoration training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default Turkish deasciifier training data
  python scripts/prepare_training_data.py data/aysnrgenc_turkishdeasciifier_train.txt

  # Combine multiple corpus files
  python scripts/prepare_training_data.py file1.txt file2.txt file3.txt

  # Specify custom output location
  python scripts/prepare_training_data.py data/*.txt --output data/my_cache.pkl

  # Adjust train/validation split ratio
  python scripts/prepare_training_data.py corpus.txt --train-ratio 0.9
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='Input text files to process (one or more)'
    )

    parser.add_argument(
        '--output',
        '-o',
        default='data/combined_cache.pkl',
        help='Output pickle file path (default: data/combined_cache.pkl)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training data ratio (default: 0.8, meaning 80%% train, 20%% validation)'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=200,
        help='Size of text windows for training samples (default: 200)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=100,
        help='Stride for sliding window (default: 100)'
    )

    args = parser.parse_args()

    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        print(f"Error: train-ratio must be between 0 and 1, got {args.train_ratio}")
        sys.exit(1)

    print(f"Processing {len(args.files)} file(s)...")
    for file_path in args.files:
        print(f"  - {file_path}")

    cache_file = prepare_dataset(
        file_paths=args.files,
        output_file=args.output,
        train_ratio=args.train_ratio
    )

    print(f"\nDataset ready for training: {cache_file}")
    print("\nTo train the model, run:")
    print(f"nokta-train --data-cache {cache_file} --output models/my_model.pth")


if __name__ == "__main__":
    main()