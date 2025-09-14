#!/usr/bin/env python3
"""
Prepare Turkish corpus data for diacritics restoration training
Combines multiple corpus files: vikipedi_corpus.txt and aysnrgenc_turkishdeasciifier_train.txt
"""

import os
import pickle
import random
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


def load_all_corpus_files() -> List[str]:
    """Load and combine all available corpus files"""
    corpus_files = [
        "data/vikipedi_corpus.txt",
        "data/aysnrgenc_turkishdeasciifier_train.txt"
    ]

    all_paragraphs = []

    for file_path in corpus_files:
        if Path(file_path).exists():
            paragraphs = load_corpus_file(file_path)
            all_paragraphs.extend(paragraphs)
        else:
            print(f"Warning: {file_path} not found, skipping...")

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


def prepare_wikipedia_dataset(train_ratio: float = 0.8):
    """Prepare combined corpus dataset for training"""
    print("Loading corpus files...")
    paragraphs = load_all_corpus_files()

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
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / "combined_cache.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train': train_samples,
            'validation': val_samples
        }, f)

    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Cache file: {cache_file}")

    # Show sample
    print("\nSample training text:")
    print(train_samples[0][:300] + "...")

    return str(cache_file)


if __name__ == "__main__":
    cache_file = prepare_wikipedia_dataset()
    print(f"\nDataset ready for training: {cache_file}")
    print("\nTo train the model, run:")
    print("python train_wiki_model.py")