#!/usr/bin/env python3
"""
Training script for Turkish diacritics restoration model
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime
import torch
from diacritics_restoration import DiacriticsRestorer, TurkishDiacriticsMapper


def load_training_data(cache_file: str = "data/dataset_cache.pkl"):
    """Load training data from cache"""
    if not Path(cache_file).exists():
        print(f"Error: Cache file {cache_file} not found.")
        print("Please run prepare_training_data.py first to create the dataset.")
        sys.exit(1)

    with open(cache_file, 'rb') as f:
        texts = pickle.load(f)

    # Split into train and validation
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    return train_texts, val_texts


def evaluate_model(restorer, test_texts, num_samples=5):
    """Evaluate model on test samples"""
    mapper = TurkishDiacriticsMapper()

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    for i, text in enumerate(test_texts[:num_samples]):
        # Take a snippet
        snippet = text[:200] if len(text) > 200 else text

        # Remove diacritics
        stripped = mapper.remove_diacritics(snippet)

        # Restore
        restored = restorer.restore_diacritics(stripped)

        print(f"\nSample {i+1}:")
        print(f"Original:  {snippet}")
        print(f"Stripped:  {stripped}")
        print(f"Restored:  {restored}")

        # Calculate accuracy
        correct = sum(1 for c1, c2 in zip(snippet, restored) if c1 == c2)
        accuracy = correct / len(snippet) * 100
        print(f"Character accuracy: {accuracy:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Turkish diacritics restoration model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden size of the model (default: 256)')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of LSTM layers (default: 3)')
    parser.add_argument('--model-name', type=str, default='diacritics_model',
                       help='Name for saved model (default: diacritics_model)')
    parser.add_argument('--data-cache', type=str, default='data/dataset_cache.pkl',
                       help='Path to data cache file')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')

    args = parser.parse_args()

    # Setup paths
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{args.model_name}.pth'

    # Check device availability (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.evaluate_only:
        # Load existing model and evaluate
        if not model_path.exists():
            print(f"Error: Model {model_path} not found.")
            sys.exit(1)

        print(f"Loading model from {model_path}")
        restorer = DiacriticsRestorer(model_path=str(model_path))

        # Load test data
        _, val_texts = load_training_data(args.data_cache)
        evaluate_model(restorer, val_texts)

    else:
        # Load training data
        print("Loading training data...")
        train_texts, val_texts = load_training_data(args.data_cache)
        print(f"Loaded {len(train_texts)} training texts and {len(val_texts)} validation texts")

        # Create model
        print("\nInitializing model...")
        restorer = DiacriticsRestorer()

        # Train model
        print(f"\nStarting training for {args.epochs} epochs...")
        print(f"Configuration:")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Num layers: {args.num_layers}")

        start_time = datetime.now()

        restorer.train(
            train_texts=train_texts,
            val_texts=val_texts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=str(model_path)
        )

        training_time = datetime.now() - start_time
        print(f"\nTraining completed in {training_time}")

        # Evaluate model
        evaluate_model(restorer, val_texts)

        print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()