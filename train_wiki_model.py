#!/usr/bin/env python3
"""
Train Turkish diacritics restoration model on Wikipedia data
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import torch
from diacritics_restoration import DiacriticsRestorer, TurkishDiacriticsMapper


def load_wikipedia_dataset(cache_file: str = "data/wikipedia_dataset_cache.pkl"):
    """Load Wikipedia training data from cache"""
    if not Path(cache_file).exists():
        print(f"Error: Cache file {cache_file} not found.")
        print("Please run prepare_wiki_data.py first to create the dataset.")
        sys.exit(1)

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    return data['train'], data['validation']


def evaluate_on_real_text(restorer):
    """Evaluate model on real Turkish sentences"""
    mapper = TurkishDiacriticsMapper()

    test_sentences = [
        "Türkiye'nin başkenti Ankara'dır.",
        "İstanbul Boğazı dünyanın en güzel yerlerinden biridir.",
        "Öğrenciler üniversitede çok çalışıyorlar.",
        "Güneşli havalarda dışarıda olmak çok güzel.",
        "Çocuklar bahçede oynayarak eğleniyorlar.",
        "Türk mutfağı dünyada çok ünlüdür.",
        "Atatürk, Türkiye Cumhuriyeti'nin kurucusudur.",
        "Kapadokya'daki peri bacaları görülmeye değer.",
        "Akdeniz kıyıları yaz aylarında çok kalabalık olur.",
        "Müzik dinlemek insanı rahatlatır."
    ]

    print("\n" + "=" * 60)
    print("Evaluation on Real Turkish Sentences")
    print("=" * 60)

    total_correct = 0
    total_chars = 0

    for original in test_sentences:
        # Remove diacritics
        stripped = mapper.remove_diacritics(original)

        # Restore
        restored = restorer.restore_diacritics(stripped)

        # Calculate accuracy
        correct = sum(1 for c1, c2 in zip(original, restored) if c1 == c2)
        accuracy = correct / len(original) * 100

        total_correct += correct
        total_chars += len(original)

        print(f"\nOriginal:  {original}")
        print(f"Stripped:  {stripped}")
        print(f"Restored:  {restored}")
        print(f"Accuracy:  {accuracy:.1f}%")

    overall_accuracy = total_correct / total_chars * 100
    print(f"\n{'=' * 60}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Train on Wikipedia corpus')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-size', type=int, default=512,
                       help='Hidden size (default: 512)')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of LSTM layers (default: 4)')
    parser.add_argument('--context-window', type=int, default=100,
                       help='Context window size (default: 100)')
    parser.add_argument('--model-name', type=str, default='wiki_diacritics_model',
                       help='Model name (default: wiki_diacritics_model)')

    args = parser.parse_args()

    # Setup paths
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{args.model_name}.pth'

    # Check device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading Wikipedia dataset...")
    train_texts, val_texts = load_wikipedia_dataset()
    print(f"Loaded {len(train_texts)} training and {len(val_texts)} validation samples")

    # Show sample of training data
    print("\nSample training text:")
    print(train_texts[0][:200] + "...")

    # Create model with larger capacity for Wikipedia
    print(f"\nInitializing model with hidden_size={args.hidden_size}, num_layers={args.num_layers}")
    restorer = DiacriticsRestorer()

    # Update model architecture
    restorer.model = restorer.model.__class__(
        vocab_size=restorer.tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.2
    ).to(device)

    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
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

    # Evaluate
    evaluate_on_real_text(restorer)

    print(f"\nModel saved to {model_path}")
    print("\nTo use the model for inference:")
    print(f"python inference.py --model {model_path} --interactive")


if __name__ == "__main__":
    main()