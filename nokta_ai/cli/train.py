#!/usr/bin/env python3
"""
Training CLI for nokta-ai
"""

import argparse
import sys
import yaml
import pickle
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from ..core import DiacriticsRestorer
from ..models import DiacriticsRestorationModel
from ..data import TurkishDiacriticsDataset


def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(cache_file: str):
    """Load dataset from cache"""
    if not Path(cache_file).exists():
        print(f"Error: Dataset cache {cache_file} not found")
        print("Please prepare your dataset first")
        sys.exit(1)

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return data.get('train', []), data.get('validation', [])
    else:
        # Legacy format - split manually
        split_idx = int(len(data) * 0.8)
        return data[:split_idx], data[split_idx:]


def train_model(args):
    """Train the diacritics restoration model"""
    # Load configuration
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file {args.config} not found")
            sys.exit(1)
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'model': {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.2},
            'training': {'epochs': 30, 'batch_size': 64, 'learning_rate': 0.001},
            'data': {'context_window': 100, 'stride': 50}
        }

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    train_texts, val_texts = load_dataset(args.data_cache)
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create model
    print("Initializing model...")
    restorer = DiacriticsRestorer()

    # Override with config parameters
    model_config = config['model']
    restorer.model = DiacriticsRestorationModel(
        vocab_size=restorer.tokenizer.vocab_size,
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    print(f"Model: {model_config['hidden_size']}h x {model_config['num_layers']}L")

    # Create datasets
    data_config = config['data']
    train_dataset = TurkishDiacriticsDataset(
        train_texts,
        restorer.tokenizer,
        context_window=data_config['context_window'],
        stride=data_config['stride']
    )

    val_dataset = TurkishDiacriticsDataset(
        val_texts,
        restorer.tokenizer,
        context_window=data_config['context_window'],
        stride=data_config['stride']
    )

    # Train
    train_config = config['training']
    print(f"Training for {train_config['epochs']} epochs...")

    start_time = datetime.now()

    restorer.train(
        train_texts=train_texts,
        val_texts=val_texts,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        learning_rate=train_config['learning_rate'],
        save_path=args.output
    )

    training_time = datetime.now() - start_time
    print(f"Training completed in {training_time}")
    print(f"Model saved to {args.output}")


def main():
    """Main training CLI"""
    parser = argparse.ArgumentParser(
        prog='nokta-train',
        description='Train Turkish diacritics restoration model'
    )

    parser.add_argument('--data-cache', type=str, required=True,
                       help='Path to dataset cache file')
    parser.add_argument('--output', type=str, default='model.pth',
                       help='Output model path')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')

    # Training parameters (override config)
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate')

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()