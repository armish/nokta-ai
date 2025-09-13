#!/usr/bin/env python3
"""
Training script that uses configuration file
"""

import os
import sys
import yaml
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from diacritics_restoration import DiacriticsRestorer, DiacriticsRestorationModel


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup computation device based on config"""
    if config['device']['prefer_mps'] and torch.backends.mps.is_available():
        return torch.device('mps')
    elif config['device']['prefer_cuda'] and torch.cuda.is_available():
        return torch.device('cuda')
    elif config['device']['cpu_fallback']:
        return torch.device('cpu')
    else:
        raise RuntimeError("No suitable device found")


def load_dataset(config):
    """Load dataset based on configuration"""
    cache_file = config['paths']['cache_file']

    if not Path(cache_file).exists():
        print(f"Cache file {cache_file} not found.")
        print("Creating dataset from corpus...")

        # Import data preparation module
        from prepare_wiki_data import prepare_wikipedia_dataset
        prepare_wikipedia_dataset(train_ratio=1.0 - config['training']['validation_split'])

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    return data.get('train', []), data.get('validation', [])


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    opt_config = config['optimization']
    train_config = config['training']

    if opt_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=opt_config['weight_decay']
        )
    elif opt_config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=opt_config['weight_decay']
        )
    elif opt_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=0.9,
            weight_decay=opt_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")

    return optimizer


def create_scheduler(optimizer, config, total_steps):
    """Create learning rate scheduler"""
    opt_config = config['optimization']

    if opt_config['scheduler'] == 'none':
        return None
    elif opt_config['scheduler'] == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif opt_config['scheduler'] == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        return None


def train_with_config(config_path: str = "config.yaml"):
    """Train model using configuration file"""
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded from:", config_path)

    # Setup device
    device = setup_device(config)
    print(f"Using device: {device}")

    # Create directories
    model_dir = Path(config['paths']['model_dir'])
    model_dir.mkdir(exist_ok=True)

    if config['logging']['save_logs']:
        log_dir = Path(config['paths']['log_dir'])
        log_dir.mkdir(exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    train_texts, val_texts = load_dataset(config)
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create model
    print("\nInitializing model...")
    restorer = DiacriticsRestorer()

    # Override model with config parameters
    model_config = config['model']
    restorer.model = DiacriticsRestorationModel(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)

    print(f"Model architecture:")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Dropout: {model_config['dropout']}")

    # Training parameters
    train_config = config['training']
    data_config = config['data']

    # Create custom training loop with config parameters
    from torch.utils.data import DataLoader
    from diacritics_restoration import TurkishDiacriticsDataset

    # Create datasets with config parameters
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False
    )

    # Setup optimization
    optimizer = create_optimizer(restorer.model, config)
    total_steps = len(train_loader) * train_config['epochs']
    scheduler = create_scheduler(optimizer, config, total_steps)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {train_config['epochs']} epochs...")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Learning rate: {train_config['learning_rate']}")

    start_time = datetime.now()

    # Training loop
    for epoch in range(train_config['epochs']):
        # Training phase
        restorer.model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = restorer.model(inputs)

            loss = criterion(
                outputs.reshape(-1, model_config['vocab_size']),
                targets.reshape(-1)
            )

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                restorer.model.parameters(),
                train_config['gradient_clip']
            )

            optimizer.step()

            if scheduler and config['optimization']['scheduler'] == 'cosine':
                scheduler.step()

            train_loss += loss.item()
            train_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{train_config['epochs']}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / train_batches

        # Validation phase
        restorer.model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = restorer.model(inputs)
                loss = criterion(
                    outputs.reshape(-1, model_config['vocab_size']),
                    targets.reshape(-1)
                )

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Update scheduler if using plateau
        if scheduler and config['optimization']['scheduler'] == 'plateau':
            scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            model_path = model_dir / "best_model.pth"
            restorer.save_model(str(model_path))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= train_config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint
        if (epoch + 1) % train_config['checkpoint_interval'] == 0:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch+1}.pth"
            restorer.save_model(str(checkpoint_path))
            print(f"Saved checkpoint at epoch {epoch+1}")

    # Save final model
    final_model_path = model_dir / "final_model.pth"
    restorer.save_model(str(final_model_path))

    training_time = datetime.now() - start_time
    print(f"\nTraining completed in {training_time}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train with configuration file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Configuration file {args.config} not found")
        sys.exit(1)

    train_with_config(args.config)


if __name__ == "__main__":
    main()