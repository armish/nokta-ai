#!/usr/bin/env python3
"""
Turkish Diacritics Restoration Neural Network
Restores diacritics (ç, ğ, ı, ö, ş, ü) from normalized text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import unicodedata
import re
import json
from pathlib import Path


class TurkishDiacriticsMapper:
    """Handles mapping between Turkish characters with/without diacritics"""

    # Mapping of characters without diacritics to their possible diacritic versions
    DIACRITIC_MAP = {
        'c': ['c', 'ç'],
        'g': ['g', 'ğ'],
        'i': ['i', 'ı', 'İ', 'I'],
        'o': ['o', 'ö'],
        's': ['s', 'ş'],
        'u': ['u', 'ü'],
        'C': ['C', 'Ç'],
        'G': ['G', 'Ğ'],
        'I': ['I', 'İ', 'ı', 'i'],
        'O': ['O', 'Ö'],
        'S': ['S', 'Ş'],
        'U': ['U', 'Ü']
    }

    # All unique Turkish characters we'll work with
    TURKISH_CHARS = set('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ')

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove diacritics from Turkish text"""
        replacements = {
            'ç': 'c', 'Ç': 'C',
            'ğ': 'g', 'Ğ': 'G',
            'ı': 'i', 'İ': 'I',
            'ö': 'o', 'Ö': 'O',
            'ş': 's', 'Ş': 'S',
            'ü': 'u', 'Ü': 'U'
        }
        for char_with, char_without in replacements.items():
            text = text.replace(char_with, char_without)
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text while preserving Turkish characters"""
        # Keep only letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class CharacterTokenizer:
    """Character-level tokenizer with context window"""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.build_vocab()

    def build_vocab(self):
        """Build character vocabulary"""
        # Special tokens
        special_tokens = [self.pad_token, self.unk_token]

        # Common characters including Turkish special chars
        chars = list('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ')
        chars += list('0123456789')
        chars += list(' .,!?;:\'\"-\n\t')

        all_chars = special_tokens + chars

        for idx, char in enumerate(all_chars[:self.vocab_size]):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

    def encode(self, text: str) -> List[int]:
        """Convert text to token indices"""
        return [self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
                for char in text]

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text"""
        return ''.join([self.idx_to_char.get(idx, self.unk_token)
                       for idx in indices if idx != self.char_to_idx[self.pad_token]])


class TurkishDiacriticsDataset(Dataset):
    """Dataset for Turkish diacritics restoration"""

    def __init__(self, texts: List[str], tokenizer: CharacterTokenizer,
                 context_window: int = 50, stride: int = 25):
        self.tokenizer = tokenizer
        self.context_window = context_window
        self.stride = stride
        self.mapper = TurkishDiacriticsMapper()
        self.samples = []

        # Process texts into training samples
        for text in texts:
            self._process_text(text)

    def _process_text(self, text: str):
        """Process text into sliding window samples"""
        # Normalize and prepare text
        text = self.mapper.normalize_text(text)
        if len(text) < self.context_window:
            return

        # Create original and stripped versions
        original = text
        stripped = self.mapper.remove_diacritics(text)

        # Create sliding windows
        for i in range(0, len(text) - self.context_window, self.stride):
            window_stripped = stripped[i:i + self.context_window]
            window_original = original[i:i + self.context_window]

            # Tokenize
            input_ids = self.tokenizer.encode(window_stripped)
            target_ids = self.tokenizer.encode(window_original)

            if len(input_ids) == self.context_window:
                self.samples.append((input_ids, target_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, target_ids = self.samples[idx]
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_ids, dtype=torch.long))


class DiacriticsRestorationModel(nn.Module):
    """Neural network model for diacritics restoration"""

    def __init__(self, vocab_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Bidirectional LSTM for context understanding
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism for focusing on relevant context
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)

        # LSTM encoding
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size * 2)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Output projection
        logits = self.output_projection(attn_out)  # (batch_size, seq_len, vocab_size)

        return logits


class DiacriticsRestorer:
    """Main class for training and using the diacritics restoration model"""

    def __init__(self, model_path: str = None):
        # Prioritize MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'Using device: {self.device}')
        self.tokenizer = CharacterTokenizer()
        self.model = DiacriticsRestorationModel(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=256,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        self.mapper = TurkishDiacriticsMapper()

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def train(self, train_texts: List[str], val_texts: List[str] = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              save_path: str = 'diacritics_model.pth'):
        """Train the diacritics restoration model"""

        # Create datasets
        train_dataset = TurkishDiacriticsDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts:
            val_dataset = TurkishDiacriticsDataset(val_texts, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.char_to_idx[self.tokenizer.pad_token])

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)

                # Calculate loss
                loss = criterion(outputs.reshape(-1, self.tokenizer.vocab_size),
                               targets.reshape(-1))

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')

            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, criterion)
                print(f'Validation Loss: {val_loss:.4f}')

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f'{save_path}.epoch{epoch+1}')

        # Save final model
        self.save_model(save_path)
        print(f'Training complete. Model saved to {save_path}')

    def _validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs.reshape(-1, self.tokenizer.vocab_size),
                               targets.reshape(-1))
                total_loss += loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def restore_diacritics(self, text: str, context_window: int = 50) -> str:
        """Restore diacritics in the given text"""
        self.model.eval()

        # Normalize and remove diacritics
        normalized = self.mapper.normalize_text(text)
        stripped = self.mapper.remove_diacritics(normalized)

        if len(stripped) <= context_window:
            # Process entire text at once
            return self._restore_window(stripped)

        # Process in overlapping windows
        result = []
        stride = context_window // 2

        for i in range(0, len(stripped), stride):
            window = stripped[i:i + context_window]
            if len(window) < 10:  # Skip very small windows
                result.append(window)
                continue

            restored = self._restore_window(window)

            if i == 0:
                result.append(restored)
            else:
                # Use middle portion to avoid edge effects
                start = stride // 2 if i + context_window < len(stripped) else 0
                result.append(restored[start:])

        return ''.join(result)

    def _restore_window(self, text: str) -> str:
        """Restore diacritics in a single window of text"""
        with torch.no_grad():
            # Tokenize
            input_ids = self.tokenizer.encode(text)
            inputs = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            # Get predictions
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim=-1)

            # Decode
            restored = self.tokenizer.decode(predictions[0].cpu().tolist())

        return restored

    def save_model(self, path: str):
        """Save model checkpoint"""
        # Move model to CPU before saving to ensure compatibility
        self.model.to('cpu')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_char_to_idx': self.tokenizer.char_to_idx,
            'tokenizer_idx_to_char': self.tokenizer.idx_to_char,
        }, path)
        # Move model back to original device
        self.model.to(self.device)
        print(f'Model saved to {path}')

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer.char_to_idx = checkpoint['tokenizer_char_to_idx']
        self.tokenizer.idx_to_char = checkpoint['tokenizer_idx_to_char']
        print(f'Model loaded from {path}')


def test_basic_functionality():
    """Test the basic functionality with sample text"""
    # Sample Turkish texts
    sample_texts = [
        "Türkiye Cumhuriyeti, Güneydoğu Avrupa ve Batı Asya'da yer alan ülkedir.",
        "İstanbul, Türkiye'nin en kalabalık şehri ve ekonomik merkezidir.",
        "Çocuklar bahçede oynuyorlar. Öğretmen sınıfta ders anlatıyor.",
        "Büyük şehirlerde yaşamak zor olabiliyor. Özellikle trafik çok yoğun.",
        "Güzel bir gün geçirdik. Arkadaşlarımla buluştuk ve yemek yedik.",
    ]

    print("Testing Turkish Diacritics Restoration")
    print("=" * 50)

    # Test diacritic removal
    mapper = TurkishDiacriticsMapper()
    for text in sample_texts[:2]:
        stripped = mapper.remove_diacritics(text)
        print(f"Original: {text}")
        print(f"Stripped: {stripped}")
        print()

    # Create and test the model with minimal training
    restorer = DiacriticsRestorer()

    # Train on sample data (in practice, you'd use much more data)
    print("Training on sample data...")
    restorer.train(sample_texts * 20, epochs=2, batch_size=4)

    # Test restoration
    print("\nTesting restoration:")
    test_text = "Bugun hava cok guzel. Cocuklar parkta oynuyorlar."
    print(f"Input:    {test_text}")
    restored = restorer.restore_diacritics(test_text)
    print(f"Restored: {restored}")


if __name__ == "__main__":
    test_basic_functionality()