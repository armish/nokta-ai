"""
Core functionality for Turkish diacritics restoration
"""

import torch
import re
from typing import List
from pathlib import Path
from .models import DiacriticsRestorationModel
from .data import CharacterTokenizer


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


class DiacriticsRestorer:
    """Main class for training and using the diacritics restoration model"""

    def __init__(self, model_path: str = None, device: str = None):
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            # Auto-detect best device
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