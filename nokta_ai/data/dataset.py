"""
Dataset classes for Turkish diacritics restoration training
"""

import torch
from torch.utils.data import Dataset
from typing import List
import re
from .tokenizer import CharacterTokenizer


class TurkishDiacriticsMapper:
    """Handles mapping between Turkish characters with/without diacritics"""

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
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


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