"""
Core functionality for Turkish diacritics restoration
"""

import torch
import re
from typing import List
from pathlib import Path
from .models.constrained import ConstrainedDiacriticsRestorer, remove_diacritics_simple


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
        return remove_diacritics_simple(text)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text while preserving Turkish characters"""
        # Keep only letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class DiacriticsRestorer:
    """Main class for Turkish diacritics restoration using constrained model"""

    def __init__(self, model_path: str = None, context_size: int = 100):
        """Initialize the constrained diacritics restorer"""
        self.restorer = ConstrainedDiacriticsRestorer(model_path=model_path, context_size=context_size)
        self.mapper = TurkishDiacriticsMapper()

    def restore_diacritics(self, text: str) -> str:
        """Restore diacritics in the given text"""
        # Normalize text first
        normalized = self.mapper.normalize_text(text)

        # Use constrained restorer
        return self.restorer.restore_diacritics(normalized)

    def save_model(self, path: str):
        """Save model checkpoint"""
        self.restorer.save_model(path)

    def load_model(self, path: str):
        """Load model checkpoint"""
        self.restorer.load_model(path)