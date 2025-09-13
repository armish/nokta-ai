"""
Character-level tokenizer for Turkish text
"""

from typing import List


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