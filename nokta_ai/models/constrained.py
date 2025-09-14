"""
Constrained Turkish diacritics restoration model
Only focuses on specific character pairs: c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedDiacriticsModel(nn.Module):
    """
    Constrained model that only predicts diacritics for specific character pairs.
    Maintains input/output length consistency and preserves non-target characters.
    """

    # Define the Turkish diacritic pairs we care about
    DIACRITIC_PAIRS = {
        'c': ['c', 'ç'],
        'C': ['C', 'Ç'],
        'g': ['g', 'ğ'],
        'G': ['G', 'Ğ'],
        'i': ['i', 'ı'],
        'I': ['I', 'İ'],
        'o': ['o', 'ö'],
        'O': ['O', 'Ö'],
        's': ['s', 'ş'],
        'S': ['S', 'Ş'],
        'u': ['u', 'ü'],
        'U': ['U', 'Ü']
    }

    def __init__(self, context_size: int = 100, hidden_size: int = 128):
        """
        Args:
            context_size: Number of characters to look at around target character
            hidden_size: Hidden dimension size for the neural network
        """
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size

        # Character embedding (for context characters)
        self.char_embedding = nn.Embedding(256, 64)  # Support basic ASCII + extended

        # Bidirectional LSTM for context understanding
        self.context_lstm = nn.LSTM(
            64, hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        # Classification head for each diacritic pair (binary choice)
        self.classifiers = nn.ModuleDict({
            base_char: nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, len(variants))
            )
            for base_char, variants in self.DIACRITIC_PAIRS.items()
        })

    def forward(self, context_chars, target_chars):
        """
        Args:
            context_chars: (batch_size, seq_len, context_size) - context around each character
            target_chars: (batch_size, seq_len) - the characters to potentially restore

        Returns:
            predictions: Dict of predictions for each character type
        """
        batch_size, seq_len, _ = context_chars.shape

        # Embed context characters
        embedded = self.char_embedding(context_chars)  # (batch, seq, context, embed)
        embedded = embedded.view(batch_size * seq_len, self.context_size, -1)

        # Process through LSTM
        lstm_out, _ = self.context_lstm(embedded)  # (batch*seq, context, hidden*2)

        # Use the middle character's representation (center of context window)
        center_idx = self.context_size // 2
        features = lstm_out[:, center_idx, :]  # (batch*seq, hidden*2)
        features = features.view(batch_size, seq_len, -1)

        # Classify each character position
        predictions = {}
        for base_char, classifier in self.classifiers.items():
            # Only classify positions that have this base character
            mask = (target_chars == ord(base_char))

            if mask.any():
                masked_features = features[mask]  # (num_matches, hidden*2)
                if masked_features.size(0) > 0:
                    pred = classifier(masked_features)  # (num_matches, num_variants)
                    predictions[base_char] = {
                        'logits': pred,
                        'mask': mask
                    }

        return predictions


class ConstrainedDiacriticsRestorer:
    """High-level interface for constrained diacritics restoration"""

    def __init__(self, model_path: str = None, context_size: int = 100):
        self.device = torch.device('mps' if torch.backends.mps.is_available()
                                 else 'cuda' if torch.cuda.is_available()
                                 else 'cpu')

        self.context_size = context_size
        self.model = ConstrainedDiacriticsModel(context_size=context_size).to(self.device)

        if model_path:
            self.load_model(model_path)

    def restore_diacritics(self, text: str) -> str:
        """Restore diacritics in text while preserving length and non-target chars"""
        self.model.eval()

        if not text.strip():
            return text

        # Pad text for context
        pad_char = ' '
        padding = pad_char * (self.context_size // 2)
        padded_text = padding + text + padding

        # Extract context windows for each character
        contexts = []
        target_chars = []

        for i in range(len(text)):
            # Get context window around character i (in original text)
            start_idx = i  # Position in padded text
            context_window = padded_text[start_idx:start_idx + self.context_size]

            # Convert to character codes
            context_codes = [ord(c) for c in context_window]
            contexts.append(context_codes)
            target_chars.append(ord(text[i]))

        if not contexts:
            return text

        # Convert to tensors
        context_tensor = torch.tensor([contexts], dtype=torch.long).to(self.device)  # (1, len, context_size)
        target_tensor = torch.tensor([target_chars], dtype=torch.long).to(self.device)  # (1, len)

        # Get predictions
        with torch.no_grad():
            predictions = self.model(context_tensor, target_tensor)

        # Apply predictions to restore diacritics
        result_chars = list(text)  # Start with original text

        for base_char, pred_data in predictions.items():
            if 'logits' in pred_data and 'mask' in pred_data:
                logits = pred_data['logits']  # (num_matches, num_variants)
                mask = pred_data['mask'][0]  # (seq_len,) - remove batch dimension

                # Get predicted variants
                predicted_variants = torch.argmax(logits, dim=1)  # (num_matches,)

                # Apply predictions to result
                match_positions = torch.where(mask)[0]
                variants = ConstrainedDiacriticsModel.DIACRITIC_PAIRS[base_char]

                for pos_idx, variant_idx in zip(match_positions, predicted_variants):
                    result_chars[pos_idx] = variants[variant_idx]

        return ''.join(result_chars)

    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'context_size': self.context_size,
        }, path)
        print(f'Constrained model saved to {path}')

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        # Get the context size from checkpoint
        saved_context_size = checkpoint.get('context_size', 7)  # Default to 7 for older models

        # Recreate model with correct context size if different
        if saved_context_size != self.context_size:
            print(f'Updating context size from {self.context_size} to {saved_context_size}')
            self.context_size = saved_context_size
            self.model = ConstrainedDiacriticsModel(
                context_size=self.context_size,
                hidden_size=128  # Default hidden size
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Constrained model loaded from {path} (context_size: {self.context_size})')


def create_constrained_training_data(texts, context_size=100):
    """Create training data for the constrained model"""
    training_samples = []

    for text in texts:
        if len(text) < 3:
            continue

        # Normalize text
        text = text.strip()
        if not text:
            continue

        # Remove diacritics to create input
        input_text = remove_diacritics_simple(text)
        target_text = text

        if input_text == target_text:
            continue  # Skip if no diacritics to restore

        # Create context windows
        pad_char = ' '
        padding = pad_char * (context_size // 2)
        padded_input = padding + input_text + padding

        contexts = []
        targets = []
        labels = {}

        for i in range(len(input_text)):
            # Context window
            start_idx = i
            context_window = padded_input[start_idx:start_idx + context_size]
            context_codes = [ord(c) for c in context_window]
            contexts.append(context_codes)

            # Target character and label
            input_char = input_text[i]
            target_char = target_text[i]
            targets.append(ord(input_char))

            # If this character has a diacritic variant, record the label
            if input_char in ConstrainedDiacriticsModel.DIACRITIC_PAIRS:
                variants = ConstrainedDiacriticsModel.DIACRITIC_PAIRS[input_char]
                if target_char in variants:
                    if input_char not in labels:
                        labels[input_char] = []
                    labels[input_char].append({
                        'position': i,
                        'label': variants.index(target_char)
                    })

        if contexts and any(labels.values()):
            training_samples.append({
                'contexts': contexts,
                'targets': targets,
                'labels': labels,
                'input_text': input_text,
                'target_text': target_text
            })

    return training_samples


def remove_diacritics_simple(text):
    """Simple diacritic removal for the constrained pairs"""
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    for diacritic, base in replacements.items():
        text = text.replace(diacritic, base)
    return text