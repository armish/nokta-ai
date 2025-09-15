"""
Constrained Turkish diacritics restoration model
Only focuses on specific character pairs: c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention layer for capturing long-range dependencies"""

    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Residual connection
        residual = x

        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.w_o(context)
        output = self.dropout(output)

        # Add & Norm
        output = self.layer_norm(output + residual)

        return output


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

    def __init__(self, context_size: int = 100, hidden_size: int = 128,
                 num_lstm_layers: int = 2, use_attention: bool = True):
        """
        Args:
            context_size: Number of characters to look at around target character
            hidden_size: Hidden dimension size for the neural network
            num_lstm_layers: Number of LSTM layers (default: 2, expert recommends 2-4)
            use_attention: Whether to use self-attention layer (default: True)
        """
        super().__init__()
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention

        # Character embedding (for context characters)
        # Expert recommends 128 for embedding dim
        self.char_embedding = nn.Embedding(256, 128)  # Increased from 64 to 128

        # Bidirectional LSTM for context understanding
        # Expert recommends dropout of 0.25 between layers
        dropout = 0.25 if num_lstm_layers > 1 else 0
        self.context_lstm = nn.LSTM(
            128, hidden_size,  # Updated embedding dim
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Self-attention layer (optional but recommended by expert)
        # Expert: "single lightweight self-attention layer helps on long compounds"
        if use_attention:
            # d_model = hidden_size * 2 (bidirectional)
            self.self_attention = MultiHeadSelfAttention(
                d_model=hidden_size * 2,
                num_heads=4,  # Expert recommends 4 heads
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

        # Reshape back to (batch, seq, context, hidden*2) for attention
        lstm_out = lstm_out.view(batch_size, seq_len, self.context_size, -1)

        # Use the middle character's representation (center of context window)
        center_idx = self.context_size // 2
        features = lstm_out[:, :, center_idx, :]  # (batch, seq, hidden*2)

        # Apply self-attention if enabled (expert recommends this)
        if self.use_attention:
            features = self.self_attention(features)  # (batch, seq, hidden*2)

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

    def __init__(self, model_path: str = None, context_size: int = 100,
                 hidden_size: int = 128, num_lstm_layers: int = 2,
                 use_attention: bool = True):
        self.device = torch.device('mps' if torch.backends.mps.is_available()
                                 else 'cuda' if torch.cuda.is_available()
                                 else 'cpu')

        self.context_size = context_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention

        self.model = ConstrainedDiacriticsModel(
            context_size=context_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            use_attention=use_attention
        ).to(self.device)

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
            'hidden_size': self.hidden_size,
            'num_lstm_layers': self.num_lstm_layers,
            'use_attention': self.use_attention,
        }, path)
        print(f'Constrained model saved to {path}')

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        # Get model configuration from checkpoint
        saved_context_size = checkpoint.get('context_size', 100)
        saved_hidden_size = checkpoint.get('hidden_size', 128)
        saved_num_lstm_layers = checkpoint.get('num_lstm_layers', 2)
        saved_use_attention = checkpoint.get('use_attention', False)  # Default False for old models

        # Recreate model with saved configuration
        if (saved_context_size != self.context_size or
            saved_hidden_size != self.hidden_size or
            saved_num_lstm_layers != self.num_lstm_layers or
            saved_use_attention != self.use_attention):
            print(f'Loading model configuration from checkpoint:')
            print(f'  context_size: {saved_context_size}')
            print(f'  hidden_size: {saved_hidden_size}')
            print(f'  num_lstm_layers: {saved_num_lstm_layers}')
            print(f'  use_attention: {saved_use_attention}')

            self.context_size = saved_context_size
            self.hidden_size = saved_hidden_size
            self.num_lstm_layers = saved_num_lstm_layers
            self.use_attention = saved_use_attention

            self.model = ConstrainedDiacriticsModel(
                context_size=self.context_size,
                hidden_size=self.hidden_size,
                num_lstm_layers=self.num_lstm_layers,
                use_attention=self.use_attention
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Constrained model loaded from {path}')


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