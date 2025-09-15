"""
Constrained Turkish diacritics restoration model
Only focuses on specific character pairs: c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


def safe_ord(char):
    """Convert character to ordinal, clamping to valid embedding range (0-255)"""
    code = ord(char)
    if code > 255:
        return 32  # ASCII space as fallback
    return code


def filter_text(text):
    """Filter text to ensure all characters are within valid range"""
    filtered_chars = []
    for char in text:
        code = ord(char)
        if code > 255:
            # Replace with closest ASCII equivalent or space
            if char.lower() in 'çğışöü':
                # Keep Turkish diacritics as they're handled by remove_diacritics_simple
                filtered_chars.append(char)
            else:
                filtered_chars.append(' ')  # Replace with space
        else:
            filtered_chars.append(char)
    return ''.join(filtered_chars)


def preserve_case_pattern(original_text, restored_text):
    """Apply the case pattern from original text to restored text with Turkish-specific handling"""
    result = []
    for i, (orig_char, restored_char) in enumerate(zip(original_text, restored_text)):
        if orig_char.isupper():
            # Turkish-specific uppercase handling
            if restored_char == 'i':
                result.append('İ')  # i → İ (with dot)
            elif restored_char == 'ı':
                result.append('I')  # ı → I (without dot)
            else:
                result.append(restored_char.upper())
        else:
            # Turkish-specific lowercase handling
            if restored_char == 'İ':
                result.append('i')  # İ → i (with dot)
            elif restored_char == 'I':
                result.append('ı')  # I → ı (without dot)
            else:
                result.append(restored_char.lower())
    return ''.join(result)


def normalize_case_for_training(text):
    """Convert text to lowercase for training with Turkish-specific handling"""
    case_pattern = [char.isupper() for char in text]

    # Turkish-specific lowercase conversion
    result = []
    for char in text:
        if char == 'İ':
            result.append('i')  # İ → i (with dot)
        elif char == 'I':
            result.append('ı')  # I → ı (without dot)
        else:
            result.append(char.lower())

    lowercase_text = ''.join(result)
    return lowercase_text, case_pattern


def detect_optimal_device(verbose=True):
    """
    Detect optimal device for PyTorch operations with detailed logging and warnings.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU

    Args:
        verbose (bool): Whether to print device detection information

    Returns:
        torch.device: Optimal device for computations
    """

    # Check for Apple Silicon MPS
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("🚀 Detected Apple Silicon with MPS acceleration")
            print("   - Excellent performance for training and inference")
            print("   - Optimized for M1/M2/M3 processors")
        return device

    # Check for NVIDIA CUDA
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_device_count = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if verbose:
            print(f"🚀 Detected NVIDIA GPU with CUDA acceleration")
            print(f"   - Device: {cuda_device_name}")
            print(f"   - GPU Memory: {cuda_memory:.1f} GB")
            print(f"   - Available GPUs: {cuda_device_count}")

            # GPU-specific recommendations
            if cuda_memory > 40:  # >40GB memory
                print("   - 🎯 High-memory GPU detected! Excellent for large-scale training")
                print("   - Recommended: Use larger batch sizes (64-256)")
                print("   - Try: --cuda-config for optimized settings")
            elif "RTX" in cuda_device_name or "GTX" in cuda_device_name:
                print("   - Consumer GPU detected - good performance expected")
            elif "Tesla" in cuda_device_name or "V100" in cuda_device_name or "A100" in cuda_device_name:
                print("   - Professional GPU detected - excellent for training")

        return device

    # Fallback to CPU with warnings
    else:
        device = torch.device('cpu')
        if verbose:
            print("⚠️  Using CPU - Limited performance expected")
            print("   - Training will be significantly slower")
            print("   - Consider using smaller models (context=20, hidden=128)")
            print("   - Recommended: Use --no-attention for faster training")
            print("   - For production: Consider cloud GPU instances")

        # Issue a warning for performance
        warnings.warn(
            "Training on CPU will be very slow. Consider using a GPU-enabled machine "
            "or cloud instance with NVIDIA CUDA or Apple Silicon MPS support.",
            UserWarning,
            stacklevel=2
        )

        return device


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
    # Simplified to lowercase only - case will be preserved separately
    DIACRITIC_PAIRS = {
        'c': ['c', 'ç'],
        'g': ['g', 'ğ'],
        'i': ['i', 'ı'],
        'o': ['o', 'ö'],
        's': ['s', 'ş'],
        'u': ['u', 'ü']
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
            mask = (target_chars == safe_ord(base_char))

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
                 use_attention: bool = True, verbose_device: bool = True):
        # Use enhanced device detection
        self.device = detect_optimal_device(verbose=verbose_device)

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

        # Store original case pattern and normalize to lowercase for processing
        original_text = text
        normalized_text, case_pattern = normalize_case_for_training(text)

        # Pad normalized text for context
        pad_char = ' '
        padding = pad_char * (self.context_size // 2)
        padded_text = padding + normalized_text + padding

        # Extract context windows for each character
        contexts = []
        target_chars = []

        for i in range(len(normalized_text)):
            # Get context window around character i (in normalized text)
            start_idx = i  # Position in padded text
            context_window = padded_text[start_idx:start_idx + self.context_size]

            # Convert to character codes with bounds checking
            context_codes = [safe_ord(c) for c in context_window]
            contexts.append(context_codes)
            target_chars.append(safe_ord(normalized_text[i]))

        if not contexts:
            return original_text

        # Convert to tensors
        context_tensor = torch.tensor([contexts], dtype=torch.long).to(self.device)  # (1, len, context_size)
        target_tensor = torch.tensor([target_chars], dtype=torch.long).to(self.device)  # (1, len)

        # Get predictions
        with torch.no_grad():
            predictions = self.model(context_tensor, target_tensor)

        # Apply predictions to restore diacritics (working with normalized lowercase text)
        result_chars = list(normalized_text)  # Start with normalized text

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

        # Restore original case pattern
        restored_lowercase = ''.join(result_chars)
        final_result = preserve_case_pattern(original_text, restored_lowercase)

        return final_result

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
    invalid_chars_count = 0


    for text in texts:
        if len(text) < 3:
            continue

        # Normalize and filter text
        text = text.strip()
        if not text:
            continue

        # Filter text to remove characters outside embedding range
        text = filter_text(text)

        # Normalize case for simplified training
        normalized_text, case_pattern = normalize_case_for_training(text)

        # Remove diacritics to create input
        input_text = remove_diacritics_simple(normalized_text)
        target_text = normalized_text

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
            # Context window with safe character conversion
            start_idx = i
            context_window = padded_input[start_idx:start_idx + context_size]
            context_codes = [safe_ord(c) for c in context_window]
            contexts.append(context_codes)

            # Target character and label
            input_char = input_text[i]
            target_char = target_text[i]
            targets.append(safe_ord(input_char))

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
    """Simple diacritic removal for the constrained pairs (lowercase only)"""
    replacements = {
        'ç': 'c',
        'ğ': 'g',
        'ı': 'i',
        'ö': 'o',
        'ş': 's',
        'ü': 'u'
    }
    for diacritic, base in replacements.items():
        text = text.replace(diacritic, base)
    return text