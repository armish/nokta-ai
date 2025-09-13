# Turkish Diacritics Restoration with Neural Networks

A PyTorch-based neural network system for restoring diacritics in Turkish text. This tool can accurately restore Turkish special characters (ç, ğ, ı, ö, ş, ü) from text where they have been removed or replaced with ASCII equivalents.

## Overview

This project implements a character-level sequence-to-sequence model using bidirectional LSTMs with attention mechanism to restore diacritics in Turkish text. Unlike rule-based or large language model approaches, this solution is:

- **Lightweight**: Small model size (~10-50MB depending on configuration)
- **Fast**: Processes text in real-time using GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA)
- **Accurate**: Achieves 90%+ accuracy with proper training data
- **Self-contained**: No external API dependencies

## Architecture

### Model Components

1. **Character-level Tokenization**: Works at the character level for fine-grained control
2. **Bidirectional LSTM**: Captures both forward and backward context (2-4 layers)
3. **Multi-head Attention**: Focuses on relevant parts of the input sequence
4. **Sliding Window Processing**: Handles texts of arbitrary length

### Technical Specifications

- **Input**: Turkish text with diacritics removed (ASCII-normalized)
- **Output**: Turkish text with diacritics restored
- **Context Window**: Configurable (default: 50-100 characters)
- **Vocabulary Size**: 256 characters (covers Turkish alphabet + common punctuation)
- **Hidden Size**: Configurable (256-512 dimensions)
- **Attention Heads**: 8

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon Mac (for MPS acceleration) or NVIDIA GPU (for CUDA)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd karakter-ai

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
karakter-ai/
├── diacritics_restoration.py  # Core model and training logic
├── prepare_wiki_data.py       # Wikipedia corpus preparation
├── train_wiki_model.py         # Training script for Wikipedia data
├── inference.py                # Inference and evaluation tools
├── data/
│   ├── vikipedi_corpus.txt    # Turkish Wikipedia corpus
│   └── wikipedia_dataset_cache.pkl  # Preprocessed training data
└── models/
    └── wiki_diacritics_model.pth  # Trained model weights
```

## Usage

### 1. Data Preparation

Prepare your Turkish text corpus for training:

```bash
# Using Wikipedia corpus
python prepare_wiki_data.py

# This creates: data/wikipedia_dataset_cache.pkl
```

### 2. Training

Train the model on your data:

```bash
# Basic training
python train_wiki_model.py --epochs 50 --batch-size 64

# Advanced training with custom parameters
python train_wiki_model.py \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001 \
    --hidden-size 512 \
    --num-layers 4 \
    --model-name my_model
```

Training parameters:
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate (default: 0.001)
- `--hidden-size`: LSTM hidden dimension size (default: 512)
- `--num-layers`: Number of LSTM layers (default: 4)
- `--context-window`: Character context window (default: 100)

### 3. Inference

Use the trained model to restore diacritics:

```bash
# Interactive mode
python inference.py --model models/wiki_diacritics_model.pth --interactive

# Process a file
python inference.py \
    --model models/wiki_diacritics_model.pth \
    --input text_without_diacritics.txt \
    --output restored_text.txt

# Direct text restoration
python inference.py \
    --model models/wiki_diacritics_model.pth \
    --text "Bugun hava cok guzel"

# Benchmark mode
python inference.py --model models/wiki_diacritics_model.pth --benchmark
```

## Example Usage in Python

```python
from diacritics_restoration import DiacriticsRestorer

# Load trained model
restorer = DiacriticsRestorer(model_path='models/wiki_diacritics_model.pth')

# Restore diacritics
text_without = "Turkiye'nin baskenti Ankara'dir"
text_restored = restorer.restore_diacritics(text_without)
print(text_restored)  # "Türkiye'nin başkenti Ankara'dır"
```

## Training Data

The system can be trained on any Turkish text corpus. Best results are achieved with:

1. **Wikipedia dumps**: Comprehensive vocabulary and proper spelling
2. **News articles**: Contemporary language usage
3. **Books and literature**: Diverse writing styles
4. **Web crawls**: Informal language patterns

### Data Requirements

- Minimum: 1MB of Turkish text
- Recommended: 10MB+ for good coverage
- Optimal: 100MB+ for production use

## Performance

### Hardware Acceleration

The system automatically detects and uses available acceleration:

1. **MPS (Metal Performance Shaders)**: Apple Silicon Macs
2. **CUDA**: NVIDIA GPUs
3. **CPU**: Fallback for other systems

### Benchmarks

On Apple M1/M2 with MPS:
- Training speed: ~1000 samples/second
- Inference speed: ~10,000 characters/second

### Accuracy

With proper training (50+ epochs on Wikipedia data):
- Character-level accuracy: 95%+
- Word-level accuracy: 90%+

## Model Improvements

To improve model performance:

1. **More Training Data**: Use larger Turkish corpora
2. **Longer Training**: Increase epochs (100-200)
3. **Larger Model**: Increase hidden_size (512-1024) and num_layers (4-6)
4. **Data Augmentation**: Add synthetic errors and variations
5. **Ensemble Methods**: Train multiple models and combine predictions

## Troubleshooting

### Common Issues

1. **Low accuracy after training**
   - Solution: Train for more epochs (50+)
   - Use more training data
   - Increase model capacity

2. **MPS not detected on Mac**
   - Ensure PyTorch 2.0+ is installed
   - Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

3. **Out of memory errors**
   - Reduce batch_size
   - Reduce context_window
   - Use gradient accumulation

## Implementation Details

### Character Mapping

The system handles these Turkish-specific transformations:
- ç ↔ c, Ç ↔ C
- ğ ↔ g, Ğ ↔ G
- ı ↔ i, İ ↔ I
- ö ↔ o, Ö ↔ O
- ş ↔ s, Ş ↔ S
- ü ↔ u, Ü ↔ U

### Sliding Window Algorithm

For long texts, the model uses overlapping windows:
1. Divide text into overlapping segments
2. Process each segment independently
3. Merge results using weighted averaging in overlap regions
4. Ensures consistent predictions across boundaries

### Attention Mechanism

The multi-head attention layer helps the model:
- Focus on relevant context for ambiguous cases
- Learn long-range dependencies
- Handle Turkish vowel harmony rules

## Future Enhancements

- [ ] Support for other Turkic languages (Azerbaijani, Kazakh)
- [ ] Web API for real-time diacritics restoration
- [ ] Browser extension for automatic correction
- [ ] Mobile keyboard integration
- [ ] Transformer-based architecture option
- [ ] Active learning for continuous improvement

## License

MIT License

## Citation

If you use this work in your research, please cite:

```bibtex
@software{turkish_diacritics_nn,
  title = {Turkish Diacritics Restoration with Neural Networks},
  year = {2024},
  url = {https://github.com/yourusername/karakter-ai}
}
```

## Contact

For questions and contributions, please open an issue on GitHub.