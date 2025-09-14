# nokta-ai

Turkish Diacritics Restoration with Neural Networks

A lightweight PyTorch-based neural network package for restoring diacritics in Turkish text. **nokta-ai** can accurately restore Turkish special characters (ç, ğ, ı, ö, ş, ü) from text where they have been removed or replaced with ASCII equivalents.

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

### From PyPI (Recommended)

```bash
pip install nokta-ai
```

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd nokta-ai

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,data]"
```

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon Mac (for MPS acceleration) or NVIDIA GPU (for CUDA)

## Package Structure

```
nokta-ai/
├── nokta_ai/
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Main restoration classes
│   ├── models/               # Neural network architectures (constrained model)
│   └── cli/                  # Command line interfaces
├── data/
│   ├── vikipedi_corpus.txt   # Turkish Wikipedia corpus
│   ├── aysnrgenc_turkishdeasciifier_train.txt  # Additional training data
│   └── *.pkl                 # Preprocessed datasets (generated)
├── models/                   # Trained model weights (generated)
└── scripts/
    └── prepare_training_data.py  # Script to prepare training cache from corpus files
```

## Usage

### Command Line Interface

After installation, nokta-ai provides three CLI commands:

#### 1. Interactive Restoration

```bash
# Interactive mode
nokta restore --model path/to/model.pth

# Direct text input
nokta restore --model path/to/model.pth --text "Bugun hava cok guzel"

# Process a file
nokta restore --model path/to/model.pth --input input.txt --output output.txt

# Benchmark model performance
nokta benchmark --model path/to/model.pth
```

#### 2. Training

```bash
# Train with prepared dataset
nokta-train --data-cache data/combined_cache.pkl --output my_model.pth

# Override training parameters
nokta-train \
    --data-cache data/combined_cache.pkl \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --context-size 96 \
    --hidden-size 256 \
    --output my_model.pth
```

#### 3. Model Evaluation

Evaluate your model's accuracy on test datasets:

```bash
# Evaluate on provided test dataset
nokta evaluate --model path/to/model.pth --test-file data/test_datasets/vikipedi_test.txt

# Save detailed evaluation results
nokta-evaluate --model path/to/model.pth \
                --test-file data/test_datasets/vikipedi_test.txt \
                --output evaluation_results.txt

# The evaluation provides:
# - Character-level accuracy (precise diacritic restoration)
# - Word-level accuracy (complete word correctness)
# - Per-sentence breakdown with detailed analysis
# - Overall performance statistics
```

#### 4. Advanced Inference

```bash
# Interactive mode
nokta-inference --model path/to/model.pth --interactive

# File processing
nokta-inference --model path/to/model.pth --file input.txt --output output.txt

# Direct text
nokta-inference --model path/to/model.pth --text "Turkiye'nin baskenti"
```

### Python API

```python
import nokta_ai

# Load trained model
restorer = nokta_ai.DiacriticsRestorer(model_path='path/to/model.pth')

# Restore diacritics
text_without = "Turkiye'nin baskenti Ankara'dir"
text_restored = restorer.restore_diacritics(text_without)
print(text_restored)  # "Türkiye'nin başkenti Ankara'dır"

# Process multiple texts
texts = ["Bugun hava guzel", "Cocuklar oynuyor"]
for text in texts:
    restored = restorer.restore_diacritics(text)
    print(f"{text} -> {restored}")

# Use mapper utilities
mapper = nokta_ai.TurkishDiacriticsMapper()
stripped = mapper.remove_diacritics("Günaydın dünya")
normalized = mapper.normalize_text("  MERHABA!!!  ")
```

## Test Datasets

The repository includes test datasets for evaluating model accuracy:

### Included Test Files

- **`data/test_datasets/vikipedi_test.txt`**: 290 lines of high-quality Turkish text from Wikipedia
- **Content**: Turkish constitutional history with proper diacritics
- **Purpose**: Benchmark model performance on real-world text

### Adding Your Own Test Files

To add custom test datasets:

1. **Place files in**: `data/test_datasets/`
2. **Format**: One sentence per line with correct diacritics
3. **Encoding**: UTF-8 text files
4. **Example structure**:
   ```
   Türkiye'nin başkenti Ankara'dır.
   Öğrenciler sınıfta ders çalışıyor.
   Çocuklar bahçede futbol oynuyorlar.
   ```

### Using Test Datasets

```bash
# Evaluate on included test dataset
nokta evaluate --model your_model.pth --test-file data/test_datasets/vikipedi_test.txt

# Evaluate on your custom test file
nokta evaluate --model your_model.pth --test-file data/test_datasets/your_test.txt

# Get detailed analysis
nokta-evaluate --model your_model.pth \
                --test-file data/test_datasets/vikipedi_test.txt \
                --output detailed_results.txt
```

The evaluation system automatically:
- Removes diacritics from test sentences to create input
- Restores diacritics using your model
- Compares against ground truth
- Reports character-level and word-level accuracy
- Provides detailed per-sentence analysis

## Training Data

### Preparing Training Data

Before training, prepare your corpus data:

```bash
# Combines vikipedi_corpus.txt and aysnrgenc_turkishdeasciifier_train.txt
python scripts/prepare_training_data.py

# This creates data/combined_cache.pkl for training
```

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