# nokta-ai

Turkish Diacritics Restoration with Neural Networks

A lightweight PyTorch-based neural network package for restoring diacritics in Turkish text. **nokta-ai** can accurately restore Turkish special characters (ç, ğ, ı, ö, ş, ü) from text where they have been removed or replaced with ASCII equivalents.

## Overview

This project implements a character-level sequence-to-sequence model using bidirectional LSTMs with multi-head self-attention to restore diacritics in Turkish text. Unlike rule-based or large language model approaches, this solution is:

- **Lightweight**: Small model size (~10-15MB depending on configuration)
- **Fast**: Processes text in real-time using GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA)
- **Accurate**: Achieves 95%+ accuracy with proper training data
- **Self-contained**: No external API dependencies

## Recent Improvements

### v2.0 - Major Architecture Enhancements

- **Case Normalization**: 50% reduction in model complexity by processing lowercase text with case pattern restoration
- **Turkish-Specific Case Handling**: Proper i/İ and ı/I mappings (critical for Turkish orthography)
- **CUDA Compatibility**: Fixed character embedding bounds checking for stable GPU training
- **Binary Classification with Loss Penalties**: Model learns when NOT to add diacritics (reduces false positives)
- **Improved Capital Letter Accuracy**: Deterministic case handling instead of learned patterns
- **Character Safety**: All Unicode characters > 255 safely handled with fallback mechanisms

## Architecture

### Model Components

1. **Character-level Tokenization**: Works at the character level for fine-grained control
2. **Case Normalization**: Processes text in lowercase with case pattern restoration (50% complexity reduction)
3. **Bidirectional LSTM**: Captures both forward and backward context (2-4 layers)
4. **Multi-Head Self-Attention**: Focuses on relevant parts of the input sequence (optional, recommended)
5. **Binary Classification**: 6 classifiers for Turkish diacritic pairs (c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü)

### Technical Specifications

- **Input**: Turkish text with diacritics removed (mixed case supported)
- **Output**: Turkish text with diacritics restored (preserves original case)
- **Context Window**: Configurable (default: 96 characters, expert recommended)
- **Vocabulary Size**: 256 characters (covers Turkish alphabet + common punctuation)
- **Hidden Size**: Configurable (default: 256 dimensions, expert recommended)
- **Attention Heads**: 4 (when self-attention enabled)
- **Turkish Case Handling**: Proper i/İ and ı/I mappings (differs from English)
- **CUDA Compatible**: Safe character bounds checking for GPU training

## Installation

### From PyPI (Recommended)

```bash
pip install nokta-ai
```

### From Source

```bash
# Clone the repository
git clone https://github.com/armish/nokta-ai.git
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

## Quick Start

### 1. Prepare Training Data

```bash
# Basic usage with the included Turkish deasciifier training data
python scripts/prepare_training_data.py data/aysnrgenc_turkishdeasciifier_train.txt

# Combine multiple corpus files for larger training set
python scripts/prepare_training_data.py data/corpus1.txt data/corpus2.txt data/corpus3.txt

# Use custom output location
python scripts/prepare_training_data.py data/*.txt --output data/my_cache.pkl

# Adjust train/validation split (90% train, 10% validation)
python scripts/prepare_training_data.py data/*.txt --train-ratio 0.9
```

This creates `data/combined_cache.pkl` containing prepared training and validation data.

### 2. Train a Model

#### Expert-Recommended Configuration (With Self-Attention)
```bash
# Best accuracy, ~15% better diacritic restoration
nokta-train --data-cache data/combined_cache.pkl \
            --output models/my_model.pth \
            --context-size 96 \
            --hidden-size 256 \
            --num-lstm-layers 2 \
            --use-attention \
            --epochs 50
```

#### Lightweight Configuration (Without Self-Attention)
```bash
# Smaller, faster model for resource-constrained environments
nokta-train --data-cache data/combined_cache.pkl \
            --output models/my_model_light.pth \
            --context-size 96 \
            --hidden-size 256 \
            --no-attention \
            --epochs 50
```

### 3. Evaluate Your Model

```bash
# Basic evaluation
nokta evaluate --model models/my_model.pth \
               --test-file data/test_datasets/vikipedi_test.txt

# Multi-pass restoration for words with multiple diacritics
nokta evaluate --model models/my_model.pth \
               --test-file data/test_datasets/vikipedi_test.txt \
               --num-passes 3
```

### 4. Use Your Model

```bash
# Interactive mode
nokta-restore --model models/my_model.pth

# Direct text input
nokta-restore --model models/my_model.pth --text "Bugun hava cok guzel"

# Process a file
nokta-restore --model models/my_model.pth --input input.txt --output output.txt
```

## Training Parameters

### Core Architecture Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--context-size` | 96 | Context window size (expert recommended: 96) |
| `--hidden-size` | 256 | LSTM hidden dimensions (expert recommended: 256) |
| `--num-lstm-layers` | 2 | Number of LSTM layers (expert recommends: 2-4) |
| `--use-attention` | True | Enable self-attention (recommended for accuracy) |
| `--no-attention` | False | Disable self-attention (smaller, faster models) |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--learning-rate` | 0.0003 | Learning rate (expert recommended: 3e-4) |
| `--max-train-texts` | 15000 | Maximum training texts to use |
| `--max-val-texts` | 1000 | Maximum validation texts to use |

### Data Sampling Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--balanced-sampling` | False | Equal representation of each character type |
| `--samples-per-char` | Auto | Target samples per character (auto-detected if not specified) |

### Platform-Specific Configurations

| Parameter | Description |
|-----------|-------------|
| `--cuda-config` | CUDA-optimized: batch=128, context=128, hidden=512, attention=on |
| `--mps-config` | MPS-optimized: batch=64, context=96, hidden=256, attention=on |
| `--cpu-config` | CPU-optimized: batch=4, context=20, hidden=128, attention=off |

### Example Training Commands

```bash
# Small model for quick experiments
nokta-train --data-cache data/combined_cache.pkl \
            --output models/small.pth \
            --context-size 20 \
            --hidden-size 128 \
            --epochs 10 \
            --batch-size 32

# Balanced training with equal character representation
nokta-train --data-cache data/combined_cache.pkl \
            --output models/balanced.pth \
            --balanced-sampling \
            --samples-per-char 1000 \
            --epochs 30

# High-accuracy configuration
nokta-train --data-cache data/combined_cache.pkl \
            --output models/high_accuracy.pth \
            --context-size 96 \
            --hidden-size 256 \
            --num-lstm-layers 3 \
            --use-attention \
            --epochs 100 \
            --batch-size 8

# CUDA GPU optimized (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/cuda_optimized.pth \
            --cuda-config \
            --epochs 50

# MPS optimized (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/mps_optimized.pth \
            --mps-config \
            --epochs 50

# CPU optimized (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/cpu_optimized.pth \
            --cpu-config \
            --epochs 20
```

## Evaluation Parameters

### Evaluation Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Path to trained model file |
| `--test-file` | Required | Path to test file with ground truth |
| `--context-size` | Auto | Override context size (use model default if not specified) |
| `--num-passes` | 1 | Number of restoration passes (improves multi-diacritic words) |
| `--output` | None | Save detailed results to file |

### Example Evaluation Commands

```bash
# Basic evaluation
nokta evaluate --model models/my_model.pth \
               --test-file data/test_datasets/vikipedi_test.txt

# Multi-pass evaluation with detailed output
nokta evaluate --model models/my_model.pth \
               --test-file data/test_datasets/vikipedi_test.txt \
               --num-passes 3 \
               --output evaluation_results.txt

# Override context size if needed
nokta evaluate --model models/my_model.pth \
               --test-file data/test_datasets/vikipedi_test.txt \
               --context-size 50
```

The evaluation provides:
- **Character-level accuracy**: Precise diacritic restoration
- **Word-level accuracy**: Complete word correctness
- **Diacritic-specific accuracy**: How well diacritics are restored
- **Multi-pass restoration**: With convergence detection
- **Per-sentence breakdown**: Detailed analysis for debugging

## Model Architecture Comparison

### With Self-Attention (Recommended)

**Benefits:**
- **~15% better diacritic accuracy**
- Better long-range context understanding
- Improved Turkish vowel harmony detection
- Expert validated architecture
- Model size: ~14MB

**Use for:** Production deployments, maximum accuracy requirements

### Without Self-Attention (Lightweight)

**Benefits:**
- **~20% smaller model size**
- Faster training and inference
- Lower memory requirements
- Still effective for most text
- Model size: ~10MB

**Use for:** Resource-constrained environments, mobile deployment

### Performance Comparison

| Metric | Without Attention | With Attention | Improvement |
|--------|------------------|----------------|-------------|
| **Diacritic accuracy** | ~6% | ~21% | **+15%** |
| **Character accuracy** | ~92% | ~93% | **+1%** |
| **Model size** | 10.2MB | 14.2MB | +4MB |

## Python API

```python
import nokta_ai

# Load trained model
restorer = nokta_ai.DiacriticsRestorer(model_path='models/my_model.pth')

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

### Included Test Files

- **`data/test_datasets/vikipedi_test.txt`**: 290 lines of high-quality Turkish text from Wikipedia
- **Content**: Turkish constitutional history with proper diacritics
- **Purpose**: Benchmark model performance on real-world text

### Adding Custom Test Files

1. **Place files in**: `data/test_datasets/`
2. **Format**: One sentence per line with correct diacritics
3. **Encoding**: UTF-8 text files
4. **Example structure**:
   ```
   Türkiye'nin başkenti Ankara'dır.
   Öğrenciler sınıfta ders çalışıyor.
   Çocuklar bahçede futbol oynuyorlar.
   ```

## Advanced Usage

### Context Size Experiments

Different context sizes offer different trade-offs:

#### Small Context (Fast Training)
```bash
# 20-character context - sees immediate neighbors only
nokta-train --data-cache data/combined_cache.pkl \
            --output models/small_ctx.pth \
            --context-size 20 \
            --hidden-size 128 \
            --epochs 15 \
            --batch-size 32
```
- **Good for**: Quick experiments, limited resources
- **Limitations**: May miss word-level patterns

#### Medium Context (Balanced)
```bash
# 50-character context - sees 1-2 full words
nokta-train --data-cache data/combined_cache.pkl \
            --output models/medium_ctx.pth \
            --context-size 50 \
            --hidden-size 256 \
            --epochs 20 \
            --batch-size 16
```
- **Good for**: General purpose, balanced accuracy/speed
- **Best for**: Most use cases

#### Large Context (Expert Recommended)
```bash
# 96-character context - sees full sentences
nokta-train --data-cache data/combined_cache.pkl \
            --output models/large_ctx.pth \
            --context-size 96 \
            --hidden-size 256 \
            --num-lstm-layers 2 \
            --use-attention \
            --epochs 25 \
            --batch-size 8
```
- **Good for**: Maximum accuracy, Turkish grammar patterns
- **Expert validated**: Optimal balance of accuracy and efficiency

### Multi-Pass Restoration

For words with multiple diacritics like "Üçüncü" → "Ucuncu":

```bash
# Single pass (default)
nokta evaluate --model models/my_model.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --num-passes 1

# Multi-pass with automatic convergence detection
nokta evaluate --model models/my_model.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --num-passes 3
```

The model automatically stops when output converges between passes.

## Performance

### Hardware Acceleration

The system automatically detects and uses optimal hardware acceleration with intelligent warnings and recommendations:

#### **1. NVIDIA CUDA (Recommended for Large-Scale Training)**
- **Professional GPUs**: Tesla, V100, A100 - excellent for large models and batch sizes
- **Consumer GPUs**: RTX/GTX - good performance for most training tasks
- **Automatic optimization tips**: Batch size and model size recommendations based on GPU memory
- **Memory detection**: Automatically suggests configurations for high-memory GPUs

```bash
# CUDA-optimized training (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/cuda_model.pth \
            --cuda-config \
            --epochs 50

# Manual CUDA optimization for high-memory GPUs
nokta-train --data-cache data/combined_cache.pkl \
            --output models/cuda_manual.pth \
            --batch-size 128 \
            --context-size 128 \
            --hidden-size 512 \
            --epochs 50
```

#### **2. Apple Silicon MPS (M1/M2/M3)**
- **Excellent performance**: Optimized for Apple Silicon processors
- **Energy efficient**: Lower power consumption than NVIDIA GPUs
- **Automatic detection**: Works seamlessly on Mac systems
- **Memory efficient**: Good balance of performance and resource usage

```bash
# MPS-optimized training (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/mps_model.pth \
            --mps-config \
            --epochs 50
```

#### **3. CPU Fallback (With Warnings)**
- **Performance warnings**: Alerts about slow training speed
- **Optimization suggestions**: Smaller models and configuration tips
- **CPU-optimized presets**: Automatic small model configurations

```bash
# CPU-optimized training (automatic configuration)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/cpu_model.pth \
            --cpu-config \
            --epochs 30
```

### Benchmarks

#### **NVIDIA CUDA GPUs**
- **High-memory GPUs** (A100, V100, Tesla): ~5,000-8,000 samples/second training, ~50,000+ characters/second inference
- **Consumer GPUs** (RTX/GTX): ~2,000-4,000 samples/second training, ~20,000-30,000 characters/second inference
- Recommended: Use `--cuda-config` for optimal settings
- Can handle large models with sufficient GPU memory

#### **Apple M1/M2/M3 with MPS**
- Training speed: ~1,000-2,000 samples/second
- Inference speed: ~10,000-20,000 characters/second
- Memory efficient: Good for models up to 512 hidden size
- Energy efficient: Lower power consumption
- Recommended: Use `--mps-config` for optimal settings

#### **CPU Fallback**
- Training speed: ~50-100 samples/second ⚠️
- Inference speed: ~1,000 characters/second ⚠️
- Use only for small models and testing

### Expected Accuracy

With proper training (50+ epochs on quality Turkish data):
- Character-level accuracy: 95%+
- Word-level accuracy: 90%+
- Diacritic-specific accuracy: 85%+

## Training Data Requirements

### Data Sources

The system can be trained on any Turkish text corpus. Best results are achieved with:

1. **Wikipedia dumps**: Comprehensive vocabulary and proper spelling
2. **News articles**: Contemporary language usage
3. **Books and literature**: Diverse writing styles
4. **Web crawls**: Informal language patterns

### Included Training Data

- **`data/aysnrgenc_turkishdeasciifier_train.txt`**: High-quality Turkish text with proper diacritics
- Ready to use for training without additional data collection

### Adding Your Own Data

1. **Format**: Plain text files (UTF-8) with proper Turkish diacritics
2. **Structure**: One or more paragraphs per file
3. **Quality**: Ensure text has correct spelling and diacritics
4. **Combine**: Use the data preparation script to merge multiple files:

```bash
# Combine your data with the included training data
python scripts/prepare_training_data.py \
    data/aysnrgenc_turkishdeasciifier_train.txt \
    your_data/*.txt \
    --output data/custom_cache.pkl
```

### Data Requirements

- **Minimum**: 1MB of Turkish text (included data meets this)
- **Recommended**: 10MB+ for good coverage
- **Optimal**: 100MB+ for production use

## Troubleshooting

### Common Issues

1. **Low accuracy after training**
   - Solution: Train for more epochs (50+)
   - Use more training data
   - Enable self-attention (`--use-attention`)
   - Increase model capacity (`--hidden-size 256`)

2. **MPS not detected on Mac**
   - Ensure PyTorch 2.0+ is installed
   - Check MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

3. **Out of memory errors**
   - Reduce `--batch-size`
   - Reduce `--context-size`
   - Disable attention (`--no-attention`)
   - Use smaller `--hidden-size`

4. **Model predictions seem random**
   - Check if you're using the correct `--context-size` during evaluation
   - Ensure model was trained for enough epochs
   - Verify training data quality

## Implementation Details

### Character Mapping

The system handles these Turkish-specific transformations:
- ç ↔ c, Ç ↔ C
- ğ ↔ g, Ğ ↔ G
- ı ↔ i, İ ↔ I
- ö ↔ o, Ö ↔ O
- ş ↔ s, Ş ↔ S
- ü ↔ u, Ü ↔ U

### Multi-Head Self-Attention

When enabled, the attention mechanism helps the model:
- Focus on relevant context for ambiguous cases
- Learn long-range dependencies
- Handle Turkish vowel harmony rules
- Improve accuracy on compound words

### Balanced Sampling

The `--balanced-sampling` option ensures equal representation:
- Groups training samples by diacritic character type
- Balances representation across all 6 Turkish character pairs
- Prevents model bias toward common characters like 'i/ı'
- Use `--samples-per-char` to control target samples per character

## Package Structure

```
nokta-ai/
├── nokta_ai/
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Main restoration classes
│   ├── models/               # Neural network architectures
│   │   └── constrained.py    # Constrained diacritics model
│   └── cli/                  # Command line interfaces
│       ├── main.py           # Main CLI (nokta command)
│       ├── train_constrained.py  # Training script
│       └── evaluate_constrained.py  # Evaluation script
├── data/
│   ├── aysnrgenc_turkishdeasciifier_train.txt  # High-quality Turkish training data  # Additional training data
│   ├── combined_cache.pkl    # Preprocessed training data (generated)
│   └── test_datasets/        # Test datasets
│       └── vikipedi_test.txt # Wikipedia test data
├── models/                   # Trained model weights (generated)
├── scripts/
│   └── prepare_training_data.py  # Data preparation script
├── ARCHITECTURE.md           # Detailed neural network documentation
└── README.md                # This file
```

## CLI Command Reference

### Training: `nokta-train`
```bash
nokta-train --data-cache DATA --output MODEL [options]
```

### Evaluation: `nokta evaluate` / `nokta-evaluate`
```bash
nokta evaluate --model MODEL --test-file FILE [options]
```

### Restoration: `nokta restore`
```bash
nokta restore --model MODEL [--text TEXT | --input FILE] [options]
```

### Benchmarking: `nokta benchmark`
```bash
nokta benchmark --model MODEL [options]
```

## License

MIT License

## Citation

If you use this work in your research, please cite:

```bibtex
@software{turkish_diacritics_nn,
  title = {Turkish Diacritics Restoration with Neural Networks},
  year = {2024},
  url = {https://github.com/armish/nokta-ai}
}
```

## Contact

For questions and contributions, please open an issue on GitHub.