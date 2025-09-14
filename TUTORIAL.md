# nokta-ai Tutorial: Constrained Diacritics Restoration

This tutorial shows how to use the constrained Turkish diacritics restoration model, which focuses specifically on the 6 Turkish character pairs and maintains input/output length consistency.

## Quick Start

### 1. Prepare Your Data
```bash
# Create training dataset from corpus files
python scripts/prepare_training_data.py
```

### 2. Train Constrained Model
```bash
# Basic training with default settings (context=100, hidden=128)
nokta-train --data-cache data/combined_cache.pkl \
            --output models/my_model.pth \
            --epochs 20
```

### 3. Test Your Model
```bash
# Quick interactive test
nokta-test --model models/my_model.pth

# Comprehensive evaluation
nokta evaluate --model models/my_model.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained
```

## Advanced Usage

### Context Size Experiments

The context size determines how many characters around each target character the model can see. Different sizes have different trade-offs.

**Important**: When evaluating or using a model, specify the same context size used during training to ensure dimension matching:

#### Small Context (Fast Training)
```bash
# 7-character context - sees immediate neighbors only
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/small_ctx.pth \
                       --context-size 7 \
                       --epochs 15 \
                       --batch-size 32

# Test with matching context size
nokta-test --model models/small_ctx.pth --context-size 7

# Evaluate with matching context size
nokta evaluate --model models/small_ctx.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 7

# Good for: Quick experiments, limited resources
# Limitations: May miss word-level patterns
```

#### Medium Context (Balanced)
```bash
# 50-character context - sees 1-2 full words
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/medium_ctx.pth \
                       --context-size 50 \
                       --epochs 20 \
                       --batch-size 16

# Test with matching context size
nokta-test --model models/medium_ctx.pth --context-size 50

# Evaluate with matching context size
nokta evaluate --model models/medium_ctx.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 50

# Good for: General purpose, balanced accuracy/speed
# Best for: Most use cases
```

#### Large Context (High Accuracy)
```bash
# 100-character context - sees full sentences
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/large_ctx.pth \
                       --context-size 100 \
                       --epochs 25 \
                       --batch-size 8

# Test with matching context size
nokta-test --model models/large_ctx.pth --context-size 100

# Evaluate with matching context size
nokta evaluate --model models/large_ctx.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 100

# Good for: Maximum accuracy, Turkish grammar patterns
# Requires: More memory and training time
```

#### Extra Large Context (Maximum Context)
```bash
# 200-character context - sees paragraph-level patterns
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/xl_ctx.pth \
                       --context-size 200 \
                       --epochs 30 \
                       --batch-size 4 \
                       --hidden-size 256

# Test with matching context size
nokta-test --model models/xl_ctx.pth --context-size 200

# Evaluate with matching context size
nokta evaluate --model models/xl_ctx.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 200

# Good for: Research, maximum possible accuracy
# Requires: Significant computational resources
```

### Model Architecture Tuning

#### Larger Hidden Layers
```bash
# Increase model capacity for better accuracy
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/large_model.pth \
                       --context-size 100 \
                       --hidden-size 256 \
                       --epochs 25
```

#### Learning Rate Optimization
```bash
# Slower learning for better convergence
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/slow_learning.pth \
                       --learning-rate 0.0005 \
                       --epochs 40

# Faster learning for quick experiments
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/fast_learning.pth \
                       --learning-rate 0.002 \
                       --epochs 15
```

## Performance Comparison

### Comparing Different Models

Train multiple models and compare their performance:

```bash
# Train different context sizes
nokta-train --data-cache data/combined_cache.pkl \
                       --output models/ctx_7.pth --context-size 7 --epochs 20

nokta-train --data-cache data/combined_cache.pkl \
                       --output models/ctx_50.pth --context-size 50 --epochs 20

nokta-train --data-cache data/combined_cache.pkl \
                       --output models/ctx_100.pth --context-size 100 --epochs 20

# Evaluate all models with their respective context sizes
echo "=== Context Size 7 ==="
nokta evaluate --model models/ctx_7.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 7

echo "=== Context Size 50 ==="
nokta evaluate --model models/ctx_50.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 50

echo "=== Context Size 100 ==="
nokta evaluate --model models/ctx_100.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 100
```

## Understanding the Output

### Evaluation Metrics

The constrained model provides three key metrics:

1. **Character Accuracy**: Overall character-by-character matching
2. **Word Accuracy**: Percentage of completely correct words
3. **Diacritic-Specific Accuracy**: How well diacritics are restored (most important!)

Example output:
```
Overall character accuracy: 94.50%
Overall word accuracy: 87.23%
Diacritic-specific accuracy: 91.15%
```

### What the Model Does

The constrained model:
- ✅ **Only modifies**: c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü
- ✅ **Preserves length**: Input and output are always the same length
- ✅ **Keeps other characters**: Punctuation, numbers, other letters unchanged
- ✅ **Context-aware**: Uses surrounding text to make decisions

### What It Doesn't Do

- ❌ **No text generation**: Cannot add or remove characters
- ❌ **No other languages**: Specifically designed for Turkish
- ❌ **No other diacritics**: Only handles the 6 main Turkish pairs

## Production Usage

### Python API
```python
from nokta_ai.models.constrained import ConstrainedDiacriticsRestorer

# Load trained model (automatically uses context size from checkpoint)
restorer = ConstrainedDiacriticsRestorer('models/my_model.pth')

# Or explicitly specify context size to match training
restorer = ConstrainedDiacriticsRestorer('models/my_model.pth', context_size=100)

# Restore diacritics
text = "Bugun hava cok guzel"
result = restorer.restore_diacritics(text)
print(result)  # "Bugün hava çok güzel"
```

### Command Line Usage
```bash
# Restore text with automatic context size detection
nokta restore --model models/my_model.pth \
    --text "Bugun hava cok guzel" \
    --constrained

# Specify context size explicitly (must match training)
nokta restore --model models/my_model.pth \
    --text "Bugun hava cok guzel" \
    --constrained --context-size 100

# Process a file
nokta restore --model models/my_model.pth \
    --input input.txt --output output.txt \
    --constrained --context-size 100
```

### Batch Processing
```python
texts = [
    "Turkiye'nin baskenti Ankara'dir",
    "Ogrenciler sinifta ders calisiyor",
    "Cocuklar bahcede oynuyorlar"
]

for text in texts:
    restored = restorer.restore_diacritics(text)
    print(f"{text} -> {restored}")
```

## Tips for Best Results

### 1. Training Data Quality
- Use clean, well-written Turkish text
- Ensure proper diacritics in training data
- Include diverse text types (news, literature, informal)

### 2. Context Size Selection
- **Small datasets**: Use smaller context (7-20) to avoid overfitting
- **Large datasets**: Use larger context (100-200) for better accuracy
- **Memory limited**: Start with context=50 as a good balance

### 3. Training Duration
- **Quick test**: 10-15 epochs
- **Production model**: 25-40 epochs
- **Research quality**: 50+ epochs

### 4. Hyperparameter Guidelines
- **Hidden size**: 128 (default) → 256 (better) → 512 (research)
- **Batch size**: Reduce if memory limited, increase if training slow
- **Learning rate**: 0.001 (safe) → 0.0005 (stable) → 0.002 (fast)

### 5. Evaluation Best Practices
- Focus on **diacritic-specific accuracy** as the main metric
- Test on diverse text types
- Compare multiple context sizes
- Save evaluation results for comparison
- **Always specify context size** when evaluating to match training dimensions

## Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Reduce batch size and context size
nokta-train-constrained --context-size 50 --batch-size 4 ...
```

#### Poor Accuracy
```bash
# Increase epochs and context size
nokta-train-constrained --context-size 100 --epochs 40 ...
```

#### Training Too Slow
```bash
# Use smaller context and higher learning rate
nokta-train-constrained --context-size 25 --learning-rate 0.002 ...
```

#### Context Size Mismatch Error
```bash
# If you get dimension errors during evaluation, specify the correct context size
# Check what context size was used during training and match it:
nokta evaluate --model models/my_model.pth \
    --test-file data/test_datasets/vikipedi_test.txt \
    --constrained --context-size 7  # Match the training context size
```

### Getting Help

1. **Check model output**: Use `nokta-test` for quick tests
2. **Verify data**: Ensure training data has proper Turkish diacritics
3. **Monitor metrics**: Watch diacritic-specific accuracy during evaluation
4. **Compare baselines**: Test different context sizes to find optimal setting

## Advanced Scenarios

### Custom Training Data
```python
# Prepare your own training data
from nokta_ai.models.constrained import create_constrained_training_data

texts = ["Your Turkish texts here..."]
training_samples = create_constrained_training_data(texts, context_size=100)
```

### Model Ensembling
```bash
# Train multiple models with different settings
# Use voting or averaging for final predictions
```

### Fine-tuning
```bash
# Start with a pre-trained model and continue training
# (Load existing model and train for more epochs)
```

This tutorial should get you started with the constrained diacritics restoration model. Experiment with different context sizes to find what works best for your use case!