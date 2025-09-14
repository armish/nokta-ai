# IDEAL.md - Expert Architecture Recommendations

This document contains expert recommendations for optimal Turkish diacritics restoration architecture, received from a domain expert. It serves as our reference for future improvements.

## Current State vs Expert Recommendation

**Our Current Approach (✅ = Validated by Expert):**
- Constrained model focusing only on the 6 diacritic pairs ✅ (Expert agrees)
- Context-based character classification ✅ (Matches expert's "sequence labeling")
- BiLSTM architecture ✅ (Expert recommends this)
- 128 hidden size, 100 context → ~547K params (~2.1MB) ✅ (Within expert's target)

**What We're Missing (Big Opportunities):**
1. **Self-attention layer** - Expert says this adds 0.2-0.4% accuracy
2. **Better training strategy** - Our current approach vs expert's weighted sampling
3. **Sliding window with overlap** - We process fixed windows, expert recommends overlapping
4. **Confidence thresholding** - Keep ambiguous predictions as-is

## Expert's Problem Framing

> "This is a perfect use‑case for a *small, fast, local* model. Two choices tend to work best for Turkish diacritics restoration:
> * **(A) Per‑character sequence labeling (recommended)**
> * **(B) Very small encoder‑only Transformer (good alternate)**"

The expert confirms our constrained approach is optimal because:
- Input and output are the same length
- Task is "choose the correct variant for a character," not "rewrite the sequence"
- 1:1 length mapping with only 6 character pairs to handle

## Recommended Configurations

### Option A: Minimal Changes (Quick Win)
Keep our current architecture but add these improvements:
```yaml
model:
  context_size: 96          # Down from 100 (expert's sweet spot)
  hidden_size: 256          # Up from 128 (expert's recommendation)
  # Add: single self-attention layer

training:
  max_train_texts: 15000    # Up from 10000
  learning_rate: 3e-4       # Down from 1e-3 (expert's rec)
  # Add: weighted sampling (70% ambiguous, 30% any)
  # Add: confidence thresholding at inference
```

**Impact:** ~3-4M params (~13-16MB), likely 1-2% accuracy boost

### Option B: Expert's Full Recipe (Bigger Change)
Implement the expert's exact BiLSTM + MHSA architecture:
```yaml
model:
  embedding_dim: 128
  lstm_hidden: 256          # per direction
  lstm_layers: 2
  lstm_dropout: 0.25
  mhsa_layers: 1           # NEW: Add self-attention
  mhsa_heads: 4
  head: confusion_set       # Our current approach

context:
  window: 96
  stride: 64               # NEW: Overlapping windows

training:
  batch_size: 256          # Up from 32
  lr: 3e-4
  epochs: 50               # Up from 20
```

**Impact:** ~3.5M params (~14MB), likely 2-4% accuracy boost

## Expert's Full Architecture Specification

### Recommended Starting Architecture (A): BiLSTM + Light Self-Attention

**Why this works**: BiLSTMs give excellent local and mid-range context for agglutinative cues; single lightweight self-attention layer helps on long compounds or when vowel harmony clues are far away.

**Spec (balanced for size / speed / accuracy):**

* **Vocab**: 256 (byte/char level)
* **Embedding dim**: 128
* **Encoder**: 2× **BiLSTM**, hidden size **256** per direction (so 512 concat), **dropout 0.25** between layers
* **Refinement**: **1× Multi‑Head Self‑Attention** layer on the BiLSTM outputs
  * d_model = 512 (matches BiLSTM output), **heads = 4**, FFN dim = 1024, dropout 0.1
* **Head**: "Confusion‑set" head (our current approach) - one tiny binary classifier per ambiguous base letter

**Parameter size**: ≈ **3.3–4.0M params (~13–16 MB)** — well within 10–50 MB target

### Alternate Architecture (B): Tiny Encoder-Only Transformer

* **d_model**: 256
* **Heads**: 4
* **Layers**: 4 (encoder blocks)
* **FFN dim**: 512 or 768
* **Dropout**: 0.1

Params ~5–7M (≈20–28 MB FP32).

## Context Window Recommendations

Turkish disambiguation needs **intra‑word** cues plus neighboring tokens:
- Vowel harmony and suffix morphology captured within **10–20 chars**
- Long compounds benefit from seeing **both sides** of word boundary

**Practical starter**:
- **Window** `W = 96` characters
- **Stride** `S = 64` (→ 32‑char overlap)
- **Merging**: triangular weights across overlaps

Performance curve:
| Window | Overlap | Gain vs. 64 |
|--------|---------|-------------|
| 64     | 24–32   | baseline    |
| 96     | 32      | **+0.2–0.5%** char acc |
| 128    | 32–48   | **≤ +0.1–0.2%** more |

## Training Recipe for Quick Convergence

### Sampling Strategy
- Train on **random windows** of length `W` cut from corpora
- **Oversample windows** that contain ≥1 ambiguous letter
- **70/30 mix**: 70% "contains ambiguity," 30% "any window"

### Loss & Targets
For confusion-set heads (our approach):
- **Binary cross-entropy** only on relevant head for each position
- No class imbalance issues

### Optimizer & Schedule
- **AdamW**, lr **3e‑4**, weight_decay **1e‑4**
- **Batch size**: as large as VRAM allows (e.g., 256 windows)
- **Grad clip**: 1.0
- **Epochs**: **40–60** (can hit 95%+ char acc with Wikipedia‑scale data)

### Regularization
- Dropout as specified above
- **MixCasing augmentation**: randomly lowercase/uppercase for I/İ/i/ı robustness
- **Punctuation jitter**: randomly insert/remove apostrophes/commas

## Inference Improvements

1. **Confidence‑aware edits**
   - If max‑prob among allowed options < **0.6–0.7**, **keep input character**
   - Reduces "creative" mistakes

2. **Hard constraints**
   - **ğ/Ğ almost never starts a word** → forbid conversion at word start
   - Keep **ASCII digits, spaces, punctuation** unchanged

3. **Overlap merging**
   - Use triangular weights and normalize when combining windows

## Expected Performance

With expert's recommendations:
- **95%+ character accuracy**
- **~90% word accuracy**
- **<20 MB model size**
- Very fast on MPS/CUDA

## Expert's Plug-and-Play Config

```yaml
# config.yaml
model:
  type: bilstm_mhsa_tagger
  vocab_size: 256
  embedding_dim: 128
  lstm_hidden: 256        # per direction
  lstm_layers: 2
  lstm_dropout: 0.25
  mhsa_layers: 1
  mhsa_heads: 4
  mhsa_ffn_dim: 1024
  mhsa_dropout: 0.10
  head: confusion_set

context:
  window: 96
  stride: 64               # 32-char overlap
  merge: triangular

training:
  epochs: 50
  batch_size: 256
  lr: 3e-4
  weight_decay: 1e-4
  grad_clip: 1.0
  amp: true
  label_smoothing: 0.05
  loss_weight_ambiguous: 5.0
  oversample_ambiguous: 0.70

inference:
  min_confidence: 0.65
  forbid_gh_start: true
  keep_nonletters: true
```

## Turkish-Specific Optimizations

- **Uppercase I/İ**: Include ALL‑CAPS lines during training
- **Apostrophes in proper nouns**: Don't strip them ("Türkiye'nin", "Ankara'da")
- **ğ frequency**: Very high mid‑word, nearly zero at word start

## Our Recommended Next Steps

### Immediate (Option A - Minimal Changes):
1. **Increase hidden_size to 256** (easy change, big impact)
2. **Lower learning rate to 3e-4** (one line change)
3. **Add confidence thresholding** at inference (keep predictions below 0.65 unchanged)
4. **Train for 50 epochs** instead of 20

### Medium-term (Full Expert Recipe):
1. **Add self-attention layer** after BiLSTM
2. **Implement sliding windows with overlap**
3. **Weighted sampling strategy** (70% ambiguous sentences)
4. **Triangular weight merging** for overlaps

## Validation Metrics

Report at least these:
- **Character accuracy (overall)**
- **Character accuracy (ambiguous positions only)** ← most informative
- **Word accuracy**
- **Sentence accuracy** (optional)

## Expert's Bottom Line

> "Start with **BiLSTM(2×256) + 1×MHSA(4‑head)** and **W=96, stride=64**.
> Use a **confusion‑set head**, **confidence‑aware edits**, and **triangular overlap merge**.
> Expect **95%+ char accuracy** and **~90% word accuracy** on Wikipedia‑style text after ~50 epochs with <20 MB model size."

---

**Key Takeaway**: Our current constrained approach is exactly right! The expert validates our architecture choice and provides a clear roadmap to 95%+ accuracy with minimal changes.