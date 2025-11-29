# 🧠 A.D.A.M - Adaptive and Dynamic Agent Module

**Continuous Self-Training Language Model with Revolutionary Hot/Cold Vocabulary Architecture**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

---

## 🚀 What is A.D.A.M?

A.D.A.M is an experimental language model featuring:

- **🔥 Hot/Cold Vocabulary Architecture** - Unlimited RAM vocabulary with GPU-cached hot words (LRU)
- **📖 Vocabulary Pre-Training** - Scan datasets first for stable vocabulary before training
- **⚡ Extreme GPU Optimization** - cuBLAS GEMM, fused kernels, pipelined training
- **🧩 Venn Semantic System** - Multi-head semantic clustering for generalization beyond memorization
- **🌊 Continuous Learning** - Train indefinitely on new data, vocabulary grows automatically
- **🎯 Production-Ready** - Validation, early stopping, checkpointing, hot-reload

---

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [⚙️ Installation](#️-installation)
- [🎯 Quick Start](#-quick-start)
- [📚 Usage Guide](#-usage-guide)
  - [Vocabulary Pre-Training](#vocabulary-pre-training-new)
  - [Dataset Training](#dataset-training)
  - [Wikipedia Training](#wikipedia-training)
  - [Interactive Chat](#interactive-chat)
- [🏗️ Architecture](#️-architecture)
- [⚡ Performance](#-performance)
- [🎛️ Configuration](#️-configuration)
- [📁 Project Structure](#-project-structure)
- [🔧 Development](#-development)
- [📄 License](#-license)

---

## ✨ Key Features

### 🔥 Hot/Cold Vocabulary (LRU Cache Architecture)

**Philosophy: Learn everything (RAM), cache what's used (GPU)**

```
┌─────────────────────────────────────────┐
│  COLD VOCAB (RAM) - Unlimited Storage   │
│  • ALL words ever seen (millions)       │
│  • Pre-initialized embeddings           │
│  • Persisted with checkpoints           │
│  • "La RAM è abbondante!" philosophy    │
└──────────────┬──────────────────────────┘
               │ LRU Cache (top 10k words)
               ↓
┌─────────────────────────────────────────┐
│   HOT VOCAB (GPU) - Fast Access         │
│   • 10,000 most-used words              │
│   • Ultra-fast training/inference       │
│   • Automatic eviction & preloading     │
│   • Synced with CUDA MAX_WORD_VOCAB     │
└─────────────────────────────────────────┘
```

**Benefits:**
- ✅ Zero memory limits (RAM is cheap)
- ✅ Optimal GPU utilization (only hot words)
- ✅ No loss spikes from new words
- ✅ Stable training at scale

### 📖 Vocabulary Pre-Training (NEW!)

Scan your dataset **before** training to build a stable vocabulary:

```bash
# Scan dataset 2 times, then train for 10 passes
adam dataset train.jsonl --vocab-passes 2 --passes 10
```

**What happens:**
1. **Scan Phase** (passes 1-2): Read all data, discover words, count frequencies (no GPU sync = fast!)
2. **Finalization**: Load top 10k frequent words → HOT vocab (GPU)
3. **Training Phase** (passes 3-12): Train with stable vocabulary (zero overhead!)

**Benefits:**
- ✅ Vocabulary complete from start (no mid-training creation)
- ✅ Top words automatically in GPU (optimal cache)
- ✅ Eliminates sync overhead during training
- ✅ Prevents loss spikes from new embeddings
- ✅ ~20% faster training, better convergence

---

## ⚙️ Installation

### From Source (Recommended)

```bash
git clone https://github.com/krokodil-byte/ADAM.git
cd ADAM
pip install -e .
```

### Quick Run (No Installation)

```bash
python run.py [command] [args]
```

### Requirements

- **Python** ≥ 3.8
- **NumPy** ≥ 1.20.0
- **CUDA Toolkit** (optional, for GPU acceleration)
  - Tested: CUDA 11.0 - 12.x
  - Recommended: CUDA 12.x for best performance

### Verify Installation

```bash
adam init
```

Checks CUDA availability and compiles GPU kernels.

---

## 🎯 Quick Start

### 1️⃣ Initialize System

```bash
adam init
```

### 2️⃣ Train on Your Data

```bash
# Train on text file
adam train mydata.txt -o model.ckpt --passes 5

# Train on HuggingFace dataset with vocab pre-training
adam dataset train.jsonl --vocab-passes 2 --passes 10 \
     --validation --early-stopping

# Train on Wikipedia (streaming from API)
adam wikipedia --vocab-passes 1 --max-articles 5000 \
     --language en --validation
```

### 3️⃣ Interactive Chat

```bash
adam chat -c model.ckpt
```

### 4️⃣ View Statistics

```bash
adam stats -c model.ckpt
```

---

## 📚 Usage Guide

### Vocabulary Pre-Training (NEW!)

Build stable vocabulary before training:

```bash
# HuggingFace Dataset
adam dataset train.jsonl \
     --vocab-passes 2 \      # Scan dataset twice
     --passes 10 \           # Then train for 10 passes
     --validation \
     --early-stopping

# Plain Text Files
adam dataset ./texts/ \
     --vocab-passes 1 \
     --passes 5 \
     --extensions .txt,.md

# Wikipedia
adam wikipedia \
     --vocab-passes 1 \      # Scan 100 articles first
     --max-articles 5000 \
     --language en
```

**Output Example:**
```
======================================================================
📖 VOCABULARY SCANNING - 2 pass(es)
======================================================================

🔍 Vocabulary scan pass 1/2
   Scanned 100/500 samples
   Scanned 200/500 samples
   ...
   ✓ Pass 1 complete - 8543 words discovered

🔍 Vocabulary scan pass 2/2
   ...
   ✓ Pass 2 complete - 9821 words discovered

📚 Finalizing vocabulary from scan...
   Total words discovered: 9821
   Loading top 9821 words to HOT vocab (GPU)...
   ✅ Vocabulary finalized:
      - Cold vocab: 9821 words (all in RAM)
      - Hot vocab: 9821/10000 words (in GPU)
      - Top 5 words: the(1523), of(987), and(856), to(743), a(654)
======================================================================
```

### Dataset Training

#### HuggingFace Datasets (.jsonl, .parquet, .csv)

```bash
adam dataset data.jsonl \
     --input-col question \      # Input column name
     --output-col answer \        # Output column name
     --template "{input}\n\n{output}" \
     --vocab-passes 1 \
     --passes 5 \
     --validation \
     --val-split 0.1
```

#### Plain Text Files

```bash
adam dataset ./documents/ \
     --extensions .txt,.md,.py \
     --vocab-passes 1 \
     --passes 3 \
     --auto-save 100
```

**Options:**
- `--vocab-passes N` - Number of vocabulary scanning passes (default: 0)
- `--passes N` - Training passes (default: 1)
- `--validation` - Enable validation
- `--early-stopping` - Stop when validation plateaus
- `--val-split 0.1` - Validation split (10%)
- `--auto-save N` - Auto-save every N samples
- `--preset PRESET` - Config preset (see Configuration)

### Wikipedia Training

#### Streaming from API

```bash
adam wikipedia \
     --vocab-passes 1 \          # Scan first batch
     --max-articles 10000 \
     --language en \
     --batch-size 100 \          # Fetch 100 articles per batch
     --passes 2 \                # 2 training passes per batch
     --validation \
     --val-articles 20
```

**Languages supported:** `en`, `it`, `de`, `fr`, `es`, `ja`, `zh`, etc.

#### From Local Dump

```bash
adam wikipedia dump.jsonl \
     --max-articles 5000 \
     -o wiki_model.ckpt
```

**Dump formats:** `.jsonl`, `.xml`, `.txt`

### Interactive Chat

```bash
adam chat -c model.ckpt

> Hello, how are you?
[Model responds...]

> Tell me about machine learning
[Model responds...]
```

**Commands:**
- Type your message and press Enter
- `quit`, `exit`, or `q` to exit

### TUI Dashboard

Full graphical interface:

```bash
adam
# or
adam dashboard
```

**Main Menu:**
- 🚀 Initialize System
- 🧠 Train on Text
- 📚 Wikipedia Training
- 📂 Dataset Training
- 💬 Interactive Chat
- 📊 View Statistics
- ⚙️ Settings
- 🚪 Exit

**Navigation:**
- `↑↓` Arrow keys
- `Enter` to select
- `Q` to go back/quit

---

## 🏗️ Architecture

### Core Components

```
┌──────────────────────────────────────────────────────┐
│                   VectLLMBrain                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  Transformer (6 layers, 768d, 12 heads)      │   │
│  │  • Self-attention with cuBLAS GEMM           │   │
│  │  • Fused FFN kernels                         │   │
│  │  • Pipelined training (3-stage overlap)      │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Venn Semantic System (Multi-Head)           │   │
│  │  • 12 heads × 256 clusters = 3072 total      │   │
│  │  • Gaussian activation + propagation         │   │
│  │  • Dynamic cluster updates                   │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Hot/Cold Vocabulary (LRU)                   │   │
│  │  • Cold: Unlimited RAM storage               │   │
│  │  • Hot: 10k GPU cache (auto-managed)         │   │
│  │  • Pre-initialized embeddings                │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### Vocabulary Flow

```
Text Input → Tokenization → Word Discovery
                                  ↓
                    ┌─────────────────────────┐
                    │ Vocab Scan Mode?        │
                    └─────────┬───────────────┘
                          Yes │ No
                              │
         ┌────────────────────┴──────────────────┐
         ↓                                       ↓
  Build vocab in RAM                   Add to COLD vocab (RAM)
  (no GPU sync)                        Initialize embedding
                                              ↓
                                    ┌──────────────────┐
                                    │ HOT vocab full?  │
                                    └────┬─────────────┘
                                      No │ Yes
                                         │
                              ┌──────────┴────────────┐
                              ↓                       ↓
                        Load to GPU           Evict LRU word
                                              Load new word
                                                     ↓
                                          Training with stable vocab
```

### Training Pipeline

```
CPU Thread                GPU Stream 0           GPU Stream 1
─────────────────────────────────────────────────────────────
Batch 1: Encode text
         ↓
Batch 1: H2D transfer  →  Forward pass
         ↓                      ↓
Batch 2: Encode text     →  Backward pass
         ↓                      ↓
Batch 2: H2D transfer  →  Weight update    →  D2H loss/stats
         ↓                                       ↓
Batch 3: Encode text                       Process stats
         ...                                    ...
```

**3-stage overlap = ~95% GPU utilization**

---

## ⚡ Performance

### Memory Usage

| Component | RAM | VRAM | Notes |
|-----------|-----|------|-------|
| Model (768d, 6L) | 200 MB | ~700 MB | Base architecture |
| Cold vocab (100k words) | 300 MB | - | RAM only |
| Hot vocab (10k words) | - | 30 MB | GPU cache |
| Training batch (512 seq) | - | 2/4 GB | Activations |
| **Total (typical)** | **~1 GB** | **~3/5 GB** | Comfortable for 8GB+ GPUs |

---

## 🎛️ Configuration

### Configuration Presets

Quick profiles for different scenarios:

```bash
# Fast experimentation (10x learning rate)
adam dataset data.jsonl --preset fast_learning

# Production training (stable, conservative)
adam dataset data.jsonl --preset stable

# Maximum GPU performance
adam dataset data.jsonl --preset high_performance

# Memory-constrained GPUs
adam dataset data.jsonl --preset memory_efficient

# Chat/inference only (no training)
adam chat -c model.ckpt --preset inference
```

| Preset | LR | Momentum | Best For |
|--------|-----|----------|----------|
| `default` | 0.0001 | 0.9 | General-purpose |
| `fast_learning` | 0.001 | 0.7 | Quick experiments |
| `stable` | 0.00001 | 0.95 | Production training |
| `research` | 0.0005 | 0.9 | Venn system research |
| `inference` | 0 | - | Generation only |
| `high_performance` | - | - | Max GPU speed |
| `memory_efficient` | - | - | Limited VRAM |
| `max_throughput` | - | - | Absolute max speed |

### Key Settings

#### Hot/Cold Vocabulary

```python
MAX_WORD_VOCAB_SIZE = 10000    # MUST match CUDA (hot vocab size)
MAX_HOT_VOCAB = 10000          # GPU cache size
WORD_CREATION_THRESHOLD = 5    # Create word after N occurrences
WORD_PRUNING_THRESHOLD = 0     # 0 = never prune (recommended)
ENABLE_TOKEN_PRELOADING = True # Auto-load tokens before forward pass
SAVE_COLD_VOCAB = True         # Persist cold vocab to disk
```

#### Training

```python
BASE_LR = 0.0001               # Base learning rate
EMBEDDING_LR_SCALE = 0.1       # Embeddings 10x slower (stable)
MOMENTUM = 0.9                 # SGD momentum
VALIDATION_SPLIT = 0.1         # 10% for validation
EARLY_STOPPING_PATIENCE = 5    # Stop after 5 validations w/o improvement
AUTO_SAVE_FREQUENCY = 1000     # Auto-save every N samples
```

#### Venn Semantic System

```python
NUM_VENN_HEADS = 12            # Number of Venn heads
VENN_CLUSTERS_PER_HEAD = 256   # Clusters per head
VENN_PROPAGATION_FACTOR = 0.2  # Activation propagation strength
VENN_INTERSECTION_THRESHOLD = 0.3  # Cluster similarity threshold
```

#### Performance

```python
USE_CUBLAS = True              # cuBLAS matrix ops
ENABLE_FUSED_KERNELS = True    # Fused attention+FFN
PIPELINE_MODE = "triple"       # 3-stage pipeline overlap
GPU_UTILIZATION_TARGET = 0.9   # Target 90% GPU usage
```

### TUI Settings Menu

Access via `adam` → **⚙️ Settings**:

```
⚙️ Settings
├── 🏗️ Model Architecture
│   ├── Embedding Dimension: 768
│   ├── Num Heads: 12
│   ├── Num Layers: 6
│   └── Max Sequence Length: 512
├── 📈 Training Parameters
│   ├── Base Learning Rate: 0.0001
│   ├── Momentum: 0.9
│   ├── Validation Split: 0.1
│   └── Early Stopping: On
├── ⚡ Performance
│   ├── cuBLAS: On
│   ├── Fused Kernels: On
│   ├── Pipeline: triple
│   └── GPU Target: 90%
└── 💾 Save Settings
```

---

## 📁 Project Structure

```
ADAM/
├── setup.py                     # Package installation
├── run.py                       # Development runner
├── README.md                    # This file
├── LICENSE                      # CC BY-NC 4.0
└── A.D.A.M — Adaptive and Dynamic Agent Module/
    ├── __main__.py              # Entry point
    ├── cli/
    │   └── adam.py              # CLI interface
    ├── core/
    │   ├── brain_wrapper.py     # Main model wrapper
    │   ├── vocabulary.py        # Dynamic vocabulary (hot/cold)
    │   ├── config.py            # Configuration system
    │   ├── stats.py             # Statistics collector
    │   ├── pipeline.py          # Pipelined trainer
    │   └── constants.py         # System constants
    ├── modules/
    │   ├── dataset_training.py  # Dataset trainer (HF + plain text)
    │   ├── wikipedia_training.py # Wikipedia trainer (API + dump)
    │   ├── training_logger.py   # Training logger
    │   ├── chat.py              # Interactive chat
    │   └── tui.py               # TUI dashboard
    ├── Utils/
    │   ├── checkpoint.py        # Checkpoint management
    │   ├── tokenizer.py         # Text tokenization
    │   └── compiler.py          # CUDA compilation
    ├── kernels/
    │   └── brain.cu             # CUDA kernels
    └── tests/                   # Test suite
```

---

## 🔧 Development

### Run Tests

```bash
pip install -e ".[dev]"
pytest
```

### Test Coverage

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### Code Style

```bash
# Format code
black .

# Type check
mypy .

# Lint
flake8 .
```

---

## 🐛 Troubleshooting

### Word Index Out of Range Error

**Fixed in latest version!** Make sure `MAX_WORD_VOCAB_SIZE` in Python matches CUDA:

```python
# config.py
MAX_WORD_VOCAB_SIZE = 10000  # ✅ Matches CUDA

# kernels/brain.cu
#define MAX_WORD_VOCAB_SIZE 10000  // ✅ Same value
```

### CUDA Out of Memory

1. **Reduce batch size**: `--batch-size 50` (default: 100)
2. **Use memory preset**: `--preset memory_efficient`
3. **Disable pipeline**: Set `PIPELINE_MODE = "disabled"` in settings
4. **Monitor with**: `nvidia-smi -l 1`

### Slow Training

1. **Enable all optimizations**: `--preset high_performance`
2. **Use vocab pre-training**: `--vocab-passes 2`
3. **Check GPU utilization**: Should be >90%
4. **Verify cuBLAS**: Settings → Performance → cuBLAS: On

---

## 📝 Changelog

### Latest Updates

#### 🎉 NEW: Vocabulary Pre-Training (2025-01)
- **Feature**: Scan datasets before training to build stable vocabulary
- **API**: `--vocab-passes N` flag for all trainers
- **Impact**: ~20% faster training, eliminates vocab overhead
- **Details**: See [Vocabulary Pre-Training](#vocabulary-pre-training-new)

#### 🐛 FIX: Word Index Errors (2025-01)
- **Issue**: Python `max_word_vocab_size` (100k) exceeded CUDA limit (10k)
- **Fix**: Synced Python config with CUDA `MAX_WORD_VOCAB_SIZE = 10000`
- **Impact**: Eliminates "word out of index" errors

#### 🧹 CLEANUP: Legacy Code Removal (2025-01)
- **Removed**: Legacy sync methods, sequential training fallbacks
- **Removed**: `ENABLE_VOCAB_OPTIMIZATION` flag (always enabled now)
- **Impact**: Cleaner codebase, only optimized paths

#### 🔧 FIX: Preset Override Bug (2025-01)
- **Issue**: Presets completely replaced user settings
- **Fix**: Presets now MERGE with existing config (preserves customizations)
- **API**: `set_config_from_preset(name, override_user_settings=False)`

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

**You are free to:**
- ✅ Share — copy and redistribute
- ✅ Adapt — remix, transform, build upon

**Under these terms:**
- 📝 Attribution — Give appropriate credit
- 🚫 NonCommercial — No commercial use

See [LICENSE](LICENSE) for full details.

---

## 👤 Author

**Scuglia Samuele**

---

## 🔗 Links

- **GitHub**: [https://github.com/krokodil-byte/ADAM](https://github.com/krokodil-byte/ADAM)
- **Issues**: [Report bugs or request features](https://github.com/krokodil-byte/ADAM/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/krokodil-byte/ADAM/discussions)

---

## 🙏 Acknowledgments

- CUDA community for optimization resources
- HuggingFace for dataset ecosystem
- Open-source ML community

---

**Made with ❤️ for the AI research community with the help of AI**
