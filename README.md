# ADAM - Adaptive and Dynamic Agent Module

Continuous Self-Training Language Model with Dynamic Vocabulary

## Overview

ADAM is an experimental language model that features continuous self-training capabilities and dynamic vocabulary expansion. Built with CUDA acceleration support for high-performance training and inference.

## Features

- **Dynamic Vocabulary**: Automatically expands vocabulary during training
- **Continuous Self-Training**: Model improves over time with new data
- **CUDA Acceleration**: GPU-optimized kernels for fast computation
- **GPU Optimizations**: cuBLAS GEMM, fused kernels, pipeline overlap (50-100x speedup)
- **Multiple Training Sources**: Support for text files, datasets, and Wikipedia dumps
- **Wikipedia API Streaming**: Train directly from Wikipedia with automatic RAM management
- **Interactive Chat**: Real-time conversation with trained models
- **Checkpoint Management**: Save and resume training sessions
- **Full TUI Dashboard**: Complete graphical interface with performance controls

## Installation

### From Source (Development)

```bash
git clone https://github.com/krokodil-byte/ADAM.git
cd ADAM
pip install -e .
```

### Quick Run (No Installation)

```bash
python run.py [command] [args]
```

## Usage

### TUI Dashboard

```bash
adam
```

Opens the full TUI dashboard with all operations. This is the recommended way to use ADAM.

### Initialize System

```bash
adam init
```

Checks CUDA availability and compiles the GPU kernels. Run this first to verify your setup.

### Train on Text File

```bash
adam train input.txt -o model.ckpt -p 5 --preset high_performance
```

Options:
- `-o, --output`: Output checkpoint file
- `-c, --checkpoint`: Resume from existing checkpoint
- `-v, --vocab`: Load vocabulary file
- `-p, --passes`: Number of training passes (default: 1)
- `--preset`: Configuration preset (see Presets section)
- `--auto-save N`: Auto-save checkpoint every N passes
- `--prune-vocab`: Prune rare words after training

### Train on Dataset

```bash
adam dataset /path/to/dataset -o model.ckpt -p 3
```

Options:
- `-o, --output`: Output checkpoint
- `-c, --checkpoint`: Resume from checkpoint
- `-p, --passes`: Number of passes (default: 1)
- `--preset`: Configuration preset
- `--auto-save N`: Auto-save every N files
- `--extensions`: File extensions to include (e.g., `.txt,.md`)

### Train on Wikipedia

#### From Local Dump

```bash
adam wikipedia dump.xml -o model.ckpt --max-articles 1000
```

#### From Wikipedia API (Streaming)

```bash
adam wikipedia -o model.ckpt --language en --batch-size 100
```

This mode streams articles directly from Wikipedia API:
1. Downloads batch of articles
2. Trains on the batch
3. Clears memory
4. Repeats

Options:
- `--language`: Wikipedia language code (default: `en`)
  - Examples: `en`, `it`, `de`, `fr`, `es`, `ja`, `zh`
- `--batch-size`: Number of articles per batch (default: `100`)
  - Lower = faster iterations, less memory
  - Higher = larger batches, more context per cycle
- `--max-articles`: Maximum articles to process (default: unlimited)
- `-p, --passes`: Training passes per batch (default: 1)
- `-o, --output`: Output checkpoint
- `-c, --checkpoint`: Resume from checkpoint
- `--auto-save N`: Auto-save every N articles
- `--preset`: Configuration preset

### Interactive Chat

```bash
adam chat -c model.ckpt
```

Start an interactive conversation with a trained model.

### Generate Text

```bash
adam generate -c model.ckpt
```

Generate text interactively with prompts.

Options:
- `-c, --checkpoint`: Checkpoint file (required)
- `--temperature`: Sampling temperature

### View Statistics

```bash
adam stats -c model.ckpt
```

Display model statistics including:
- Training cycles
- Tokens processed
- Current loss and perplexity
- Temperature and momentum
- Vocabulary size and utilization

### Vocabulary Management

```bash
# View vocabulary stats
adam vocab stats -f vocab.json

# Prune rare words
adam vocab prune -f vocab.json
```

### ADAM TUI Dashboard

```bash
adam
```

Opens the complete TUI dashboard with all operations. You can also use `adam dashboard`.

**Main Menu:**
- 🚀 Initialize System
- 🧠 Train on Text
- 📚 Wikipedia Training
- 📂 Dataset Training
- 💬 Interactive Chat
- 📊 View Statistics
- ⚙️ Settings
- 🚪 Exit

**Settings Menu:**
- 🏗️ Model Architecture
- 📈 Training Parameters
- ⚡ Performance (GPU optimizations)
- 🖥️ System
- 💾 Save Settings

**Navigation:**
- `↑↓` Arrow keys to move
- `Enter` to select
- `Q` to go back/quit
- `Esc` to cancel dialogs

## Configuration Settings

### 🧠 Hot/Cold Vocabulary Architecture

**Philosophy**: Learn everything (RAM), cache what's used (GPU).

ADAM uses a two-tier vocabulary system:

- **Cold Vocab (RAM)**: Unlimited semantic memory
  - Stores ALL words ever seen (millions possible)
  - Persisted to disk with checkpoint
  - Pre-initialized embeddings for stability
  - **La RAM è abbondante!** - let it grow

- **Hot Vocab (GPU)**: Fast working memory
  - LRU cache of 10,000 most-used words
  - Ultra-fast access for training/inference
  - Automatic eviction and pre-loading
  - Aligned with kernel MAX_WORD_VOCAB_SIZE

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_HOT_VOCAB` | 10000 | GPU cache size (10k words = ~30MB) |
| `WORD_CREATION_THRESHOLD` | 5 | Create word after N occurrences (prevents typo spam) |
| `WORD_PRUNING_THRESHOLD` | 0 | 0 = never prune (infinite RAM vocab recommended) |
| `MAX_WORD_LENGTH` | 20 | Max chars per word (sanity check) |
| `ENABLE_TOKEN_PRELOADING` | True | Auto-load tokens from cold→hot before forward pass |
| `PRELOAD_BATCH_SIZE` | 100 | Max tokens to preload per batch |
| `SAVE_COLD_VOCAB` | True | Persist cold embeddings to disk (.cold.npz) |
| `AUTO_LOAD_COLD` | True | Auto-load cold vocab from checkpoint |

### 🎯 Venn Semantic System

ADAM's Venn system creates semantic clusters for generalization:

**Activation & Propagation**:
- Each token activates nearby clusters (Gaussian)
- Activation propagates through similar clusters
- Enables semantic reasoning beyond memorization

| Setting | Default | Description |
|---------|---------|-------------|
| `VENN_CLUSTERS` | 256 | Number of semantic clusters |
| `VENN_PROPAGATION_FACTOR` | 0.2 | How much activations propagate (0.0-1.0) <br>Higher = more generalization |
| `VENN_INTERSECTION_THRESHOLD` | 0.3 | Similarity threshold for cluster connections (0.0-1.0) <br>Lower = more connections |
| `MAX_PROPAGATED_ACTIVATION` | 5.0 | Cap for propagated activations (prevents explosion) |
| `VENN_ACTIVATION_TEMPERATURE` | 1.0 | Gaussian temperature (higher = broader activation) |
| `PRIMARY_MEMBERSHIP_WEIGHT` | 0.6 | Weight for closest cluster |
| `SECONDARY_MEMBERSHIP_WEIGHT` | 0.4 | Weight for second-closest cluster |
| `CLUSTER_UPDATE_LR` | 0.1 | Learning rate for cluster center updates |
| `VENN_UPDATE_FREQUENCY` | 100 | Update clusters every N training cycles |

**Tuning Tips**:
- ↑ `VENN_PROPAGATION_FACTOR` = more creative but less precise
- ↓ `VENN_INTERSECTION_THRESHOLD` = more semantic connections
- ↑ `VENN_CLUSTERS` = finer-grained semantics (but slower)

### 🎓 Training Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| **Learning Rates** |
| `BASE_LR` | 0.0001 | Base learning rate (conservative for continuous training) |
| `EMBEDDING_LR_SCALE` | 0.1 | Embeddings LR = BASE_LR × 0.1 (10x slower = stable) |
| `OUTPUT_LR_SCALE` | 1.0 | Output weights LR = BASE_LR × 1.0 (normal speed) |
| `MOMENTUM` | 0.9 | SGD momentum (0.0-1.0, higher = smoother) |
| **Validation** |
| `VALIDATION_SPLIT` | 0.1 | 10% of data for validation |
| `VALIDATE_PER_PASS` | True | Validate at end of pass (vs every N samples) |
| `VALIDATION_FREQUENCY` | 100 | If VALIDATE_PER_PASS=False, validate every N samples |
| `EARLY_STOPPING_PATIENCE` | 5 | Stop after N validations without improvement |
| `MIN_VALIDATION_SAMPLES` | 10 | Minimum samples needed for validation |
| **Checkpointing** |
| `AUTO_SAVE_FREQUENCY` | 1000 | Auto-save every N articles/samples |
| `CHECKPOINT_DIR` | checkpoints/ | Directory for checkpoint files |

### Model Architecture

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBED_DIM` | 768 | Embedding dimension (64/128/256/512/768/1024) |
| `NUM_HEADS` | 12 | Attention heads (must divide EMBED_DIM evenly) |
| `NUM_LAYERS` | 6 | Transformer layers (more = better but slower) |
| `MAX_SEQ_LEN` | 512 | Maximum sequence length in tokens |
| `CHAR_VOCAB_SIZE` | 256 | ASCII/UTF-8 base vocabulary (fixed) |

## Configuration Presets

Quick configuration profiles for different use cases:

### Training Presets

| Preset | Key Settings | Best For |
|--------|--------------|----------|
| **default** | BASE_LR=0.0001, MOMENTUM=0.9 | General-purpose training, balanced |
| **fast_learning** | BASE_LR=0.001 (10x higher)<br>MOMENTUM=0.7 (lower)<br>VENN_UPDATE_FREQ=50 (2x faster) | Quick experiments<br>Small datasets<br>Rapid iteration |
| **stable** | BASE_LR=0.00001 (10x lower)<br>MOMENTUM=0.95 (higher)<br>WORD_PRUNING=2 (prunes rare words) | Production training<br>Large datasets<br>Long training runs |
| **research** | BASE_LR=0.0005 (medium)<br>VENN_UPDATE_FREQ=25 (4x faster) | Experimentation<br>Venn system research<br>Frequent cluster updates |
| **inference** | BASE_LR=0 (no training)<br>WORD_CREATION=0 (frozen vocab)<br>DEFERRED_SYNC=False | Generation only<br>Checkpoint evaluation<br>Chat mode |

### Performance Presets

| Preset | Key Optimizations | Best For |
|--------|-------------------|----------|
| **high_performance** | cuBLAS: On<br>Fused kernels: On<br>Pipeline: double<br>GPU target: 90% | Maximum speed<br>Modern GPUs (RTX 30xx/40xx)<br>96k+ tok/s achievable |
| **memory_efficient** | Fused kernels: On<br>Pipeline: disabled<br>Pinned memory: Off<br>Preallocate: Off | Limited VRAM (6-8GB)<br>Older GPUs<br>Multiple models on same GPU |
| **max_throughput** | Pipeline: triple<br>Compute streams: 2<br>Pinned memory: On<br>GPU target: 95% | Absolute maximum speed<br>High-end GPUs (A100, H100)<br>Training speed priority |

**Usage**:
```bash
adam wikipedia dump.xml --preset fast_learning
adam dataset ./data --preset high_performance
adam chat -c model.ckpt --preset inference
```

## GPU Optimizations

ADAM includes extensive GPU optimizations that provide 50-100x speedup:

### Tier 1 Optimizations
- **cuBLAS GEMM**: Matrix operations use optimized cuBLAS instead of manual loops
- **Block-reduce**: Cross-entropy loss uses block-level reduction (eliminates atomic contention)
- **Parallel Venn**: 256 threads per block instead of 1 for intersection computation
- **Coalesced access**: Embedding backward with better memory patterns

### Tier 2 Optimizations
- **Pipeline overlap**: Concurrent H2D/compute/D2H using multiple streams
- **Fused kernels**: Attention+FFN combined to reduce memory bandwidth
- **Warp primitives**: Warp-level shuffle for efficient reductions

### Performance Settings (TUI)

```
⚡ Performance Settings
├── 🔢 Use cuBLAS: On/Off
├── 🧩 Fused Kernels: On/Off
├── 🔀 Pipeline Mode: disabled/double/triple
├── ⚡ Async Transfers: On/Off
├── 📌 Pinned Memory: On/Off
├── 🔄 Warp Primitives: On/Off
└── 🎯 GPU Target: 80%
```

## Project Structure

```
ADAM/
├── setup.py                 # Package installation
├── run.py                   # Development runner
├── pytest.ini               # Test configuration
├── README.md
├── LICENSE
└── A.D.A.M — Adaptive and Dynamic Agent Module/
    ├── __main__.py          # Module entry point
    ├── cli/
    │   └── adam.py          # CLI implementation (run with 'adam' command)
    ├── core/
    │   ├── brain_wrapper.py # Main model wrapper
    │   ├── vocabulary.py    # Dynamic vocabulary
    │   ├── config.py        # Configuration
    │   ├── stats.py         # Statistics collector
    │   ├── constants.py     # System constants
    │   └── exceptions.py    # Custom exceptions
    ├── modules/
    │   ├── training.py      # Multi-pass trainer
    │   ├── chat.py          # Interactive chat
    │   ├── tui.py           # Full TUI interface
    │   ├── dataset_training.py
    │   └── wikipedia_training.py
    ├── Utils/
    │   ├── checkpoint.py    # Checkpoint management
    │   ├── tokenizer.py     # Text tokenization
    │   └── compiler.py      # CUDA compilation
    ├── kernels/             # CUDA kernel files
    └── tests/               # Test suite
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- CUDA toolkit (optional, for GPU acceleration)

## Development

### Run Tests

```bash
pip install -e ".[dev]"
pytest
```

### Test Coverage

```bash
pytest --cov=. --cov-report=html
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit
- **NonCommercial** — You may not use the material for commercial purposes

See [LICENSE](LICENSE) for details.

## Author

Scuglia Samuele

## Links

- [GitHub Repository](https://github.com/krokodil-byte/ADAM)
