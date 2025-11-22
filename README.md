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

### Model Architecture

| Setting | Default | Description |
|---------|---------|-------------|
| `num_layers` | 6 | Number of transformer layers. More layers = more capacity but slower |
| `embed_dim` | 768 | Embedding dimension. Options: 64, 128, 256, 512, 768, 1024, 2048, 4096 |
| `num_heads` | 12 | Number of attention heads. Should divide embed_dim evenly |
| `max_seq_len` | 512 | Maximum sequence length in tokens |
| `max_word_vocab_size` | 100000 | Maximum number of word tokens in dynamic vocabulary |
| `word_creation_threshold` | 5 | Create new word token after N occurrences |
| `word_pruning_threshold` | 2 | Remove word if frequency below N |
| `venn_clusters` | 256 | Number of semantic clusters for Venn system |

### Training Parameters

| Setting | Default | Description |
|---------|---------|-------------|
| `base_lr` | 0.0001 | Base learning rate. Higher = faster but less stable |
| `embedding_lr_scale` | 0.1 | Embedding LR = base_lr × this value |
| `momentum` | 0.9 | Momentum for optimizer. Higher = smoother updates |
| `temperature` | 1.0 | Exploration temperature. Higher = more random sampling |
| `venn_update_freq` | 100 | Update semantic clusters every N cycles |
| `stats_sync_freq` | 10 | Sync GPU statistics every N cycles |
| `vocab_pruning_freq` | 10000 | Prune vocabulary every N cycles |

### Wikipedia Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `language` | `en` | Wikipedia language code |
| `batch_size` | 100 | Number of articles per batch |
| `min_article_length` | 500 | Minimum article length in characters |
| `max_article_length` | 50000 | Maximum article length in characters |
| `auto_save_interval` | 100 | Auto-save every N articles |

### System Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `device_id` | 0 | CUDA device ID (for multi-GPU systems) |
| `nvcc_arch` | auto | NVCC architecture. `auto` or specific like `sm_86` |
| `checkpoint_interval` | 100 | Auto-checkpoint every N samples |

## Configuration Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default` | Balanced settings | General use |
| `fast_learning` | Higher LR (0.001), lower momentum (0.7), higher temperature (1.5) | Quick experiments, small datasets |
| `stable` | Lower LR (0.00001), higher momentum (0.95), lower temperature (0.7) | Production, large datasets |
| `inference` | LR=0, momentum=0, temperature=0.5 | Generation only, no training |
| `research` | Medium LR (0.0005), frequent cluster updates | Experimentation, analysis |
| `high_performance` | All GPU optimizations enabled, 90% target utilization | Maximum speed |
| `memory_efficient` | Fused kernels, no pipeline buffers | Limited GPU memory |
| `max_throughput` | Triple buffer, 2 compute streams, 95% target | Absolute maximum speed |

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
