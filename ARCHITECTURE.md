# A.D.A.M Architecture Documentation

**A.D.A.M** - Adaptive and Dynamic Agent Module

This document provides a comprehensive overview of the A.D.A.M system architecture, designed for researchers, developers, and anyone interested in understanding the internal workings of this adaptive language model.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Dynamic Vocabulary System](#dynamic-vocabulary-system)
4. [Venn Semantic System](#venn-semantic-system)
5. [Configuration System](#configuration-system)
6. [Dynamic Compilation](#dynamic-compilation)
7. [Training Pipeline](#training-pipeline)
8. [File Structure](#file-structure)
9. [Performance Optimizations](#performance-optimizations)

---

## Overview

A.D.A.M is a transformer-based language model with several key innovations:

- **Hot/Cold Vocabulary Architecture**: Unlimited vocabulary size with GPU caching
- **Multi-Head Venn Semantic System**: Soft clustering for semantic representations
- **Dynamic Architecture**: TUI-configurable model dimensions with runtime compilation
- **Continuous Learning**: Online training with episodic memory
- **Adaptive Generation**: Confidence-based stopping instead of fixed lengths

### Key Statistics (Default Configuration)

| Parameter | Default Value | Configurable |
|-----------|---------------|--------------|
| Embedding Dimension | 768 | ✅ Via TUI |
| Attention Heads | 12 | ✅ Via TUI |
| Transformer Layers | 6 | ✅ Via TUI |
| Max Sequence Length | 512 | ✅ Via TUI |
| Hot Vocabulary Size | 10,000 words | ✅ Via TUI |
| Cold Vocabulary Size | Unlimited | N/A |
| Venn Clusters per Head | 256 | ✅ Via TUI |

---

## Core Architecture

### Transformer Foundation

A.D.A.M uses a standard transformer architecture with custom modifications:

```
Input Text
    ↓
[Tokenization] ← Hot/Cold Vocab System
    ↓
[Embedding Layer] ← Dual: Char + Word Embeddings
    ↓
[Positional Encoding]
    ↓
┌─────────────────────────────┐
│ Transformer Layer 1         │
│  ├─ Multi-Head Attention    │
│  ├─ Venn Semantic Injection │ ← Unique to A.D.A.M
│  ├─ Feed-Forward Network    │
│  └─ LayerNorm + Residual    │
└─────────────────────────────┘
    ↓ (repeated N times)
┌─────────────────────────────┐
│ Transformer Layer N         │
└─────────────────────────────┘
    ↓
[Output Projection] → Vocabulary Logits
    ↓
[Sampling / Argmax] ← Confidence-Based Stopping
    ↓
Output Text
```

### Novel Components

1. **Dual Embedding System**
   - Character embeddings: 256 ASCII/UTF-8 (always available)
   - Word embeddings: Hot (GPU) + Cold (RAM) split

2. **Venn Semantic Injection**
   - Adds semantic cluster information to attention
   - Each head maintains its own cluster space
   - Soft membership allows multi-cluster belonging

3. **Dynamic Architecture**
   - Model dimensions configurable at runtime
   - CUDA kernel recompiled when architecture changes
   - No hardcoded dimensions in production code

---

## Dynamic Vocabulary System

### Problem: Fixed Vocabulary Limitations

Traditional language models use fixed vocabularies (e.g., 50k BPE tokens). This has several issues:

- Cannot learn new words without retraining tokenizer
- Vocabulary size limited by GPU memory
- Poor handling of rare words and typos

### Solution: Hot/Cold Architecture

A.D.A.M uses a three-tier vocabulary system:

#### Tier 1: Character Fallback (256 tokens)
- Always available on GPU
- Used for unknown character sequences
- Ensures model never encounters OOV (out-of-vocabulary)

#### Tier 2: Hot Vocabulary (10,000 words)
- Most frequently used words
- Stored on GPU for fast access
- LRU eviction when full

#### Tier 3: Cold Vocabulary (Unlimited)
- All other learned words
- Stored in RAM
- Embeddings computed from character composition
- Swapped to hot vocab on demand

### Vocabulary Learning Process

```
1. Text Input: "The quick brown fox"
   ↓
2. Tokenize:
   - "The" → Known word (hot vocab)
   - "quick" → Unknown, try cold vocab
   - "brown" → Unknown, use chars: ['b','r','o','w','n']
   ↓
3. Track Frequency:
   - ['b','r','o','w','n'] appears multiple times
   ↓
4. Word Creation (threshold: 5 occurrences):
   - Create word token for "brown"
   - Add to cold vocabulary
   - Compute initial embedding from char sequence
   ↓
5. Promotion to Hot Vocab (LRU):
   - If "brown" is used frequently
   - Evict least recently used hot word
   - Promote "brown" to hot vocab (GPU)
```

### Implementation Details

**File**: `core/vocabulary.py`

```python
class DynamicVocabulary:
    def __init__(self):
        self.word_to_id = {}          # Word → token ID
        self.id_to_word = {}          # Token ID → word
        self.frequency = {}           # Word → frequency count
        self.hot_vocab = set()        # GPU-resident word IDs
        self.cold_vocab = {}          # RAM: word_id → embedding
```

**File**: `kernels/vocabulary.cu`
- GPU-side word management
- Fast lookup tables
- LRU eviction logic

---

## Venn Semantic System

### Concept: Soft Semantic Clustering

Traditional word embeddings place each token at a single point in semantic space. The Venn system extends this by:

1. Dividing semantic space into **clusters** (like Venn diagram regions)
2. Allowing tokens to have **soft membership** in multiple clusters
3. Propagating activations between **similar clusters**

### Multi-Head Architecture

Each attention head maintains its own cluster space:

```
Head 1 Clusters: [Geography, Politics, History, ...]
Head 2 Clusters: [Colors, Shapes, Sizes, ...]
Head 3 Clusters: [Emotions, Actions, States, ...]
...
```

This allows different heads to specialize in different semantic aspects.

### Example: The Word "Green"

```
Token: "green"
Embedding: [0.1, 0.3, -0.2, ...]

Head 1 (Concrete Concepts):
├─ Cluster 42 "Colors" → 70% membership
├─ Cluster 15 "Nature" → 30% membership
└─ Activation spreads to nearby clusters

Head 2 (Abstract Concepts):
├─ Cluster 89 "Environmental" → 50% membership
├─ Cluster 12 "Positive Qualities" → 20% membership
└─ Activation spreads

Combined representation captures:
- Physical color (concrete)
- Environmental meaning (abstract)
- Positive connotation (sentiment)
```

### Cluster Update (Online K-Means)

```
1. Token processed: "green"
   ↓
2. Find nearest clusters in each head
   ↓
3. Compute membership weights (Gaussian distance)
   ↓
4. Update cluster centers (moving average):
   cluster_center += lr * (token_embedding - cluster_center)
   ↓
5. Propagate activations to similar clusters
```

### Configuration

**File**: `core/config.py` → `ModelConfig`

```python
# Venn system parameters
ENABLE_VENN_MULTIHEAD: bool = True
NUM_VENN_HEADS: int = 12              # Matches attention heads
VENN_CLUSTERS_PER_HEAD: int = 256     # Clusters per head
VENN_PROPAGATION_FACTOR: float = 0.2  # Activation spread
CLUSTER_UPDATE_LR: float = 0.1        # Learning rate
```

**File**: `kernels/venn_system.cu`
- CUDA implementation of cluster updates
- Parallel propagation kernels
- Membership computation

---

## Configuration System

### Design Philosophy

A.D.A.M's configuration system is designed around three principles:

1. **Centralization**: All config in `core/config.py`
2. **Type Safety**: Dataclass-based with type hints
3. **Preset Support**: Pre-configured setups for common use cases

### Configuration Classes

#### 1. ModelConfig
**Controls**: Architecture, vocabulary, Venn system
**Protected**: Architecture params NEVER overridden by presets
**Example**:
```python
MODEL_CONFIG.NUM_LAYERS = 12    # Set by TUI only
MODEL_CONFIG.EMBED_DIM = 1024   # Set by TUI only
MODEL_CONFIG.WORD_CREATION_THRESHOLD = 5  # Can be preset
```

#### 2. TrainingConfig
**Controls**: Learning rates, batch sizes, validation
**Example**:
```python
TRAINING_CONFIG.BASE_LR = 0.0001
TRAINING_CONFIG.VALIDATION_SPLIT = 0.1
```

#### 3. PerformanceConfig
**Controls**: GPU optimizations, pipeline mode
**Example**:
```python
PERFORMANCE_CONFIG.USE_CUBLAS = True
PERFORMANCE_CONFIG.PIPELINE_MODE = "double"
```

#### 4. VocabOptimizationConfig
**Controls**: Hot/cold vocab behavior
**Example**:
```python
VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB = 10000
VOCAB_OPTIMIZATION_CONFIG.LRU_EVICTION = True
```

### TUI Integration

The TUI (Text User Interface) provides a user-friendly way to configure A.D.A.M:

```
1. User opens TUI (adam command without args)
   ↓
2. Navigates to Settings → Architecture
   ↓
3. Changes: NUM_LAYERS=12, EMBED_DIM=1024
   ↓
4. Saves settings → ~/.adam/tui_settings.json
   ↓
5. Next training run:
   - load_tui_settings() loads JSON
   - Applies to MODEL_CONFIG
   - Triggers CUDA recompilation (if architecture changed)
```

**File**: `modules/tui.py`

### Preset System

Presets provide pre-configured settings for common scenarios:

| Preset | Use Case | Key Changes |
|--------|----------|-------------|
| `none` | Use TUI settings only | No changes |
| `default` | Balanced performance | Default values |
| `fast_learning` | Quick adaptation | Higher LR, frequent updates |
| `stable` | Production inference | Low LR, conservative |
| `research` | Experimentation | Aggressive word creation, high Venn propagation |
| `high_performance` | Maximum speed | All GPU optimizations enabled |

**Important**: Presets NEVER modify architecture parameters (NUM_LAYERS, EMBED_DIM, etc.). Architecture is TUI-only.

### Usage Example

```python
from core.config import load_tui_settings, set_config_from_preset, MODEL_CONFIG

# Load TUI settings (user preferences)
load_tui_settings()

# Apply preset for behavior (preserves TUI architecture)
set_config_from_preset('fast_learning')

# Access config
print(f"Layers: {MODEL_CONFIG.NUM_LAYERS}")  # From TUI
print(f"LR: {TRAINING_CONFIG.BASE_LR}")      # From preset
```

---

## Dynamic Compilation

### Problem: Hardcoded Architecture

Traditional CUDA kernels use compile-time constants:

```c
#define NUM_LAYERS 6
#define EMBED_DIM 768
```

This requires recompilation to change model architecture, which is slow and error-prone.

### Solution: Template-Based Compilation

A.D.A.M uses a template-based approach:

```
┌─────────────────────────┐
│ TUI Settings            │
│ NUM_LAYERS: 12          │
│ EMBED_DIM: 1024         │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ core/config.py          │
│ MODEL_CONFIG.NUM_LAYERS │
│ MODEL_CONFIG.EMBED_DIM  │
└──────────┬──────────────┘
           ↓
┌─────────────────────────────────────┐
│ core/brain_wrapper.py (CUDACompiler)│
│                                     │
│ 1. Read kernels/brain.cu            │
│ 2. Replace #define with config      │
│ 3. Write to temp file               │
│ 4. Compile with nvcc                │
│ 5. Cache compiled .so               │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────┐
│ Compiled Kernel         │
│ libvectllm_<hash>.so    │
│ (architecture-specific) │
└─────────────────────────┘
```

### Compilation Pipeline

**File**: `core/brain_wrapper.py`

```python
class CUDACompiler:
    def compile(self, kernel_path: Path) -> Path:
        # 1. Read source
        with open(kernel_path, 'r') as f:
            kernel_code = f.read()

        # 2. Inject config values
        replacements = {
            '#define NUM_LAYERS 6':
                f'#define NUM_LAYERS {MODEL_CONFIG.NUM_LAYERS}',
            '#define EMBED_DIM 768':
                f'#define EMBED_DIM {MODEL_CONFIG.EMBED_DIM}',
            # ... etc
        }

        for old, new in replacements.items():
            kernel_code = kernel_code.replace(old, new)

        # 3. Compile modified source
        nvcc.compile(kernel_code)
```

### Cache Invalidation

The compiler computes a hash including:
- Source code
- Architecture parameters
- Venn parameters

When any of these change, the kernel is recompiled:

```python
def get_cache_path(self, kernel_path: Path) -> Path:
    with open(kernel_path, 'rb') as f:
        code = f.read()

    arch_params = f"{MODEL_CONFIG.NUM_LAYERS}_{MODEL_CONFIG.EMBED_DIM}_..."
    combined = code + arch_params.encode()

    code_hash = hashlib.sha256(combined).hexdigest()[:16]
    return cache_dir / f"libvectllm_{code_hash}.so"
```

### Runtime Configuration

After compilation, runtime values are passed via `set_model_config()`:

```c
extern "C" void set_model_config(int num_layers, int embed_dim,
                                  int num_heads, int max_seq_len) {
    runtime_num_layers = num_layers;
    runtime_embed_dim = embed_dim;
    runtime_num_heads = num_heads;
    runtime_max_seq_len = max_seq_len;
}
```

This allows:
- Validation that compiled values match runtime values
- Dynamic memory allocation based on config
- Logging of active configuration

---

## Training Pipeline

### Continuous Learning

A.D.A.M supports continuous online learning:

```
Input Stream (Wikipedia, datasets, chat, etc.)
    ↓
┌─────────────────────────┐
│ Preprocessing           │
│ - Tokenization          │
│ - Vocab update          │
│ - Batch formation       │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Forward Pass            │
│ - Embedding lookup      │
│ - Transformer layers    │
│ - Venn semantic inject  │
│ - Output projection     │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Loss Computation        │
│ - Cross-entropy (current)│
│ - Top-K accuracy (future)│
│ - Venn semantic reward  │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Backward Pass           │
│ - Gradient computation  │
│ - Venn cluster updates  │
│ - Parameter updates     │
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ Sync & Checkpoint       │
│ - Vocab sync (GPU→CPU)  │
│ - Stats collection      │
│ - Checkpoint save       │
└─────────────────────────┘
```

### Training Modes

#### 1. Dataset Training
**File**: `modules/dataset_training.py`

```bash
adam train-dataset --path ./data --passes 3 --preset fast_learning
```

Supports:
- Local text files
- HuggingFace datasets
- Multi-pass training
- Validation split

#### 2. Wikipedia Training
**File**: `modules/wikipedia_training.py`

```bash
adam train-wiki --categories "Science" "Technology" --max-articles 1000
```

Supports:
- API-based streaming
- Category filtering
- Auto-checkpoint
- Continuous streaming mode

#### 3. Interactive Chat
**File**: `modules/chat.py`

```bash
adam chat --load checkpoints/model.ckpt
```

Supports:
- User conversations
- Online learning from chat
- Confidence-based generation
- Context preservation

### Validation Strategy

A.D.A.M supports flexible validation:

```python
# Per-pass validation (default)
TRAINING_CONFIG.VALIDATE_PER_PASS = True

# Periodic validation
TRAINING_CONFIG.VALIDATE_PER_PASS = False
TRAINING_CONFIG.VALIDATION_FREQUENCY = 100  # Every 100 samples

# Early stopping
TRAINING_CONFIG.EARLY_STOPPING_PATIENCE = 5
```

---

## File Structure

```
A.D.A.M — Adaptive and Dynamic Agent Module/
├── core/                          # Core system components
│   ├── config.py                  # Centralized configuration
│   ├── constants.py               # Constants (deprecated, use config.py)
│   ├── brain_wrapper.py           # Python wrapper for CUDA kernel
│   ├── vocabulary.py              # Dynamic vocabulary manager
│   ├── stats.py                   # Statistics collection
│   └── pipeline.py                # Training pipeline
│
├── kernels/                       # CUDA kernel implementations
│   ├── brain.cu                   # Main unified kernel
│   ├── vocabulary.cu              # Vocab hot/cold management
│   └── venn_system.cu             # Venn semantic clustering
│
├── modules/                       # High-level modules
│   ├── dataset_training.py        # Dataset training logic
│   ├── wikipedia_training.py      # Wikipedia streaming training
│   ├── chat.py                    # Interactive chat interface
│   ├── tui.py                     # Text User Interface
│   └── training_logger.py         # Training progress logging
│
├── cli/                           # Command-line interface
│   └── adam.py                    # Main CLI entry point
│
├── Utils/                         # Utility modules
│   ├── checkpoint.py              # Checkpoint save/load
│   ├── compiler.py                # CUDA compilation helpers
│   └── tokenizer.py               # Tokenization utilities
│
├── tests/                         # Unit tests
│   ├── test_config.py
│   ├── test_vocabulary.py
│   └── ...
│
├── ARCHITECTURE.md                # This file
└── README.md                      # User guide
```

### Key File Responsibilities

| File | Responsibility |
|------|----------------|
| `core/config.py` | All configuration, constants, presets |
| `core/brain_wrapper.py` | CUDA compilation, Python↔CUDA bridge |
| `core/vocabulary.py` | Hot/cold vocab management, word creation |
| `kernels/brain.cu` | Main CUDA kernel, transformer implementation |
| `kernels/venn_system.cu` | Venn clustering, propagation |
| `modules/tui.py` | User interface for configuration |
| `cli/adam.py` | Command-line interface, workflow orchestration |

---

## Performance Optimizations

### GPU Optimizations

#### 1. cuBLAS Integration
- Matrix multiplications offloaded to NVIDIA's tuned library
- 2-3x speedup on attention and FFN layers

#### 2. Fused Kernels
- Attention + FFN fusion reduces kernel launch overhead
- Embedding + LayerNorm fusion improves cache locality

#### 3. Pipeline Parallelism
```
Mode: Triple Pipeline

[CPU Thread 1]     [CPU Thread 2]     [GPU Stream 1]    [GPU Stream 2]
Load Batch 1   →   Preprocess     →   H2D Transfer  →
                   Load Batch 2   →   Preprocess     →  H2D Transfer  →
                                      Load Batch 3   →  Compute Batch 1  →
                                                         Compute Batch 2  →
                                                         D2H Results 1
```

#### 4. Memory Optimizations
- Pinned memory for faster CPU↔GPU transfers
- Pre-allocated buffers (no runtime malloc)
- Warp-level primitives for reductions

### Vocabulary Optimizations

#### 1. Deferred Sync
```
During Training:
- Vocabulary changes accumulate in buffer
- No immediate GPU sync (avoids contention)

End of Batch:
- Batch sync all changes (single GPU call)
- Reduces sync overhead by 10-100x
```

#### 2. Pre-loading
```
Before Forward Pass:
1. Scan input tokens
2. Identify cold vocab words
3. Batch load to hot vocab (evict LRU)
4. Process forward pass (all tokens in hot vocab)
```

#### 3. Cached Embeddings
- Character embeddings cached on CPU
- Refresh every 1000 syncs
- Avoids repeated GPU reads

---

## Future Directions

### Planned Features

1. **Reward-Based Learning**
   - Replace cross-entropy with Top-K accuracy
   - Add Venn semantic similarity reward
   - Combined reward: `α * topk + (1-α) * venn`

2. **Multi-GPU Support**
   - Model parallelism across layers
   - Data parallelism for training

3. **Quantization**
   - INT8/FP16 inference
   - Dynamic quantization during training

4. **Retrieval Augmentation**
   - External knowledge base integration
   - Episodic memory expansion

5. **Recursive Improvement**
   - Self-generated training data
   - Bootstrapped learning

---

## Contributing

When contributing to A.D.A.M, please:

1. **Follow the config system**: Add new params to `core/config.py` dataclasses
2. **Document CUDA changes**: Update kernel headers if modifying architecture
3. **Update this doc**: Keep ARCHITECTURE.md in sync with code
4. **Add tests**: Unit tests for new features
5. **Preserve TUI architecture**: Never let presets override architecture params

---

## References

- Configuration: `core/config.py`
- CUDA Kernel: `kernels/brain.cu`
- Vocabulary: `core/vocabulary.py`, `kernels/vocabulary.cu`
- Venn System: `kernels/venn_system.cu`
- Training: `modules/*_training.py`
- TUI: `modules/tui.py`
- CLI: `cli/adam.py`

---

**Last Updated**: 2025-11-29
**Version**: 1.0
**Maintainer**: A.D.A.M Development Team
