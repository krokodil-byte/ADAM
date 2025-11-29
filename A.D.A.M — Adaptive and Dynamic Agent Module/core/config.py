#!/usr/bin/env python3
"""
VectLLM Configuration System
============================

This module centralizes ALL configuration for the A.D.A.M (Adaptive and Dynamic Agent Module).
It provides a clean dataclass-based config system with preset support and TUI integration.

Architecture:
- ModelConfig: Model architecture (layers, dimensions, vocab, Venn system)
- TrainingConfig: Learning rates, batch sizes, validation
- PerformanceConfig: GPU optimizations, cuBLAS, memory settings
- GenerationConfig: Text generation parameters
- CheckpointConfig: Model persistence
- RuntimeConfig: GPU device, compilation flags
- VocabOptimizationConfig: Hot/cold vocab architecture

Key Features:
- TUI integration: Settings from ~/.adam/tui_settings.json
- Preset system: Pre-configured settings for different use cases
- Architecture protection: Presets NEVER override TUI architecture settings
- Dynamic compilation: Config values injected into CUDA #define at compile time

Usage:
    from core.config import MODEL_CONFIG, TRAINING_CONFIG

    # Load TUI settings (if available)
    load_tui_settings()

    # Apply preset (preserves TUI architecture)
    set_config_from_preset('fast_learning')

    # Access settings
    print(MODEL_CONFIG.NUM_LAYERS)  # 6 (or TUI value)
"""

from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# CONSTANTS
# ============================================================================
# These constants are used throughout the codebase for magic numbers and
# file format specifications.

# ASCII Characters
ASCII_SPACE = 32
ASCII_NEWLINE = 10
ASCII_TAB = 9
ASCII_CR = 13

# Checkpoint Magic Numbers (for backward compatibility detection)
CHECKPOINT_VERSION = 3
MAGIC_V3 = "VECTLLM3"
MAGIC_V2 = "VECTLLM2"
MAGIC_V1 = "VECTLLM"

# File Extensions
CHECKPOINT_EXT = ".ckpt"
VOCAB_EXT = ".vocab"
FREQ_EXT = ".freq"
METADATA_EXT = ".json"

# Formatting thresholds for human-readable numbers (1.2K, 3.5M, etc.)
NUMBER_THRESHOLDS = {
    'K': 1_000,
    'M': 1_000_000,
    'B': 1_000_000_000,
}


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class ModelConfig:
    """
    Model Architecture Configuration
    =================================

    Controls the fundamental architecture of the transformer model.

    IMPORTANT: These architecture parameters are NEVER overridden by presets.
    They can only be set via TUI (Text User Interface) to ensure consistent
    model architecture across training sessions.

    Architecture Parameters:
    - EMBED_DIM: Embedding dimension (must be divisible by NUM_HEADS)
    - NUM_HEADS: Number of attention heads
    - NUM_LAYERS: Number of transformer layers
    - MAX_SEQ_LEN: Maximum sequence length for positional encoding

    Vocabulary System:
    A.D.A.M uses a hot/cold vocabulary architecture:
    - Hot vocab (GPU): 10,000 most frequent words (fast access)
    - Cold vocab (RAM): Unlimited size (LRU eviction to hot)
    - Character fallback: 256 ASCII/UTF-8 characters (always available)

    Venn Semantic System:
    Multi-head semantic clustering for context-aware representations:
    - Each head maintains its own cluster space
    - Tokens can have soft membership across multiple clusters
    - Clusters are updated online during training
    """

    # ========================================
    # Model Dimensions (TUI-controlled)
    # ========================================
    EMBED_DIM: int = 768        # Embedding dimension
    NUM_HEADS: int = 12         # Attention heads (EMBED_DIM must be divisible by this)
    NUM_LAYERS: int = 6         # Transformer layers
    MAX_SEQ_LEN: int = 512      # Maximum sequence length

    # ========================================
    # Dynamic Vocabulary (Hot/Cold Architecture)
    # ========================================
    CHAR_VOCAB_SIZE: int = 256                  # ASCII/UTF-8 character set (always available)
    WORD_CREATION_THRESHOLD: int = 5            # Create word token after N char sequence occurrences
    WORD_PRUNING_THRESHOLD: int = 0             # Pruning threshold (0 = never prune cold vocab)
    MAX_WORD_LENGTH: int = 20                   # Max word length in characters (sanity check)
    MAX_WORD_VOCAB_SIZE: int = 10000            # Hot vocab size (MUST match CUDA #define)

    # ========================================
    # Venn Semantic System - Multi-Head
    # ========================================
    # The Venn system creates semantic clusters that capture token meanings.
    # Multi-head architecture allows different semantic perspectives.

    ENABLE_VENN_MULTIHEAD: bool = True          # Enable multi-head Venn (recommended)
    NUM_VENN_HEADS: int = 12                    # Number of Venn heads (matches NUM_HEADS)
    VENN_CLUSTERS_PER_HEAD: int = 256           # Clusters per head
    VENN_CLUSTERS: int = 256                    # Legacy: total clusters if single-head

    # Venn Activation & Propagation
    VENN_PROPAGATION_FACTOR: float = 0.2        # Activation spread to nearby clusters (0.0-1.0)
    VENN_INTERSECTION_THRESHOLD: float = 0.3    # Threshold for cluster connectivity (0.0-1.0)
    MAX_PROPAGATED_ACTIVATION: float = 5.0      # Maximum activation after propagation
    VENN_ACTIVATION_TEMPERATURE: float = 1.0    # Gaussian activation temperature (higher = broader)

    # Venn Membership Weights
    PRIMARY_MEMBERSHIP_WEIGHT: float = 0.6      # Weight for primary (nearest) cluster
    SECONDARY_MEMBERSHIP_WEIGHT: float = 0.4    # Weight for secondary cluster

    # Venn Cluster Updates
    CLUSTER_UPDATE_LR: float = 0.1              # Learning rate for cluster center updates

    # ========================================
    # Episodic Memory
    # ========================================
    EPISODIC_BUFFER_SIZE: int = 1024            # Size of episodic memory buffer (TUI-controlled)


@dataclass
class VocabOptimizationConfig:
    """
    Vocabulary Optimization Configuration
    ======================================

    Controls the hot/cold vocabulary architecture for efficient GPU/RAM usage.

    Hot/Cold Architecture:
    - Hot vocab (GPU): 10,000 most frequently used words for fast access
    - Cold vocab (RAM): Unlimited size, swapped to hot via LRU eviction
    - Pre-loading: Batch load cold words to hot before forward pass
    - Deferred sync: Batch GPU syncs to avoid contention during training

    Performance Features:
    - Batch operations for multi-word sync (single GPU call)
    - Cached char embeddings (refresh periodically)
    - Numpy batch ops for cold vocab embedding computation
    - AMD Smart Access Memory (SAM) support
    - CUDA Unified Memory (experimental)

    Persistence:
    - Cold vocab saved to disk alongside checkpoints
    - Optional compression (.npz) - trades speed for disk space
    - Auto-load from checkpoint
    """

    # Master switch
    ENABLE_VOCAB_OPTIMIZATION: bool = True  # Enable optimized sync path

    # Caching
    CACHE_CHAR_EMBEDDINGS: bool = True  # Cache char embeddings from GPU
    CHAR_EMBEDDING_CACHE_TTL: int = 1000  # Refresh cache every N syncs

    # Performance
    USE_NUMPY_BATCH_OPS: bool = True  # Use numpy for batch embedding computation
    USE_BATCH_SYNC: bool = True  # Use batch GPU sync (single call for N words)

    # Hot/Cold vocab architecture (SEMPRE ATTIVO ora)
    MAX_HOT_VOCAB: int = 10000  # Maximum words in GPU (hot vocab cache)
    LRU_EVICTION: bool = True  # Use LRU eviction (vs frequency-based)

    # Deferred Sync (evita GPU contention durante training/validation)
    ENABLE_DEFERRED_SYNC: bool = True  # Batch vocab syncs fino a fine pass
    DEFER_DURING_VALIDATION: bool = True  # Defer anche durante validation

    # Pre-loading (carica tokens da cold a hot prima del forward pass)
    ENABLE_TOKEN_PRELOADING: bool = True  # Pre-carica tokens mancanti in hot
    PRELOAD_BATCH_SIZE: int = 100  # Max tokens da pre-caricare in un batch

    # Cold vocab persistence
    SAVE_COLD_VOCAB: bool = True  # Salva cold vocab embeddings su disco
    COLD_VOCAB_COMPRESSION: bool = False  # Usa compressione (.npz) per cold vocab (False = 10-100x più veloce ma file ~3x più grande)
    AUTO_LOAD_COLD: bool = True  # Carica automaticamente cold vocab da checkpoint

    # AMD Smart Access Memory (SAM) - funziona con qualsiasi CPU AMD + GPU
    ENABLE_AMD_SAM: bool = False  # Enable AMD Smart Access Memory (Resizable BAR)

    # AMD Infinity Cache - richiede combo CPU+GPU AMD specifica
    ENABLE_AMD_INFINITY_CACHE: bool = False  # Enable AMD Infinity Cache optimization (solo combo AMD)

    # CUDA Unified Memory (experimental)
    PREFER_UNIFIED_MEMORY: bool = False  # Use CUDA unified memory quando disponibile


@dataclass
class TrainingConfig:
    """
    Training Configuration
    ======================

    Controls all training hyperparameters and update frequencies.

    Learning Rates:
    - BASE_LR: Applied to transformer weights
    - EMBEDDING_LR_SCALE: Embeddings learn slower (10% of base)
    - OUTPUT_LR_SCALE: Output projection at full speed

    Validation:
    - Supports both per-pass and periodic validation
    - Early stopping with patience
    - Configurable train/val split

    Update Frequencies:
    - VENN_UPDATE_FREQUENCY: How often to update semantic clusters
    - STATS_SYNC_FREQUENCY: How often to sync GPU stats to CPU

    Presets can modify these parameters (but never architecture params).
    """

    # Learning rates
    BASE_LR: float = 0.0001  # Conservative per training continuo
    EMBEDDING_LR_SCALE: float = 0.1  # Char/Word embeddings al 10% del LR base (più lento)
    OUTPUT_LR_SCALE: float = 1.0  # Output weights al 100% del LR base (normale)

    # Momentum
    MOMENTUM: float = 0.9  # High stability (0.0-1.0)

    # Temperature
    EXPLORATION_TEMPERATURE: float = 1.0  # Temperature per sampling durante self-loop

    # Frequenze di update
    VENN_UPDATE_FREQUENCY: int = 100  # Aggiorna clusters ogni N cicli
    STATS_SYNC_FREQUENCY: int = 10  # Sync GPU stats ogni N cicli

    # Validation settings
    VALIDATION_SPLIT: float = 0.1  # 10% dei dati per validation
    VALIDATION_FREQUENCY: int = 100  # Valida ogni N batches/samples (se validate_per_pass=False)
    VALIDATE_PER_PASS: bool = True  # True = valida a fine pass, False = ogni N samples
    EARLY_STOPPING_PATIENCE: int = 5  # Stop dopo N validations senza improvement
    MIN_VALIDATION_SAMPLES: int = 10  # Minimo samples per validation

    # Checkpoint settings
    AUTO_SAVE_FREQUENCY: int = 1000  # Auto-save ogni N articles/samples

    # Generic training settings (used by Wikipedia, Dataset, and all data sources)
    BATCH_SIZE: int = 100  # Items (articles/files) per batch
    MIN_TEXT_LENGTH: int = 500  # Minimum text length in chars (for filtering)
    API_BATCH_SIZE: int = 10  # Items per API request (for Wikipedia/future API sources)


@dataclass
class PerformanceConfig:
    """
    Performance & GPU Optimization Configuration
    ============================================

    Controls CUDA optimizations, memory management, and compute pipeline.

    cuBLAS Optimizations:
    - Matrix multiplications offloaded to cuBLAS (NVIDIA tuned)
    - Backward pass gradient computation via cuBLAS

    Fused Kernels:
    - Attention + FFN fusion (reduces kernel launches)
    - Embedding + LayerNorm fusion

    Pipeline Modes:
    - disabled: Sequential execution (lowest memory, lowest throughput)
    - double: Overlap compute with H2D/D2H (balanced)
    - triple: Maximum parallelism (highest throughput, highest memory)

    Memory Management:
    - Pinned memory for faster CPU<->GPU transfers
    - Pre-allocated buffers (avoid runtime malloc overhead)
    - Warp primitives for efficient GPU reductions

    Target GPU utilization: 80-95% (configurable)
    """

    # cuBLAS optimizations
    USE_CUBLAS: bool = True  # Use cuBLAS for matrix operations
    USE_CUBLAS_BACKWARD: bool = True  # Use cuBLAS for gradient computation

    # Fused kernels
    USE_FUSED_KERNELS: bool = True  # Fused attention+FFN
    USE_FUSED_EMBEDDING: bool = True  # Fused embedding+layernorm

    # Pipeline settings
    PIPELINE_MODE: str = "double"  # "disabled", "double", "triple"
    ASYNC_TRANSFERS: bool = True  # Async H2D/D2H transfers

    # CPU worker settings
    NUM_CPU_WORKERS: int = 1  # Number of CPU workers for preprocessing (1 = single thread)
    PREFETCH_SIZE: int = 3  # Number of batches to prefetch

    # Stream configuration
    NUM_COMPUTE_STREAMS: int = 1  # Number of compute streams
    OVERLAP_H2D_COMPUTE: bool = True  # Overlap transfers with compute

    # Memory optimization
    USE_PINNED_MEMORY: bool = True  # Use pinned host memory for faster transfers
    PREALLOCATE_BUFFERS: bool = True  # Pre-allocate all buffers at init
    PINNED_BUFFER_MIN_WORDS: int = 1000  # Minimum pinned buffer size for vocab sync

    # Kernel tuning
    BLOCK_SIZE: int = 256  # Default CUDA block size
    USE_WARP_PRIMITIVES: bool = True  # Use warp-level shuffle operations

    # Target utilization
    GPU_UTILIZATION_TARGET: int = 80  # Target GPU utilization %

    # Stats collection
    STATS_WINDOW_SIZE: int = 100  # Rolling window for stats averaging
    STATS_HISTORY_SIZE: int = 1000  # Number of historical stats to keep


@dataclass
class GenerationConfig:
    """
    Text Generation Configuration
    ==============================

    Controls text generation behavior with confidence-based stopping.

    Continuation Bias:
    A.D.A.M uses continuation bias instead of traditional length limits.
    The model stops generating when token confidence drops below threshold.

    Key Parameters:
    - MIN_TOKEN_CONFIDENCE: Stop when token probability < this value
    - CONFIDENCE_DECAY: Exponential moving average for confidence tracking
    - LOW_CONFIDENCE_STREAK: Stop after N consecutive low-confidence tokens

    This approach produces more natural, contextually-appropriate responses
    compared to fixed max_tokens limits.

    Temperature:
    - Lower (0.5-0.7): Focused, deterministic outputs
    - Medium (0.8-1.2): Balanced creativity
    - Higher (1.5-2.0): Exploratory, diverse outputs
    """

    # Temperature per sampling
    TEMPERATURE: float = 1.0

    # Continuation bias - stop quando confidenza scende
    MIN_TOKEN_CONFIDENCE: float = 0.05  # Soglia minima probabilità token
    CONFIDENCE_DECAY: float = 0.9  # Decay per media mobile confidenza
    LOW_CONFIDENCE_STREAK: int = 3  # Stop dopo N token consecutivi a bassa confidenza

    # Limiti generazione
    MAX_TOKENS: int = 256  # Massimo token per risposta
    MIN_TOKENS: int = 5  # Minimo token (ignora confidence per primi N)

    # Stop tokens
    STOP_ON_NEWLINE: bool = True  # Stop su doppio newline
    STOP_ON_PERIOD: bool = False  # Stop su punto (per risposte brevi)


@dataclass
class CheckpointConfig:
    """Configurazione checkpoint system"""

    CHECKPOINT_DIR: Path = Path("checkpoints")
    AUTO_CHECKPOINT_INTERVAL: int = 100  # Salva ogni N esempi
    CHECKPOINT_VERSION: int = 3  # V3 con vocab dinamico
    MAGIC_V3: str = "VECTLLM3"
    MAGIC_V2: str = "VECTLLM2"
    MAGIC_V1: str = "VECTLLM"


@dataclass
class RuntimeConfig:
    """Configurazione runtime"""
    
    # GPU
    DEVICE_ID: int = 0
    
    # Compilation
    NVCC_ARCH: str = "auto"  # "auto" o "sm_XX"
    
    NVCC_FLAGS: list = None
    
    def __post_init__(self):
        if self.NVCC_FLAGS is None:
            self.NVCC_FLAGS = [
                '-O3',
                '--shared',
                '-Xcompiler', '-fPIC',
                '--use_fast_math',
                '-lcublas', '-lcurand', '-lpthread'
            ]
    
    # Cache
    CACHE_DIR: Path = Path.home() / ".cache" / "vectllm"


# Config presets
def get_config_preset(preset_name: str = "default"):
    """Ottieni preset di configurazione"""

    presets = {
        "none": {
            # Empty preset - doesn't modify anything (keeps TUI settings)
            "training": TrainingConfig(
                # No changes - will use TUI settings
            ),
            "model": ModelConfig(
                # No changes - will use TUI settings
            ),
            "performance": PerformanceConfig(
                # No changes - will use TUI settings
            ),
            "vocab_optimization": VocabOptimizationConfig(
                # No changes - will use TUI settings
            ),
        },

        "default": {
            "training": TrainingConfig(),
            "model": ModelConfig(),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        "fast_learning": {
            "training": TrainingConfig(
                BASE_LR=0.001,
                MOMENTUM=0.7,
                EXPLORATION_TEMPERATURE=1.5,
                VENN_UPDATE_FREQUENCY=50,
                BATCH_SIZE=200,  # Larger batches for faster learning
                AUTO_SAVE_FREQUENCY=500,  # More frequent saves
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
                # Only Venn and vocabulary parameters
                WORD_CREATION_THRESHOLD=3,  # Accept words faster
                VENN_PROPAGATION_FACTOR=0.2,
                VENN_INTERSECTION_THRESHOLD=0.3,
            ),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        "stable": {
            "training": TrainingConfig(
                BASE_LR=0.00001,
                MOMENTUM=0.95,
                EXPLORATION_TEMPERATURE=0.7,
                VENN_UPDATE_FREQUENCY=200
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
                # Only Venn and vocabulary parameters
                WORD_PRUNING_THRESHOLD=2,  # In stable mode, prune rarely used words
                VENN_PROPAGATION_FACTOR=0.15,  # Lower propagation for stability
                VENN_INTERSECTION_THRESHOLD=0.4,
            ),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        "inference": {
            "training": TrainingConfig(
                BASE_LR=0.0,
                MOMENTUM=0.0,
                EXPLORATION_TEMPERATURE=0.5,
                VENN_UPDATE_FREQUENCY=0,  # No Venn updates during inference
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
                # Only Venn and vocabulary parameters
                WORD_CREATION_THRESHOLD=0,  # No new words during inference
            ),
            "performance": PerformanceConfig(
                USE_FUSED_KERNELS=True,
                PIPELINE_MODE="disabled",  # No training pipeline needed
            ),
            "vocab_optimization": VocabOptimizationConfig(
                ENABLE_DEFERRED_SYNC=False,  # No training syncs in inference
                SAVE_COLD_VOCAB=False,  # Don't save during inference
            ),
        },

        "research": {
            "training": TrainingConfig(
                BASE_LR=0.0005,
                MOMENTUM=0.8,
                EXPLORATION_TEMPERATURE=2.0,
                VENN_UPDATE_FREQUENCY=25,
                AUTO_SAVE_FREQUENCY=100,  # Frequent checkpoints for experiments
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
                # Only Venn and vocabulary parameters
                WORD_CREATION_THRESHOLD=2,  # More aggressive word creation
                VENN_PROPAGATION_FACTOR=0.3,  # Higher semantic propagation
                VENN_INTERSECTION_THRESHOLD=0.25,
                VENN_ACTIVATION_TEMPERATURE=1.2,
            ),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        # High performance preset
        "high_performance": {
            "training": TrainingConfig(
                BASE_LR=0.0001,
                MOMENTUM=0.9,
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
            ),
            "performance": PerformanceConfig(
                USE_CUBLAS=True,
                USE_CUBLAS_BACKWARD=True,
                USE_FUSED_KERNELS=True,
                USE_FUSED_EMBEDDING=True,
                PIPELINE_MODE="double",
                ASYNC_TRANSFERS=True,
                USE_PINNED_MEMORY=True,
                USE_WARP_PRIMITIVES=True,
                GPU_UTILIZATION_TARGET=90,
            ),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        # Memory efficient preset
        "memory_efficient": {
            "training": TrainingConfig(
                BASE_LR=0.0001,
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
            ),
            "performance": PerformanceConfig(
                USE_CUBLAS=True,
                USE_FUSED_KERNELS=True,  # Reduces intermediate memory
                PIPELINE_MODE="disabled",  # No extra buffers
                ASYNC_TRANSFERS=False,
                USE_PINNED_MEMORY=False,
                PREALLOCATE_BUFFERS=False,
            ),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        # Maximum throughput
        "max_throughput": {
            "training": TrainingConfig(
                BASE_LR=0.0001,
                VENN_UPDATE_FREQUENCY=200,  # Less frequent updates
            ),
            "model": ModelConfig(
                # Architecture NOT modified by presets (use TUI settings)
            ),
            "performance": PerformanceConfig(
                USE_CUBLAS=True,
                USE_CUBLAS_BACKWARD=True,
                USE_FUSED_KERNELS=True,
                USE_FUSED_EMBEDDING=True,
                PIPELINE_MODE="triple",  # Maximum overlap
                ASYNC_TRANSFERS=True,
                NUM_COMPUTE_STREAMS=2,
                USE_PINNED_MEMORY=True,
                USE_WARP_PRIMITIVES=True,
                GPU_UTILIZATION_TARGET=95,
            ),
            "vocab_optimization": VocabOptimizationConfig(),
        },
    }

    return presets.get(preset_name, presets["default"])


# Configurazione globale
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
PERFORMANCE_CONFIG = PerformanceConfig()
GENERATION_CONFIG = GenerationConfig()
CHECKPOINT_CONFIG = CheckpointConfig()
RUNTIME_CONFIG = RuntimeConfig()
VOCAB_OPTIMIZATION_CONFIG = VocabOptimizationConfig()


def set_config_from_preset(preset_name: str, override_user_settings: bool = False):
    """
    Imposta configurazione da preset.

    Args:
        preset_name: Nome del preset da applicare
        override_user_settings: Se False (default), preserva le impostazioni utente correnti.
                                Se True, sovrascrive completamente con il preset.

    IMPORTANT: By default, this function MERGES preset values with existing config.
    This prevents presets from overriding user-configured settings.
    """
    global MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_CONFIG, VOCAB_OPTIMIZATION_CONFIG

    preset = get_config_preset(preset_name)

    if override_user_settings:
        # Complete override - use preset values only
        MODEL_CONFIG = preset["model"]
        TRAINING_CONFIG = preset["training"]
        PERFORMANCE_CONFIG = preset["performance"]
        if "vocab_optimization" in preset:
            VOCAB_OPTIMIZATION_CONFIG = preset["vocab_optimization"]
    else:
        # Merge mode - update only the fields that differ from defaults
        # This preserves user-configured settings
        _merge_config(MODEL_CONFIG, preset["model"])
        _merge_config(TRAINING_CONFIG, preset["training"])
        _merge_config(PERFORMANCE_CONFIG, preset["performance"])
        if "vocab_optimization" in preset:
            _merge_config(VOCAB_OPTIMIZATION_CONFIG, preset["vocab_optimization"])


def _merge_config(target_config, preset_config):
    """
    Merge preset config into target config, updating only non-default values.
    Preserves user customizations.

    IMPORTANT: Architecture parameters are NEVER overwritten by presets.
    Architecture is controlled exclusively by TUI settings.
    """
    # Architecture parameters that presets should NEVER modify
    ARCHITECTURE_PARAMS = {
        'NUM_LAYERS', 'EMBED_DIM', 'NUM_HEADS', 'MAX_SEQ_LEN', 'EPISODIC_BUFFER_SIZE'
    }

    for key in preset_config.__dict__:
        # Skip architecture parameters - these are TUI-only
        if key in ARCHITECTURE_PARAMS:
            continue

        if hasattr(target_config, key):
            setattr(target_config, key, getattr(preset_config, key))


def load_tui_settings():
    """
    Load settings from TUI file and apply to global config objects.
    This ensures the model uses the settings configured in the TUI.

    Returns:
        bool: True if settings were loaded, False otherwise
    """
    import json
    from pathlib import Path

    global MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_CONFIG, GENERATION_CONFIG, VOCAB_OPTIMIZATION_CONFIG, RUNTIME_CONFIG

    tui_settings_file = Path.home() / ".adam" / "tui_settings.json"

    if not tui_settings_file.exists():
        return False

    try:
        with open(tui_settings_file, 'r') as f:
            data = json.load(f)

        # Extract settings (handle both old and new format)
        settings = data.get('settings', data)

        if not settings:
            return False

        loaded_count = 0

        def apply_setting(key: str, value):
            """Apply a setting to the correct config object (searches all configs)"""
            nonlocal loaded_count
            key_upper = key.upper()

            # Try all config objects (order matters: most likely first)
            for config_obj in [MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_CONFIG,
                              GENERATION_CONFIG, VOCAB_OPTIMIZATION_CONFIG, RUNTIME_CONFIG]:
                if hasattr(config_obj, key_upper):
                    setattr(config_obj, key_upper, value)
                    loaded_count += 1
                    return True
                elif hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    loaded_count += 1
                    return True

            return False

        # Apply ARCHITECTURE settings
        if 'architecture' in settings:
            for key, value in settings['architecture'].items():
                apply_setting(key, value)

        # Apply VENN settings (note: venn_update_frequency is in TRAINING_CONFIG)
        if 'venn' in settings:
            for key, value in settings['venn'].items():
                apply_setting(key, value)

        # Apply VOCABULARY settings
        if 'vocabulary' in settings:
            for key, value in settings['vocabulary'].items():
                apply_setting(key, value)

        # Legacy: Apply MODEL settings (old format compatibility)
        if 'model' in settings:
            for key, value in settings['model'].items():
                apply_setting(key, value)

        # Apply TRAINING settings
        if 'training' in settings:
            for key, value in settings['training'].items():
                apply_setting(key, value)

        # Apply GENERATION settings
        if 'generation' in settings:
            for key, value in settings['generation'].items():
                apply_setting(key, value)

        # Apply PERFORMANCE settings
        if 'performance' in settings:
            for key, value in settings['performance'].items():
                apply_setting(key, value)

        # Apply VOCAB_OPTIMIZATION settings
        if 'vocab_optimization' in settings:
            for key, value in settings['vocab_optimization'].items():
                apply_setting(key, value)

        if loaded_count > 0:
            print(f"✓ Loaded {loaded_count} settings from TUI config")
            return True

        return False

    except Exception as e:
        print(f"⚠ Could not load TUI settings: {e}")
        return False


def update_config(**kwargs):
    """Aggiorna parametri runtime"""
    global TRAINING_CONFIG

    for key, value in kwargs.items():
        if hasattr(TRAINING_CONFIG, key):
            setattr(TRAINING_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")


if __name__ == '__main__':
    print("=== VectLLM Configuration ===\n")
    
    print("MODEL:")
    for k, v in MODEL_CONFIG.__dict__.items():
        print(f"  {k}: {v}")
    
    print("\nTRAINING:")
    for k, v in TRAINING_CONFIG.__dict__.items():
        print(f"  {k}: {v}")
    
    print("\nPresets:", list(get_config_preset.__doc__))
