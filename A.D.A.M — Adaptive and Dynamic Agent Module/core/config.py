#!/usr/bin/env python3
"""
VectLLM Configuration
Tutti i parametri modificabili del modello
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Configurazione architettura del modello"""

    # Dimensioni modello
    EMBED_DIM: int = 768
    NUM_HEADS: int = 12
    NUM_LAYERS: int = 6
    MAX_SEQ_LEN: int = 512

    # Vocabolario dinamico
    CHAR_VOCAB_SIZE: int = 256  # ASCII/UTF-8 base
    WORD_CREATION_THRESHOLD: int = 5  # Crea word token dopo N occorrenze (evita spam)
    WORD_PRUNING_THRESHOLD: int = 2  # Rimuovi word se freq < N (cleanup cold vocab)
    MAX_WORD_LENGTH: int = 20  # Massima lunghezza parola in char (sanity check)

    # Venn Semantic System
    VENN_CLUSTERS: int = 256

    # Venn Activation & Propagation
    VENN_PROPAGATION_FACTOR: float = 0.2  # Quanto le attivazioni si propagano tra cluster vicini (0.0-1.0)
    VENN_INTERSECTION_THRESHOLD: float = 0.3  # Threshold per considerare cluster "connessi" (0.0-1.0)
    MAX_PROPAGATED_ACTIVATION: float = 5.0  # Cap massimo per attivazioni propagate
    VENN_ACTIVATION_TEMPERATURE: float = 1.0  # Temperature per Gaussian activation (più alto = più broad)

    # Venn Membership Weights
    PRIMARY_MEMBERSHIP_WEIGHT: float = 0.6  # Peso per cluster primario (più vicino)
    SECONDARY_MEMBERSHIP_WEIGHT: float = 0.4  # Peso per cluster secondario

    # Venn Cluster Updates
    CLUSTER_UPDATE_LR: float = 0.1  # Learning rate per aggiornamento centri cluster

    # Episodic Memory
    EPISODIC_BUFFER_SIZE: int = 1024


@dataclass
class VocabOptimizationConfig:
    """Configurazione ottimizzazione vocabolario CPU/GPU hybrid"""

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
    COLD_VOCAB_COMPRESSION: bool = True  # Usa compressione (.npz) per cold vocab
    AUTO_LOAD_COLD: bool = True  # Carica automaticamente cold vocab da checkpoint

    # AMD Smart Access Memory (SAM) - funziona con qualsiasi CPU AMD + GPU
    ENABLE_AMD_SAM: bool = False  # Enable AMD Smart Access Memory (Resizable BAR)

    # AMD Infinity Cache - richiede combo CPU+GPU AMD specifica
    ENABLE_AMD_INFINITY_CACHE: bool = False  # Enable AMD Infinity Cache optimization (solo combo AMD)

    # CUDA Unified Memory (experimental)
    PREFER_UNIFIED_MEMORY: bool = False  # Use CUDA unified memory quando disponibile


@dataclass
class TrainingConfig:
    """Configurazione training"""

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


@dataclass
class PerformanceConfig:
    """Configurazione ottimizzazioni GPU (Tier 1 & 2)"""

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

    # Kernel tuning
    BLOCK_SIZE: int = 256  # Default CUDA block size
    USE_WARP_PRIMITIVES: bool = True  # Use warp-level shuffle operations

    # Target utilization
    GPU_UTILIZATION_TARGET: int = 80  # Target GPU utilization %


@dataclass
class GenerationConfig:
    """Configurazione generazione testo con continuation bias"""

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
                VENN_UPDATE_FREQUENCY=50
            ),
            "model": ModelConfig(),
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
            "model": ModelConfig(),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(
                ENABLE_VOCAB_OPTIMIZATION=False,  # No dynamic expansion in stable
            ),
        },

        "inference": {
            "training": TrainingConfig(
                BASE_LR=0.0,
                MOMENTUM=0.0,
                EXPLORATION_TEMPERATURE=0.5,
            ),
            "model": ModelConfig(),
            "performance": PerformanceConfig(
                USE_FUSED_KERNELS=True,
                PIPELINE_MODE="disabled",  # No training pipeline needed
            ),
            "vocab_optimization": VocabOptimizationConfig(
                ENABLE_VOCAB_OPTIMIZATION=False,  # No expansion during inference
            ),
        },

        "research": {
            "training": TrainingConfig(
                BASE_LR=0.0005,
                MOMENTUM=0.8,
                EXPLORATION_TEMPERATURE=2.0,
                VENN_UPDATE_FREQUENCY=25
            ),
            "model": ModelConfig(),
            "performance": PerformanceConfig(),
            "vocab_optimization": VocabOptimizationConfig(),
        },

        # High performance preset
        "high_performance": {
            "training": TrainingConfig(
                BASE_LR=0.0001,
                MOMENTUM=0.9,
            ),
            "model": ModelConfig(),
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
            "model": ModelConfig(),
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
            "model": ModelConfig(),
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


def set_config_from_preset(preset_name: str):
    """Imposta configurazione da preset"""
    global MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_CONFIG, VOCAB_OPTIMIZATION_CONFIG

    preset = get_config_preset(preset_name)
    MODEL_CONFIG = preset["model"]
    TRAINING_CONFIG = preset["training"]
    PERFORMANCE_CONFIG = preset["performance"]
    if "vocab_optimization" in preset:
        VOCAB_OPTIMIZATION_CONFIG = preset["vocab_optimization"]


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
