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
    MAX_WORD_VOCAB_SIZE: int = 100000  # Massimo numero di word tokens
    WORD_CREATION_THRESHOLD: int = 5  # Crea word token dopo N occorrenze
    WORD_PRUNING_THRESHOLD: int = 2  # Rimuovi word se freq < N
    MAX_WORD_LENGTH: int = 20  # Massima lunghezza parola in char
    
    # Venn Semantic System
    VENN_CLUSTERS: int = 256
    INTERSECTION_THRESHOLD: float = 0.5  # Threshold per connessioni cluster
    CLUSTER_UPDATE_LR: float = 0.1  # Learning rate per aggiornamento centri cluster
    
    # Episodic Memory
    EPISODIC_BUFFER_SIZE: int = 1024


@dataclass
class TrainingConfig:
    """Configurazione training"""

    # Learning rates
    BASE_LR: float = 0.0001  # Conservative per training continuo
    EMBEDDING_LR_SCALE: float = 0.1  # Embeddings al 10% del LR base

    # Momentum
    MOMENTUM: float = 0.9  # High stability

    # Temperature
    EXPLORATION_TEMPERATURE: float = 1.0

    # Frequenze di update
    VENN_UPDATE_FREQUENCY: int = 100  # Aggiorna clusters ogni N cicli
    STATS_SYNC_FREQUENCY: int = 10  # Sync GPU stats ogni N cicli
    VOCAB_PRUNING_FREQUENCY: int = 10000  # Pruning vocabolario ogni N cicli

    # Sleep timings (microseconds)
    SELF_LOOP_SLEEP_US: int = 5000  # 5ms → ~200 cycles/sec
    EXTERNAL_INPUT_SLEEP_US: int = 100000  # 100ms


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
        },

        # NEW: High performance preset
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
        },

        # NEW: Memory efficient preset
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
        },

        # NEW: Maximum throughput
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


def set_config_from_preset(preset_name: str):
    """Imposta configurazione da preset"""
    global MODEL_CONFIG, TRAINING_CONFIG
    
    preset = get_config_preset(preset_name)
    MODEL_CONFIG = preset["model"]
    TRAINING_CONFIG = preset["training"]


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
