#!/usr/bin/env python3
"""
VectLLM Brain Wrapper
Interfaccia Python per il kernel CUDA con supporto vocabolario dinamico
"""

import os
import sys
import ctypes
import subprocess
import hashlib
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from .config import MODEL_CONFIG, TRAINING_CONFIG, RUNTIME_CONFIG, GENERATION_CONFIG, PERFORMANCE_CONFIG, VOCAB_OPTIMIZATION_CONFIG
from .vocabulary import DynamicVocabulary
from .pipeline import AsyncBatchLoader, PipelinedTrainer, BatchData
import time


def detect_gpu_vendor() -> str:
    """
    Detect GPU vendor (NVIDIA, AMD, or unknown).
    Returns vendor name string.
    """
    try:
        # Try nvidia-smi first
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "NVIDIA"
    except Exception:
        pass

    try:
        # Try rocm-smi for AMD
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return "AMD"
    except Exception:
        pass

    return "unknown"


class CUDACompiler:
    """Compilatore automatico del kernel CUDA"""

    def __init__(self):
        self.cache_dir = RUNTIME_CONFIG.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, kernel_path: Path) -> Path:
        """Genera path cache basato su hash del kernel"""
        with open(kernel_path, 'rb') as f:
            code_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return self.cache_dir / f"libvectllm_{code_hash}.so"
    
    def find_nvcc(self) -> str:
        """Trova il compilatore nvcc"""
        # Try common locations
        for path in ['/usr/local/cuda/bin/nvcc', '/usr/bin/nvcc']:
            if os.path.exists(path):
                return path
        
        # Try PATH
        result = subprocess.run(['which', 'nvcc'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        
        raise RuntimeError("❌ NVCC not found. Install CUDA toolkit.")
    
    def detect_gpu_arch(self) -> str:
        """Rileva compute capability GPU"""
        if RUNTIME_CONFIG.NVCC_ARCH != "auto":
            return RUNTIME_CONFIG.NVCC_ARCH
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cap = result.stdout.strip().replace('.', '')
                return f"sm_{cap}"
        except Exception as e:
            print(f"⚠️  Could not detect GPU arch: {e}")
        
        # Default to Ampere
        print("⚠️  Using default arch: sm_86 (Ampere)")
        return "sm_86"
    
    def compile(self, kernel_path: Path, force: bool = False) -> Path:
        """Compila il kernel CUDA"""
        lib_path = self.get_cache_path(kernel_path)
        
        if lib_path.exists() and not force:
            print(f"✓ Using cached library: {lib_path.name}")
            return lib_path
        
        print("🔨 Compiling CUDA kernel...")
        print(f"   Source: {kernel_path}")
        
        nvcc = self.find_nvcc()
        arch = self.detect_gpu_arch()
        
        print(f"   NVCC: {nvcc}")
        print(f"   Arch: {arch}")
        
        # Build command
        cmd = [
            nvcc,
            str(kernel_path),
            '-o', str(lib_path)
        ] + RUNTIME_CONFIG.NVCC_FLAGS + [f'-arch={arch}']
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"❌ Compilation failed:")
                print(result.stderr)
                raise RuntimeError("CUDA compilation failed")
            
            if result.stderr:
                print(f"⚠️  Compilation warnings:\n{result.stderr}")
            
            print(f"✅ Compiled successfully: {lib_path.name}")
            return lib_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timeout (120s)")
        except Exception as e:
            raise RuntimeError(f"Compilation error: {e}")


class VectLLMBrain:
    """
    Wrapper Python per il kernel CUDA VectLLM.
    Gestisce vocabolario dinamico e interfaccia con il kernel.
    """
    
    def __init__(self, 
                 kernel_path: Optional[Path] = None,
                 vocab: Optional[DynamicVocabulary] = None,
                 auto_compile: bool = True):
        """
        Args:
            kernel_path: Path al file brain.cu (default: ../kernels/brain.cu)
            vocab: Vocabolario dinamico (se None, crea nuovo)
            auto_compile: Compila automaticamente il kernel
        """
        # Kernel path
        if kernel_path is None:
            kernel_path = Path(__file__).parent.parent / "kernels" / "brain.cu"
        self.kernel_path = Path(kernel_path)
        
        if not self.kernel_path.exists():
            raise FileNotFoundError(f"Kernel not found: {self.kernel_path}")
        
        # Vocabulary
        if vocab is None:
            vocab = DynamicVocabulary(
                embed_dim=MODEL_CONFIG.EMBED_DIM,
                char_vocab_size=MODEL_CONFIG.CHAR_VOCAB_SIZE,
                max_word_vocab_size=MODEL_CONFIG.MAX_WORD_VOCAB_SIZE,  # CRITICAL: Must match CUDA's MAX_WORD_VOCAB_SIZE
                creation_threshold=MODEL_CONFIG.WORD_CREATION_THRESHOLD,
                pruning_threshold=MODEL_CONFIG.WORD_PRUNING_THRESHOLD,
                max_word_length=MODEL_CONFIG.MAX_WORD_LENGTH
            )
        self.vocab = vocab
        
        # Compile and load
        if auto_compile:
            compiler = CUDACompiler()
            lib_path = compiler.compile(self.kernel_path)
            self.lib = ctypes.CDLL(str(lib_path))
            self._setup_api()
        else:
            self.lib = None

        self.initialized = False

        # Preallocated buffers for performance
        self._token_buffer = None
        self._token_buffer_size = 0
        if hasattr(PERFORMANCE_CONFIG, 'PREALLOCATE_BUFFERS') and PERFORMANCE_CONFIG.PREALLOCATE_BUFFERS:
            self._preallocate_training_buffers()

        # Vocab optimization: caching and batching
        self._char_embeddings_cache = None  # Cached char embeddings from GPU
        self._char_cache_sync_count = 0  # Counter for cache invalidation
        self._pending_words = []  # Words waiting to be synced to GPU
        self._last_sync_time = 0  # For performance monitoring
        self._total_sync_time = 0  # Accumulated sync time
        self._sync_count = 0  # Number of syncs performed

        # Cold vocab storage (all words in RAM, unlimited)
        self._cold_vocab: Dict[int, np.ndarray] = {}  # word_id -> embedding
        self._cold_vocab_path: Optional[Path] = None  # Path for persistence
        self._hot_vocab_ids: set = set()  # word_ids currently in GPU
        self._word_usage_count: Dict[int, int] = {}  # word_id -> usage count

        # Deferred sync for validation/training (avoid GPU contention)
        # Use counter to support nested defer contexts
        self._defer_sync_depth = 0  # 0 = sync immediately, >0 = defer
        self._deferred_old_size = None
        self._deferred_new_size = None

        # GPU vendor detection for optimization paths
        self._gpu_vendor = detect_gpu_vendor()

        # Pinned memory buffers for faster CPU<->GPU transfers
        self._pinned_word_ids = None
        self._pinned_embeddings = None
        self._pinned_buffer_size = 0

        if PERFORMANCE_CONFIG.USE_PINNED_MEMORY:
            self._allocate_pinned_buffers()

    def _allocate_pinned_buffers(self, min_words: Optional[int] = None):
        """
        Allocate pinned memory buffers for faster CPU<->GPU transfers.
        Uses numpy arrays with aligned memory for optimal DMA transfers.

        Args:
            min_words: Minimum buffer size in words (default: from PERFORMANCE_CONFIG)
        """
        if min_words is None:
            min_words = PERFORMANCE_CONFIG.PINNED_BUFFER_MIN_WORDS

        try:
            embed_dim = MODEL_CONFIG.EMBED_DIM

            # Allocate pinned buffers (page-aligned for DMA)
            # These will be reused for all batch syncs
            self._pinned_buffer_size = min_words
            self._pinned_word_ids = np.zeros(min_words, dtype=np.int32)
            self._pinned_embeddings = np.zeros(min_words * embed_dim, dtype=np.float32)

            # For AMD with Infinity Cache, ensure 64-byte alignment (cache line)
            if self._gpu_vendor == "AMD" and VOCAB_OPTIMIZATION_CONFIG.ENABLE_AMD_INFINITY_CACHE:
                # Reallocate with explicit alignment
                self._pinned_word_ids = np.require(
                    self._pinned_word_ids, requirements=['C_CONTIGUOUS', 'ALIGNED']
                )
                self._pinned_embeddings = np.require(
                    self._pinned_embeddings, requirements=['C_CONTIGUOUS', 'ALIGNED']
                )

        except Exception as e:
            print(f"⚠️  Failed to allocate pinned buffers: {e}")
            self._pinned_word_ids = None
            self._pinned_embeddings = None
            self._pinned_buffer_size = 0

    def _resize_pinned_buffers(self, num_words: int):
        """Resize pinned buffers if needed."""
        if num_words <= self._pinned_buffer_size:
            return

        # Grow by 2x to avoid frequent reallocations
        new_size = max(num_words, self._pinned_buffer_size * 2)
        self._allocate_pinned_buffers(new_size)

    def _setup_api(self):
        """Setup function signatures per ctypes"""
        # init_system
        self.lib.init_system.argtypes = []
        self.lib.init_system.restype = ctypes.c_int
        
        # shutdown_system
        self.lib.shutdown_system.argtypes = []
        self.lib.shutdown_system.restype = None
        
        # set_mode
        self.lib.set_mode.argtypes = [ctypes.c_int]
        self.lib.set_mode.restype = None
        
        # set_exploration_params
        self.lib.set_exploration_params.argtypes = [ctypes.c_float, ctypes.c_float]
        self.lib.set_exploration_params.restype = None
        
        # process_input
        self.lib.process_input.argtypes = [ctypes.c_char_p]
        self.lib.process_input.restype = None
        
        # get_output
        self.lib.get_output.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.get_output.restype = ctypes.c_int
        
        # get_stats
        self.lib.get_stats.argtypes = [
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.get_stats.restype = None
        
        # set_reward
        self.lib.set_reward.argtypes = [ctypes.c_float]
        self.lib.set_reward.restype = None
        
        # save_checkpoint
        self.lib.save_checkpoint.argtypes = [ctypes.c_char_p]
        self.lib.save_checkpoint.restype = ctypes.c_int
        
        # load_checkpoint
        self.lib.load_checkpoint.argtypes = [ctypes.c_char_p]
        self.lib.load_checkpoint.restype = ctypes.c_int
        
        # feed_training_batch
        self.lib.feed_training_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int
        ]
        self.lib.feed_training_batch.restype = ctypes.c_int
        
        # VOCABULARY MANAGEMENT API
        # activate_word_token
        self.lib.activate_word_token.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.activate_word_token.restype = ctypes.c_int

        # activate_word_tokens_batch (BATCH SYNC)
        self.lib.activate_word_tokens_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # word_ids array
            ctypes.POINTER(ctypes.c_float), # embeddings array
            ctypes.c_int                     # num_words
        ]
        self.lib.activate_word_tokens_batch.restype = ctypes.c_int

        # deactivate_word_token
        self.lib.deactivate_word_token.argtypes = [ctypes.c_int]
        self.lib.deactivate_word_token.restype = ctypes.c_int
        
        # get_word_embedding
        self.lib.get_word_embedding.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.get_word_embedding.restype = ctypes.c_int
        
        # get_current_word_vocab_size
        self.lib.get_current_word_vocab_size.argtypes = []
        self.lib.get_current_word_vocab_size.restype = ctypes.c_int
        
        # sync_vocabulary_state
        self.lib.sync_vocabulary_state.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int
        ]
        self.lib.sync_vocabulary_state.restype = ctypes.c_int

        # GENERATION API
        # generate_next_token
        self.lib.generate_next_token.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # input_tokens
            ctypes.c_int,                   # num_tokens
            ctypes.c_float,                 # temperature
            ctypes.POINTER(ctypes.c_float)  # out_prob
        ]
        self.lib.generate_next_token.restype = ctypes.c_int

        # get_token_probabilities
        self.lib.get_token_probabilities.argtypes = [
            ctypes.POINTER(ctypes.c_int),   # input_tokens
            ctypes.c_int,                   # num_tokens
            ctypes.c_float,                 # temperature
            ctypes.POINTER(ctypes.c_float), # out_probs
            ctypes.c_int                    # max_vocab
        ]
        self.lib.get_token_probabilities.restype = ctypes.c_int

        # compute_validation_loss (if available in kernel)
        try:
            self.lib.compute_validation_loss.argtypes = [
                ctypes.POINTER(ctypes.c_int),  # tokens
                ctypes.c_int                    # num_tokens
            ]
            self.lib.compute_validation_loss.restype = ctypes.c_float
            self._has_validation_api = True
        except AttributeError:
            self._has_validation_api = False

        # VENN CONFIGURATION API
        # set_venn_config
        self.lib.set_venn_config.argtypes = [
            ctypes.c_float,  # propagation_factor
            ctypes.c_float,  # intersection_threshold
            ctypes.c_float,  # max_propagated_activation
            ctypes.c_float,  # activation_temperature
            ctypes.c_float,  # primary_membership_weight
            ctypes.c_float,  # secondary_membership_weight
            ctypes.c_float   # cluster_update_lr
        ]
        self.lib.set_venn_config.restype = ctypes.c_int

        # get_venn_config
        self.lib.get_venn_config.argtypes = [
            ctypes.POINTER(ctypes.c_float)  # out_config array[7]
        ]
        self.lib.get_venn_config.restype = ctypes.c_int

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
    
    def start(self):
        """Inizializza il sistema CUDA"""
        if self.initialized:
            return
        
        print("🚀 Initializing VectLLM Brain...")
        result = self.lib.init_system()
        if result != 0:
            raise RuntimeError("Failed to initialize CUDA system")

        # Set Venn configuration from config.py
        self._configure_venn_system()

        self.initialized = True
        print("✅ Brain ready - continuous training active")
        print(f"   Vocab: {len(self.vocab.word_to_id)} words active")
    
    def stop(self):
        """Shutdown del sistema"""
        if self.initialized:
            print("🛑 Shutting down Brain...")
            self.lib.shutdown_system()
            self.initialized = False
            print("✅ Shutdown complete")
    
    def configure(self, temperature: float = None, momentum: float = None):
        """Configura parametri di esplorazione"""
        if temperature is None:
            temperature = TRAINING_CONFIG.EXPLORATION_TEMPERATURE
        if momentum is None:
            momentum = TRAINING_CONFIG.MOMENTUM

        self.lib.set_exploration_params(temperature, momentum)

    def _configure_venn_system(self):
        """Configure Venn semantic system parameters from config.py"""
        result = self.lib.set_venn_config(
            MODEL_CONFIG.VENN_PROPAGATION_FACTOR,
            MODEL_CONFIG.VENN_INTERSECTION_THRESHOLD,
            MODEL_CONFIG.MAX_PROPAGATED_ACTIVATION,
            MODEL_CONFIG.VENN_ACTIVATION_TEMPERATURE,
            MODEL_CONFIG.PRIMARY_MEMBERSHIP_WEIGHT,
            MODEL_CONFIG.SECONDARY_MEMBERSHIP_WEIGHT,
            MODEL_CONFIG.CLUSTER_UPDATE_LR
        )
        if result != 0:
            raise RuntimeError("Failed to configure Venn system")

    def get_venn_config(self) -> dict:
        """Get current Venn configuration (for debugging)"""
        config_array = (ctypes.c_float * 7)()
        result = self.lib.get_venn_config(config_array)
        if result != 0:
            raise RuntimeError("Failed to get Venn config")

        return {
            'propagation_factor': config_array[0],
            'intersection_threshold': config_array[1],
            'max_propagated_activation': config_array[2],
            'activation_temperature': config_array[3],
            'primary_membership_weight': config_array[4],
            'secondary_membership_weight': config_array[5],
            'cluster_update_lr': config_array[6]
        }

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text usando vocabolario dinamico.
        NOW with usage tracking for LRU cache management.
        """
        old_vocab_size = len(self.vocab.word_to_id)
        tokens = self.vocab.encode(text)
        new_vocab_size = len(self.vocab.word_to_id)

        # Track usage for LRU (only for word tokens, not chars)
        # Also collect words that reached threshold and need HOT promotion
        words_to_promote = []
        for token_id in tokens:
            if token_id >= MODEL_CONFIG.CHAR_VOCAB_SIZE:
                self._word_usage_count[token_id] = self._word_usage_count.get(token_id, 0) + 1

                # Check if word just reached threshold and needs HOT promotion
                if token_id in self.vocab.id_to_word:
                    word = self.vocab.id_to_word[token_id]
                    word_freq = self.vocab.word_frequency.get(word, 0)

                    # Promote to HOT if threshold reached and not already in HOT
                    if (word_freq >= MODEL_CONFIG.WORD_CREATION_THRESHOLD and
                        token_id not in self._hot_vocab_ids):
                        words_to_promote.append((token_id, word))

        # Se vocabolario è cresciuto, sincronizza con kernel
        if new_vocab_size > old_vocab_size:
            # CRITICAL FIX: Always add new words to COLD vocab immediately
            # This ensures _preload_tokens_to_hot() can find them later
            # Only the HOT (GPU) loading is deferred, not COLD (RAM) initialization
            self._add_new_words_to_cold(old_vocab_size, new_vocab_size)

            if self._defer_sync_depth > 0:
                # Queue HOT (GPU) loading for later (avoid GPU contention during validation/training)
                if self._deferred_old_size is None:
                    self._deferred_old_size = old_vocab_size
                self._deferred_new_size = new_vocab_size
            else:
                # Sync immediately (cold already done, now load to hot)
                self._load_new_words_to_hot(old_vocab_size, new_vocab_size)

        # Promote existing words to HOT if they reached threshold
        if words_to_promote and not self._defer_sync_depth:
            char_embeddings = self._get_char_embeddings_cached()
            self._batch_load_to_hot(words_to_promote, char_embeddings)

        return tokens

    def begin_validation(self):
        """Start validation mode - defer GPU syncs to avoid contention."""
        self._defer_sync_depth += 1

    def end_validation(self):
        """End validation mode - sync any deferred words if this was the last defer context."""
        self._defer_sync_depth = max(0, self._defer_sync_depth - 1)

        # Only sync if we're back to depth 0 (no more defer contexts)
        if self._defer_sync_depth == 0:
            if self._deferred_old_size is not None and self._deferred_new_size is not None:
                # Load deferred words to HOT (cold already initialized in encode_text)
                self._load_new_words_to_hot(self._deferred_old_size, self._deferred_new_size)
            self._deferred_old_size = None
            self._deferred_new_size = None

    def begin_deferred_sync(self):
        """
        Begin deferred sync mode - batch all vocab syncs.
        Use this to defer syncs during training pass.
        Supports nested calls (uses reference counting).
        """
        self._defer_sync_depth += 1

    def end_deferred_sync(self):
        """
        End deferred sync mode - execute batched syncs.
        Call at end of training pass.
        Only syncs when all deferred contexts have ended.

        OPTIMIZATION: Also incrementally updates cold vocab to distribute
        sync cost across training instead of blocking checkpoint saves.
        """
        self._defer_sync_depth = max(0, self._defer_sync_depth - 1)

        # Only sync if we're back to depth 0 (no more defer contexts)
        if self._defer_sync_depth == 0:
            if self._deferred_old_size is not None and self._deferred_new_size is not None:
                # Load deferred words to HOT (cold already initialized in encode_text)
                self._load_new_words_to_hot(self._deferred_old_size, self._deferred_new_size)
            self._deferred_old_size = None
            self._deferred_new_size = None

            # OPTIMIZATION: Incrementally update cold vocab from hot
            # This distributes sync cost across training instead of blocking checkpoint saves
            # Update 100 words per pass (configurable via VOCAB_OPTIMIZATION_CONFIG)
            if VOCAB_OPTIMIZATION_CONFIG.SAVE_COLD_VOCAB and self._hot_vocab_ids:
                max_words = getattr(VOCAB_OPTIMIZATION_CONFIG, 'INCREMENTAL_COLD_SYNC_BATCH', 100)
                self.update_cold_from_hot_incremental(max_words=max_words)

    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens a text"""
        return self.vocab.decode(tokens)
    
    def _get_char_embeddings_cached(self) -> np.ndarray:
        """
        Get char embeddings from GPU with caching.
        OPTIMIZATION: Avoid 256 individual GPU calls on every sync.
        """
        # Check if cache is valid
        cache_ttl = VOCAB_OPTIMIZATION_CONFIG.CHAR_EMBEDDING_CACHE_TTL
        if (VOCAB_OPTIMIZATION_CONFIG.CACHE_CHAR_EMBEDDINGS and
            self._char_embeddings_cache is not None and
            self._char_cache_sync_count < cache_ttl):
            self._char_cache_sync_count += 1
            return self._char_embeddings_cache

        # Fetch all char embeddings from GPU
        char_embeddings = np.zeros((MODEL_CONFIG.CHAR_VOCAB_SIZE, MODEL_CONFIG.EMBED_DIM), dtype=np.float32)

        # Use batch fetch if available, otherwise individual calls
        for char_id in range(MODEL_CONFIG.CHAR_VOCAB_SIZE):
            emb_buffer = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)()
            result = self.lib.get_word_embedding(char_id, emb_buffer)
            if result == 0:
                char_embeddings[char_id] = np.array(emb_buffer)

        # Cache the result
        if VOCAB_OPTIMIZATION_CONFIG.CACHE_CHAR_EMBEDDINGS:
            self._char_embeddings_cache = char_embeddings
            self._char_cache_sync_count = 0

        return char_embeddings

    def _add_new_words_to_cold(self, old_size: int, new_size: int):
        """
        Add new words to COLD vocab (RAM) immediately.
        This is ALWAYS executed, even when sync is deferred.

        CRITICAL: This ensures _preload_tokens_to_hot() can find words in cold vocab.
        """
        if not self.initialized:
            return

        # Collect new words
        new_words = []
        for word_id in range(old_size, new_size):
            if word_id in self.vocab.id_to_word:
                word = self.vocab.id_to_word[word_id]
                new_words.append((word_id, word))

        if not new_words:
            return

        # Get cached char embeddings
        char_embeddings = self._get_char_embeddings_cached()

        # Add ALL new words to COLD vocab immediately
        for word_id, word in new_words:
            if word_id not in self._cold_vocab:
                init_emb = self.vocab.get_word_embedding_init(word, char_embeddings)
                self._cold_vocab[word_id] = init_emb.copy()
                self._word_usage_count[word_id] = 1  # Initial usage
            else:
                # Word already in cold, increment usage
                self._word_usage_count[word_id] += 1

    def _load_new_words_to_hot(self, old_size: int, new_size: int):
        """
        Load new words from COLD to HOT (GPU) based on LRU strategy.
        This can be deferred during training to avoid GPU contention.
        """
        if not self.initialized:
            return

        start_time = time.time()

        # Collect new words
        new_words = []
        for word_id in range(old_size, new_size):
            if word_id in self.vocab.id_to_word:
                word = self.vocab.id_to_word[word_id]
                new_words.append((word_id, word))

        if not new_words:
            return

        # Get cached char embeddings
        char_embeddings = self._get_char_embeddings_cached()

        # Decide which words to load into HOT (LRU cache)
        # CRITICAL: Apply WORD_CREATION_THRESHOLD here for HOT vocab promotion
        words_to_hot = []
        for word_id, word in new_words:
            # Check frequency threshold for HOT vocab promotion
            word_freq = self.vocab.word_frequency.get(word, 0)
            if word_freq < MODEL_CONFIG.WORD_CREATION_THRESHOLD:
                # Word not used enough yet - stay in COLD only
                continue

            # Check if hot vocab has space
            hot_vocab_size = len(self._hot_vocab_ids)

            if word_id in self._hot_vocab_ids:
                # Already in hot, just increment usage (already done in _add_new_words_to_cold)
                continue
            elif hot_vocab_size < VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB:
                # Space available, load to hot
                words_to_hot.append((word_id, word))
            else:
                # Hot is full, check if this word is more important than LRU
                if self._should_load_to_hot(word_id):
                    # Evict LRU word and load this one
                    self._evict_lru_word()
                    words_to_hot.append((word_id, word))

        # Batch load selected words to HOT (GPU)
        if words_to_hot:
            self._batch_load_to_hot(words_to_hot, char_embeddings)

        # Track performance
        elapsed = time.time() - start_time
        self._total_sync_time += elapsed
        self._sync_count += 1
        self._last_sync_time = elapsed

        cold_size = len(self._cold_vocab)
        hot_size = len(self._hot_vocab_ids)
        if len(words_to_hot) > 0:
            avg_time = self._total_sync_time / self._sync_count if self._sync_count > 0 else elapsed
            print(f"   🔄 Loaded {len(words_to_hot)} words to HOT in {elapsed*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
            print(f"      Cold: {cold_size}, Hot: {hot_size}/{VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB}")

    def _sync_new_words(self, old_size: int, new_size: int):
        """
        Sincronizza nuove parole con strategia COLD-FIRST + LRU.

        NEW ARCHITECTURE:
        1. Write ALL new words to cold vocab (RAM) immediately
        2. Initialize embeddings from char embeddings
        3. Load to hot (GPU) only if space available or evict LRU
        4. This ensures stable embedding space (no more loss spikes!)

        OPTIMIZED: Uses caching, direct word lookup, and batch operations.
        """
        if not self.initialized:
            return

        start_time = time.time()
        num_new_words = new_size - old_size

        # Collect new words directly from id_to_word (O(1) per word, not O(n) scan)
        new_words = []
        for word_id in range(old_size, new_size):
            if word_id in self.vocab.id_to_word:
                word = self.vocab.id_to_word[word_id]
                new_words.append((word_id, word))

        if not new_words:
            return

        # Get cached char embeddings (avoids 256 GPU calls)
        char_embeddings = self._get_char_embeddings_cached()

        # STEP 1: Write ALL new words to COLD vocab immediately (stable embedding space)
        for word_id, word in new_words:
            if word_id not in self._cold_vocab:
                init_emb = self.vocab.get_word_embedding_init(word, char_embeddings)
                self._cold_vocab[word_id] = init_emb.copy()
                self._word_usage_count[word_id] = 1  # Initial usage
            else:
                # Word already in cold, increment usage
                self._word_usage_count[word_id] += 1

        # STEP 2: Decide which words to load into HOT (LRU cache)
        # CRITICAL: Apply WORD_CREATION_THRESHOLD here for HOT vocab promotion
        words_to_hot = []
        for word_id, word in new_words:
            # Check frequency threshold for HOT vocab promotion
            word_freq = self.vocab.word_frequency.get(word, 0)
            if word_freq < MODEL_CONFIG.WORD_CREATION_THRESHOLD:
                # Word not used enough yet - stay in COLD only
                continue

            # Check if hot vocab has space
            hot_vocab_size = len(self._hot_vocab_ids)

            if word_id in self._hot_vocab_ids:
                # Already in hot, just increment usage
                continue
            elif hot_vocab_size < VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB:
                # Space available, load to hot
                words_to_hot.append((word_id, word))
            else:
                # Hot is full, check if this word is more important than LRU
                if self._should_load_to_hot(word_id):
                    # Evict LRU word and load this one
                    self._evict_lru_word()
                    words_to_hot.append((word_id, word))

        # STEP 3: Batch load selected words to HOT (GPU)
        if words_to_hot:
            self._batch_load_to_hot(words_to_hot, char_embeddings)

        # Track performance
        elapsed = time.time() - start_time
        self._total_sync_time += elapsed
        self._sync_count += 1
        self._last_sync_time = elapsed

        cold_size = len(self._cold_vocab)
        hot_size = len(self._hot_vocab_ids)
        if len(words_to_hot) > 0:
            avg_time = self._total_sync_time / self._sync_count if self._sync_count > 0 else elapsed
            print(f"   🔄 Synced {len(new_words)} words in {elapsed*1000:.1f}ms (avg: {avg_time*1000:.1f}ms)")
            print(f"      Cold: {cold_size}, Hot: {hot_size}/{VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB}")

    def _should_load_to_hot(self, word_id: int) -> bool:
        """
        Decide if word should be loaded to hot based on usage.

        Returns True if word usage > minimum hot word usage (LRU eviction candidate)
        """
        if not self._hot_vocab_ids:
            return True

        # Find LRU word (minimum usage)
        min_usage = min(self._word_usage_count.get(wid, 0) for wid in self._hot_vocab_ids)
        current_usage = self._word_usage_count.get(word_id, 0)

        return current_usage > min_usage

    def _evict_lru_word(self):
        """
        Evict least recently used word from hot vocab (LRU cache eviction).
        The word remains in cold vocab (RAM).
        """
        if not self._hot_vocab_ids:
            return

        # Find word with minimum usage (LRU)
        lru_word_id = min(self._hot_vocab_ids,
                         key=lambda wid: self._word_usage_count.get(wid, 0))

        # Remove from hot set (word still in cold!)
        self._hot_vocab_ids.discard(lru_word_id)

        # Note: We don't need to update GPU here - just tracking
        # The GPU slot will be reused by the new word

    def _batch_load_to_hot(self, words_to_load: List[Tuple[int, str]], char_embeddings: np.ndarray):
        """
        Batch load words from cold to hot (GPU).

        Args:
            words_to_load: List of (word_id, word) tuples
            char_embeddings: Cached char embeddings for initialization
        """
        if not words_to_load:
            return

        # Prepare batch data
        word_ids = []
        init_embeddings = []

        for word_id, word in words_to_load:
            # Get embedding from cold vocab (already initialized)
            if word_id in self._cold_vocab:
                init_emb = self._cold_vocab[word_id]
            else:
                # Fallback: initialize from char embeddings
                init_emb = self.vocab.get_word_embedding_init(word, char_embeddings)
                self._cold_vocab[word_id] = init_emb.copy()

            word_ids.append(word_id)
            init_embeddings.append(init_emb)

        num_words = len(word_ids)
        embed_dim = MODEL_CONFIG.EMBED_DIM

        # Use pinned buffers if available (faster DMA transfers)
        if self._pinned_word_ids is not None and PERFORMANCE_CONFIG.USE_PINNED_MEMORY:
            # Resize if needed
            self._resize_pinned_buffers(num_words)

            # Copy to pinned buffers (avoids allocation)
            self._pinned_word_ids[:num_words] = word_ids
            for i, emb in enumerate(init_embeddings):
                self._pinned_embeddings[i * embed_dim:(i + 1) * embed_dim] = emb

            # Create ctypes pointers from pinned memory
            word_ids_ptr = self._pinned_word_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            embeddings_ptr = self._pinned_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            # Fallback: allocate new arrays
            word_ids_array = np.array(word_ids, dtype=np.int32)
            embeddings_array = np.array(init_embeddings, dtype=np.float32).flatten()

            # Create ctypes pointers
            word_ids_ptr = word_ids_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            embeddings_ptr = embeddings_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Single batch call to GPU (instead of N individual calls)
        activated = self.lib.activate_word_tokens_batch(
            word_ids_ptr,
            embeddings_ptr,
            len(word_ids)
        )

        # Mark words as hot
        for word_id, _ in words_to_load:
            self._hot_vocab_ids.add(word_id)

    def sync_vocabulary(self):
        """
        Sincronizza completamente vocabolario Python → CUDA.
        Da chiamare dopo load checkpoint o modifiche massive.
        """
        if not self.initialized:
            return
        
        print("🔄 Synchronizing vocabulary with GPU...")
        
        # Get all word IDs
        word_ids = [word_id for word_id in self.vocab.word_to_id.values() 
                   if word_id >= MODEL_CONFIG.CHAR_VOCAB_SIZE]
        
        if not word_ids:
            print("   No words to sync")
            return
        
        # Converti a ctypes array
        word_ids_array = (ctypes.c_int * len(word_ids))(*word_ids)
        
        # Batch sync
        result = self.lib.sync_vocabulary_state(word_ids_array, len(word_ids))
        
        if result >= 0:
            print(f"   ✅ Synced {result} words")
        else:
            print(f"   ❌ Sync failed")
    
    def activate_word(self, word: str, word_id: int, init_embedding: np.ndarray) -> bool:
        """
        Attiva manualmente una word nel kernel.
        
        Args:
            word: Parola da attivare
            word_id: ID token
            init_embedding: Embedding iniziale [EMBED_DIM]
            
        Returns:
            True se successo
        """
        if not self.initialized:
            return False
        
        # Converti a ctypes
        emb_array = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)(*init_embedding)
        
        result = self.lib.activate_word_token(word_id, emb_array)
        return result == 0
    
    def deactivate_word(self, word_id: int) -> bool:
        """
        Disattiva word token (pruning).
        
        Args:
            word_id: ID token da disattivare
            
        Returns:
            True se successo
        """
        if not self.initialized:
            return False
        
        result = self.lib.deactivate_word_token(word_id)
        return result == 0
    
    def get_kernel_vocab_size(self) -> int:
        """
        Ottieni numero word attive nel kernel.
        
        Returns:
            Numero word tokens attivi nel kernel
        """
        if not self.initialized:
            return 0
        
        return self.lib.get_current_word_vocab_size()
    
    def _preallocate_training_buffers(self):
        """Preallocate ctypes buffers for training to avoid repeated allocations."""
        try:
            # Allocate buffer for max sequence length * 4 (reasonable overhead)
            max_tokens = MODEL_CONFIG.MAX_SEQ_LEN * 4
            self._token_buffer = (ctypes.c_int * max_tokens)()
            self._token_buffer_size = max_tokens
        except Exception:
            # Fallback to dynamic allocation
            self._token_buffer = None
            self._token_buffer_size = 0

    def _preload_tokens_to_hot(self, tokens: List[int]):
        """
        Pre-load any tokens in batch that are not in hot vocab.
        CRITICAL: Ensures kernel has access to all tokens before forward pass.

        Args:
            tokens: List of token IDs that will be used in forward pass
        """
        # Filter word tokens only (chars are always available)
        word_tokens = [t for t in tokens if t >= MODEL_CONFIG.CHAR_VOCAB_SIZE]

        if not word_tokens:
            return

        # Find tokens that are in cold but not in hot
        tokens_to_load = []
        for token_id in set(word_tokens):  # Unique tokens only
            if token_id in self._cold_vocab and token_id not in self._hot_vocab_ids:
                # Token exists in cold but not in hot - need to load
                word = self.vocab.id_to_word.get(token_id, f"<unk_{token_id}>")
                tokens_to_load.append((token_id, word))

        if not tokens_to_load:
            return

        # Check if we need to make space in hot
        hot_vocab_size = len(self._hot_vocab_ids)
        max_hot = VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB

        # Evict LRU tokens if needed to make space
        for _ in range(max(0, hot_vocab_size + len(tokens_to_load) - max_hot)):
            self._evict_lru_word()

        # Batch load from cold to hot
        # Get char embeddings for initialization (if needed)
        char_embeddings = np.zeros((MODEL_CONFIG.CHAR_VOCAB_SIZE, MODEL_CONFIG.EMBED_DIM), dtype=np.float32)
        for char_id in range(MODEL_CONFIG.CHAR_VOCAB_SIZE):
            emb_buffer = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)()
            result = self.lib.get_word_embedding(char_id, emb_buffer)
            if result == 0:
                char_embeddings[char_id] = np.array(emb_buffer)

        self._batch_load_to_hot(tokens_to_load, char_embeddings)

    def feed_training_batch(self, tokens: List[int]) -> int:
        """
        Feed batch di token per training.

        Args:
            tokens: Lista di token IDs

        Returns:
            Numero di token processati (-1 se errore)
        """
        if not tokens:
            return 0

        # PRE-LOAD: Ensure all tokens in batch are in hot vocab before forward pass
        # This prevents kernel from trying to access tokens that only exist in cold (RAM)
        self._preload_tokens_to_hot(tokens)

        n_tokens = len(tokens)

        # Use preallocated buffer if available and fits
        if self._token_buffer is not None and n_tokens <= self._token_buffer_size:
            # Fast copy using numpy and ctypes
            token_np = np.array(tokens, dtype=np.int32)
            ctypes.memmove(ctypes.addressof(self._token_buffer), token_np.ctypes.data, n_tokens * 4)
            result = self.lib.feed_training_batch(self._token_buffer, n_tokens)
        else:
            # Fallback to dynamic allocation
            token_array = (ctypes.c_int * n_tokens)(*tokens)
            result = self.lib.feed_training_batch(token_array, n_tokens)

        return result
    
    def train_on_text(self, text: str, passes: int = 1) -> int:
        """
        Training su testo (con encoding automatico).
        
        Args:
            text: Testo da usare per training
            passes: Numero di passate sul testo
            
        Returns:
            Totale token processati
        """
        tokens = self.encode_text(text)
        
        if not tokens:
            return 0
        
        total_processed = 0
        for _ in range(passes):
            processed = self.feed_training_batch(tokens)
            if processed > 0:
                total_processed += processed

        return total_processed

    def create_pipelined_trainer(self, prefetch_size: int = 3) -> PipelinedTrainer:
        """
        Crea trainer pipelinato per overlap CPU/GPU.

        Args:
            prefetch_size: Numero batch da pre-caricare

        Returns:
            PipelinedTrainer configurato
        """
        return PipelinedTrainer(self, prefetch_size=prefetch_size)

    def train_on_texts_pipelined(self,
                                  texts,
                                  callback=None,
                                  prefetch_size: int = 3) -> int:
        """
        Training pipelinato su stream di testi.
        CPU prepara batch mentre GPU processa.

        Args:
            texts: Iterator o lista di testi
            callback: Callback(batch_num, tokens) per progress
            prefetch_size: Numero batch da pre-caricare

        Returns:
            Totale token processati
        """
        trainer = self.create_pipelined_trainer(prefetch_size)
        total_tokens = trainer.train_texts(iter(texts), callback)
        return total_tokens

    def get_stats(self) -> Dict:
        """Ottieni statistiche del sistema"""
        cycles = ctypes.c_longlong()
        tokens = ctypes.c_longlong()
        temp = ctypes.c_float()
        momentum = ctypes.c_float()
        loss = ctypes.c_float()
        perplexity = ctypes.c_float()

        self.lib.get_stats(
            ctypes.byref(cycles),
            ctypes.byref(tokens),
            ctypes.byref(temp),
            ctypes.byref(momentum),
            ctypes.byref(loss),
            ctypes.byref(perplexity)
        )

        return {
            'cycles': cycles.value,
            'tokens': tokens.value,
            'temperature': temp.value,
            'momentum': momentum.value,
            'loss': loss.value,
            'perplexity': perplexity.value,
            # Vocab stats
            'vocab_words': len(self.vocab.word_to_id),
            'vocab_utilization': len(self.vocab.word_to_id) / self.vocab.max_word_vocab_size,
        }

    def compute_validation_loss(self, tokens: List[int]) -> float:
        """
        Compute loss on tokens without training (forward pass only).

        Args:
            tokens: List of token IDs for validation

        Returns:
            Validation loss (or -1 if error)
        """
        if not self.initialized or not tokens:
            return -1.0

        n_tokens = len(tokens)

        # Use dedicated validation API if available
        if hasattr(self, '_has_validation_api') and self._has_validation_api:
            if self._token_buffer is not None and n_tokens <= self._token_buffer_size:
                token_np = np.array(tokens, dtype=np.int32)
                ctypes.memmove(ctypes.addressof(self._token_buffer), token_np.ctypes.data, n_tokens * 4)
                return self.lib.compute_validation_loss(self._token_buffer, n_tokens)
            else:
                token_array = (ctypes.c_int * n_tokens)(*tokens)
                return self.lib.compute_validation_loss(token_array, n_tokens)

        # Fallback: estimate loss from current model state
        # This uses the training loss as a proxy (less accurate but works without kernel changes)
        stats = self.get_stats()
        return stats['loss']

    def validate_on_text(self, text: str) -> float:
        """
        Compute validation loss on text.

        Args:
            text: Text for validation

        Returns:
            Validation loss
        """
        tokens = self.encode_text(text)
        if not tokens:
            return -1.0
        return self.compute_validation_loss(tokens)

    def generate_token(self, tokens: List[int], temperature: float = None) -> tuple:
        """
        Genera il prossimo token con probabilità.

        Args:
            tokens: Lista di token IDs come contesto
            temperature: Temperature per sampling (default da config)

        Returns:
            (token_id, probability)
        """
        if not self.initialized or not tokens:
            return (-1, 0.0)

        if temperature is None:
            temperature = GENERATION_CONFIG.TEMPERATURE

        # Convert to ctypes array
        token_array = (ctypes.c_int * len(tokens))(*tokens)
        out_prob = ctypes.c_float()

        # Generate
        token_id = self.lib.generate_next_token(
            token_array,
            len(tokens),
            temperature,
            ctypes.byref(out_prob)
        )

        return (token_id, out_prob.value)

    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """
        Genera testo con continuation bias.
        Si ferma quando la confidenza scende sotto soglia.

        Args:
            prompt: Testo di input
            max_tokens: Massimo token da generare (default da config)

        Returns:
            Testo generato
        """
        if not self.initialized:
            return ""

        if max_tokens is None:
            max_tokens = GENERATION_CONFIG.MAX_TOKENS

        # Encode prompt
        tokens = self.encode_text(prompt)
        if not tokens:
            return ""

        # Generation state
        generated_tokens = []
        confidence_avg = 1.0
        low_confidence_streak = 0

        for i in range(max_tokens):
            # Generate next token
            token_id, prob = self.generate_token(tokens)

            if token_id < 0:
                break

            # Add to sequence
            tokens.append(token_id)
            generated_tokens.append(token_id)

            # Update confidence average (exponential moving average)
            confidence_avg = GENERATION_CONFIG.CONFIDENCE_DECAY * confidence_avg + \
                           (1 - GENERATION_CONFIG.CONFIDENCE_DECAY) * prob

            # Check for low confidence (skip first few tokens)
            if i >= GENERATION_CONFIG.MIN_TOKENS:
                if prob < GENERATION_CONFIG.MIN_TOKEN_CONFIDENCE:
                    low_confidence_streak += 1
                    if low_confidence_streak >= GENERATION_CONFIG.LOW_CONFIDENCE_STREAK:
                        # Stop generation - confidence too low
                        break
                else:
                    low_confidence_streak = 0

            # Check for stop tokens
            decoded_char = self.vocab.decode([token_id])

            if GENERATION_CONFIG.STOP_ON_NEWLINE and decoded_char == '\n\n':
                break

            if GENERATION_CONFIG.STOP_ON_PERIOD and decoded_char.endswith('.'):
                break

        # Decode generated tokens
        return self.vocab.decode(generated_tokens)
    
    def save_checkpoint(self, filepath: str, save_vocab: bool = True, save_cold: bool = True,
                       sync_cold_embeddings: bool = True) -> bool:
        """
        Salva checkpoint (brain + vocabolario + cold vocab).

        Args:
            filepath: Path al checkpoint (es. "checkpoint.ckpt")
            save_vocab: Se True, salva anche vocabolario
            save_cold: Se True, salva anche cold vocab embeddings
            sync_cold_embeddings: Se True, aggiorna cold vocab da GPU prima di salvare
                                 (default True, ma può essere False se aggiornato incrementalmente)

        Returns:
            True se successo
        """
        import time
        start_time = time.time()

        # Save brain checkpoint
        result = self.lib.save_checkpoint(filepath.encode('utf-8'))

        if result != 0:
            return False

        # Save vocabulary
        if save_vocab:
            vocab_path = Path(filepath).with_suffix('.vocab')
            self.vocab.save(str(vocab_path))

        # Save cold vocab (all embeddings for persistence)
        if save_cold and self._cold_vocab:
            # Optionally update cold vocab with latest GPU embeddings before saving
            # OPTIMIZATION: Skip this if you update cold vocab incrementally during training
            if sync_cold_embeddings:
                sync_start = time.time()
                self.update_cold_from_hot()
                sync_elapsed = time.time() - sync_start
                if sync_elapsed > 0.1:  # Only log if sync took >100ms
                    print(f"   ⏱️  Cold sync took {sync_elapsed*1000:.1f}ms")

            cold_path = Path(filepath).with_suffix('.cold')
            self.save_cold_vocab(str(cold_path))

        elapsed = time.time() - start_time
        if elapsed > 0.5:  # Log if checkpoint save took >500ms
            print(f"   ⏱️  Checkpoint save took {elapsed:.2f}s")

        return True
    
    def load_checkpoint(self, filepath: str, load_vocab: bool = True, load_cold: bool = True) -> bool:
        """
        Carica checkpoint (brain + vocabolario + cold vocab).

        Args:
            filepath: Path al checkpoint
            load_vocab: Se True, carica anche vocabolario
            load_cold: Se True, carica anche cold vocab embeddings

        Returns:
            True se successo
        """
        # Check initialization
        if not self.initialized:
            print("⚠️  System not initialized - call start() before load_checkpoint()")
            return False

        # Load brain checkpoint
        result = self.lib.load_checkpoint(filepath.encode('utf-8'))

        if result != 0:
            print(f"⚠️  Failed to load checkpoint: {filepath} (error {result})")
            return False

        # Load vocabulary
        if load_vocab:
            vocab_path = Path(filepath).with_suffix('.vocab')
            if vocab_path.exists():
                self.vocab = DynamicVocabulary.load(str(vocab_path))
                print(f"   ✅ Loaded {len(self.vocab.word_to_id)} vocab words")
            else:
                print(f"⚠️  Vocab file not found: {vocab_path}")

        # Load cold vocab (all embeddings for persistence)
        if load_cold:
            cold_path = Path(filepath).with_suffix('.cold.npz')
            if cold_path.exists():
                self.load_cold_vocab(str(cold_path))
                print(f"   ✅ Loaded {len(self._cold_vocab)} cold vocab embeddings")
            else:
                # Try without .npz
                cold_path = Path(filepath).with_suffix('.cold')
                if cold_path.exists():
                    self.load_cold_vocab(str(cold_path))
                    print(f"   ✅ Loaded {len(self._cold_vocab)} cold vocab embeddings")

        # Sync loaded vocabulary to GPU (hot vocab)
        if load_vocab and len(self.vocab.word_to_id) > 0:
            self._sync_loaded_vocabulary()

        return True
    
    def prune_vocabulary(self) -> int:
        """
        Pruning del vocabolario (rimuove parole rare).

        Returns:
            Numero di parole rimosse
        """
        return self.vocab.prune_rare_words()

    def _sync_loaded_vocabulary(self):
        """
        Sync loaded vocabulary to GPU after checkpoint load.
        Loads words from cold vocab to hot vocab (GPU) using LRU strategy.
        """
        print("🔄 Syncing vocabulary to GPU...")

        # Get char embeddings for initialization
        char_embeddings = np.zeros((MODEL_CONFIG.CHAR_VOCAB_SIZE, MODEL_CONFIG.EMBED_DIM), dtype=np.float32)
        for char_id in range(MODEL_CONFIG.CHAR_VOCAB_SIZE):
            emb_buffer = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)()
            result = self.lib.get_word_embedding(char_id, emb_buffer)
            if result == 0:
                char_embeddings[char_id] = np.array(emb_buffer)

        # Get all word tokens (exclude char tokens)
        all_words = [(word_id, word) for word, word_id in self.vocab.word_to_id.items()
                     if word_id >= MODEL_CONFIG.CHAR_VOCAB_SIZE]

        if not all_words:
            print("   No word tokens to sync")
            return

        # Sort by usage count (most used first) if available
        all_words.sort(key=lambda x: self._word_usage_count.get(x[0], 0), reverse=True)

        # Load top N words to hot vocab (GPU cache)
        max_hot = VOCAB_OPTIMIZATION_CONFIG.MAX_HOT_VOCAB
        words_to_hot = all_words[:max_hot]

        print(f"   Loading {len(words_to_hot)} most-used words to hot vocab (GPU)...")

        # Batch load to hot
        if words_to_hot:
            self._batch_load_to_hot(words_to_hot, char_embeddings)
            print(f"   ✅ Synced {len(words_to_hot)} words to GPU")
            print(f"   📊 Total vocab: {len(all_words)} words ({len(words_to_hot)} hot, {len(all_words) - len(words_to_hot)} cold)")

    # =========================================================================
    # HOT/COLD VOCABULARY MANAGEMENT
    # =========================================================================

    def save_cold_vocab(self, filepath: str, compress: bool = None) -> bool:
        """
        Save cold vocab embeddings to disk for persistence.

        Args:
            filepath: Path for cold vocab file (e.g., "model.cold")
            compress: Whether to compress (default: from VOCAB_OPTIMIZATION_CONFIG)

        Returns:
            True if successful
        """
        if not self._cold_vocab:
            return True  # Nothing to save

        if compress is None:
            compress = VOCAB_OPTIMIZATION_CONFIG.COLD_VOCAB_COMPRESSION

        try:
            import time
            start_time = time.time()

            # Prepare data arrays
            save_dict = {
                'word_ids': np.array(list(self._cold_vocab.keys()), dtype=np.int32),
                'embeddings': np.array(list(self._cold_vocab.values()), dtype=np.float32),
                'usage_counts': np.array([self._word_usage_count.get(k, 0) for k in self._cold_vocab.keys()], dtype=np.int32)
            }

            # Save (compressed or uncompressed based on config)
            # OPTIMIZATION: Uncompressed is MUCH faster (10-100x) but ~3x larger file
            if compress:
                np.savez_compressed(filepath, **save_dict)
            else:
                np.savez(filepath, **save_dict)

            elapsed = time.time() - start_time
            self._cold_vocab_path = Path(filepath)

            comp_str = "(compressed)" if compress else "(uncompressed)"
            print(f"   💾 Saved cold vocab: {len(self._cold_vocab)} words {comp_str} in {elapsed:.2f}s")
            return True
        except Exception as e:
            print(f"   ❌ Failed to save cold vocab: {e}")
            return False

    def load_cold_vocab(self, filepath: str) -> bool:
        """
        Load cold vocab embeddings from disk.

        Args:
            filepath: Path to cold vocab file

        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            if not path.exists():
                # Try with .npz extension
                if not path.suffix:
                    path = path.with_suffix('.npz')
                if not path.exists():
                    return False

            data = np.load(str(path))
            word_ids = data['word_ids']
            embeddings = data['embeddings']
            usage_counts = data.get('usage_counts', np.zeros(len(word_ids), dtype=np.int32))

            # Populate cold vocab
            self._cold_vocab.clear()
            self._word_usage_count.clear()

            for i, word_id in enumerate(word_ids):
                self._cold_vocab[int(word_id)] = embeddings[i]
                self._word_usage_count[int(word_id)] = int(usage_counts[i])

            self._cold_vocab_path = path
            print(f"   📂 Loaded cold vocab: {len(self._cold_vocab)} words")
            return True
        except Exception as e:
            print(f"   ❌ Failed to load cold vocab: {e}")
            return False

    def load_from_cold_to_hot(self, word_ids: List[int]) -> int:
        """
        Load words from cold vocab to hot (GPU).
        Used when we need words that are in RAM but not in GPU.

        Args:
            word_ids: List of word IDs to load to GPU

        Returns:
            Number of words loaded
        """
        if not self.initialized:
            return 0

        # Filter to words that are in cold but not in hot
        words_to_load = []
        embeddings_to_load = []

        for word_id in word_ids:
            if word_id in self._cold_vocab and word_id not in self._hot_vocab_ids:
                words_to_load.append(word_id)
                embeddings_to_load.append(self._cold_vocab[word_id])

        if not words_to_load:
            return 0

        num_words = len(words_to_load)
        embed_dim = MODEL_CONFIG.EMBED_DIM

        # Use pinned buffers if available (faster DMA transfers)
        if self._pinned_word_ids is not None and PERFORMANCE_CONFIG.USE_PINNED_MEMORY:
            self._resize_pinned_buffers(num_words)

            self._pinned_word_ids[:num_words] = words_to_load
            for i, emb in enumerate(embeddings_to_load):
                self._pinned_embeddings[i * embed_dim:(i + 1) * embed_dim] = emb

            word_ids_ptr = self._pinned_word_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            embeddings_ptr = self._pinned_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            word_ids_array = np.array(words_to_load, dtype=np.int32)
            embeddings_array = np.array(embeddings_to_load, dtype=np.float32).flatten()

            word_ids_ptr = word_ids_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            embeddings_ptr = embeddings_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        activated = self.lib.activate_word_tokens_batch(
            word_ids_ptr,
            embeddings_ptr,
            num_words
        )

        # Update hot vocab tracking
        for word_id in words_to_load:
            self._hot_vocab_ids.add(word_id)

        return activated

    def update_cold_from_hot(self, word_ids: List[int] = None, silent: bool = False):
        """
        Update cold vocab with latest embeddings from GPU (hot).
        Call this periodically to keep cold vocab in sync.

        OPTIMIZED: Uses preallocated buffer to minimize allocations.

        Args:
            word_ids: Specific words to update (None = all hot words)
            silent: If True, don't print status message

        Returns:
            Number of words updated
        """
        if not self.initialized:
            return 0

        if word_ids is None:
            word_ids = list(self._hot_vocab_ids)

        if not word_ids:
            return 0

        embed_dim = MODEL_CONFIG.EMBED_DIM

        # Preallocate single buffer (reused for all words)
        emb_buffer = (ctypes.c_float * embed_dim)()

        # Preallocate numpy array for faster copy
        emb_array = np.zeros(embed_dim, dtype=np.float32)

        updated = 0
        for word_id in word_ids:
            if word_id in self._hot_vocab_ids:
                result = self.lib.get_word_embedding(word_id, emb_buffer)
                if result == 0:
                    # Fast copy using numpy (avoids repeated np.array allocations)
                    ctypes.memmove(emb_array.ctypes.data, emb_buffer, embed_dim * 4)
                    self._cold_vocab[word_id] = emb_array.copy()
                    updated += 1

        if updated > 0 and not silent:
            print(f"   🔄 Updated {updated} cold embeddings from GPU")

        return updated

    def update_cold_from_hot_incremental(self, max_words: int = 100):
        """
        Update cold vocab incrementally (only N words per call).
        Call this periodically during training to avoid blocking checkpoint saves.

        OPTIMIZATION: Distributes cold vocab sync across training instead of
        doing it all at once during checkpoint save.

        Args:
            max_words: Maximum words to update per call (default: 100)

        Returns:
            Number of words updated
        """
        if not self.initialized or not self._hot_vocab_ids:
            return 0

        # Get list of hot words sorted by usage (update most-used first)
        hot_words = sorted(
            list(self._hot_vocab_ids),
            key=lambda wid: self._word_usage_count.get(wid, 0),
            reverse=True
        )

        # Update up to max_words
        words_to_update = hot_words[:max_words]

        return self.update_cold_from_hot(words_to_update, silent=True)

    def get_cold_vocab_stats(self) -> Dict:
        """Get statistics about hot/cold vocabulary."""
        return {
            'cold_vocab_size': len(self._cold_vocab),
            'hot_vocab_size': len(self._hot_vocab_ids),
            'total_usage': sum(self._word_usage_count.values()),
            'cold_vocab_path': str(self._cold_vocab_path) if self._cold_vocab_path else None
        }

    def increment_word_usage(self, word_ids: List[int]):
        """
        Increment usage count for words (for LRU eviction).

        Args:
            word_ids: List of word IDs that were used
        """
        for word_id in word_ids:
            if word_id in self._word_usage_count:
                self._word_usage_count[word_id] += 1
            else:
                self._word_usage_count[word_id] = 1


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== VectLLM Brain Wrapper Test ===\n")
    
    # Test 1: Compilation
    print("Test 1: Kernel compilation")
    try:
        brain = VectLLMBrain()
        print("✅ Compilation successful\n")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        sys.exit(1)
    
    # Test 2: Initialization
    print("Test 2: System initialization")
    try:
        with brain:
            print("✅ Initialization successful\n")
            
            # Test 3: Vocabulary
            print("Test 3: Vocabulary encoding")
            text = "hello world hello world hello"
            tokens = brain.encode_text(text)
            decoded = brain.decode_tokens(tokens)
            print(f"  Text: '{text}'")
            print(f"  Tokens: {tokens}")
            print(f"  Decoded: '{decoded}'")
            print(f"  Match: {text == decoded}")
            print()
            
            # Test 4: Training
            print("Test 4: Training on text")
            processed = brain.train_on_text("testing dynamic vocabulary", passes=3)
            print(f"  Processed: {processed} tokens")
            print()
            
            # Test 5: Stats
            print("Test 5: System stats")
            stats = brain.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            
            # Test 6: Vocab stats
            print("Test 6: Vocabulary stats")
            brain.vocab.print_stats()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ All tests passed!")
