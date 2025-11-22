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
from typing import Optional, List, Dict

from .config import MODEL_CONFIG, TRAINING_CONFIG, RUNTIME_CONFIG
from .vocabulary import DynamicVocabulary


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
                max_word_vocab_size=MODEL_CONFIG.MAX_WORD_VOCAB_SIZE,
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
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text usando vocabolario dinamico"""
        old_vocab_size = len(self.vocab.word_to_id)
        tokens = self.vocab.encode(text)
        new_vocab_size = len(self.vocab.word_to_id)
        
        # Se vocabolario è cresciuto, sincronizza con kernel
        if new_vocab_size > old_vocab_size:
            self._sync_new_words(old_vocab_size, new_vocab_size)
        
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens a text"""
        return self.vocab.decode(tokens)
    
    def _sync_new_words(self, old_size: int, new_size: int):
        """
        Sincronizza nuove parole con il kernel CUDA.
        Attiva word embeddings per parole appena create.
        """
        if not self.initialized:
            return
        
        # Get char embeddings from kernel (per calcolare init embeddings)
        char_embeddings = np.zeros((MODEL_CONFIG.CHAR_VOCAB_SIZE, MODEL_CONFIG.EMBED_DIM), dtype=np.float32)
        for char_id in range(MODEL_CONFIG.CHAR_VOCAB_SIZE):
            emb_buffer = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)()
            result = self.lib.get_word_embedding(char_id, emb_buffer)
            if result == 0:
                char_embeddings[char_id] = np.array(emb_buffer)
        
        # Attiva nuove parole
        activated = 0
        for word, word_id in self.vocab.word_to_id.items():
            if word_id >= MODEL_CONFIG.CHAR_VOCAB_SIZE:
                # Questa è una word (non char)
                word_idx = word_id - MODEL_CONFIG.CHAR_VOCAB_SIZE
                
                # Calcola embedding iniziale (media dei char)
                init_emb = self.vocab.get_word_embedding_init(word, char_embeddings)
                
                # Converti a ctypes array
                emb_array = (ctypes.c_float * MODEL_CONFIG.EMBED_DIM)(*init_emb)
                
                # Attiva nel kernel
                result = self.lib.activate_word_token(word_id, emb_array)
                if result == 0:
                    activated += 1
        
        if activated > 0:
            print(f"   🔄 Synced {activated} new words to GPU")
    
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
        
        # Convert to ctypes array
        token_array = (ctypes.c_int * len(tokens))(*tokens)
        
        result = self.lib.feed_training_batch(token_array, len(tokens))
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
    
    def save_checkpoint(self, filepath: str, save_vocab: bool = True) -> bool:
        """
        Salva checkpoint (brain + vocabolario).
        
        Args:
            filepath: Path al checkpoint (es. "checkpoint.ckpt")
            save_vocab: Se True, salva anche vocabolario
            
        Returns:
            True se successo
        """
        # Save brain checkpoint
        result = self.lib.save_checkpoint(filepath.encode('utf-8'))
        
        if result != 0:
            return False
        
        # Save vocabulary
        if save_vocab:
            vocab_path = Path(filepath).with_suffix('.vocab')
            self.vocab.save(str(vocab_path))
        
        return True
    
    def load_checkpoint(self, filepath: str, load_vocab: bool = True) -> bool:
        """
        Carica checkpoint (brain + vocabolario).
        
        Args:
            filepath: Path al checkpoint
            load_vocab: Se True, carica anche vocabolario
            
        Returns:
            True se successo
        """
        # Load brain checkpoint
        result = self.lib.load_checkpoint(filepath.encode('utf-8'))
        
        if result != 0:
            return False
        
        # Load vocabulary
        if load_vocab:
            vocab_path = Path(filepath).with_suffix('.vocab')
            if vocab_path.exists():
                self.vocab = DynamicVocabulary.load(str(vocab_path))
            else:
                print(f"⚠️  Vocab file not found: {vocab_path}")
        
        return True
    
    def prune_vocabulary(self) -> int:
        """
        Pruning del vocabolario (rimuove parole rare).
        
        Returns:
            Numero di parole rimosse
        """
        return self.vocab.prune_rare_words()


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
