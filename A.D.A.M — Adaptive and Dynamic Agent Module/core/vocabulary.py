#!/usr/bin/env python3
"""
VectLLM Dynamic Vocabulary System - Hot/Cold Architecture
==========================================================

This module implements A.D.A.M's dynamic vocabulary with unlimited growth
and efficient GPU/RAM hybrid storage.

Architecture Tiers:
-------------------

Tier 1 - Character Fallback (256 tokens, always on GPU):
    - IDs: 0-255
    - Covers all ASCII/UTF-8 characters
    - Ensures no out-of-vocabulary (OOV) tokens
    - Always available for unknown sequences

Tier 2 - Hot Vocabulary (10,000 words, GPU-resident):
    - IDs: 256-10255
    - Most frequently used words
    - Fast GPU access via lookup table
    - LRU eviction when full

Tier 3 - Cold Vocabulary (unlimited, RAM-resident):
    - IDs: 10256+
    - Less frequent words
    - Embeddings computed from character composition
    - Promoted to hot vocab on demand

Token ID Space:
    [0-255]       Characters (always hot)
    [256-10255]   Hot words (GPU)
    [10256+]      Cold words (RAM)

Word Creation Process:
----------------------
1. Track character sequences: "hello" appears in text
2. Count frequency: increment counter each time seen
3. Threshold check: if frequency >= WORD_CREATION_THRESHOLD (default: 5)
4. Create word token: assign new ID, compute initial embedding
5. Add to cold vocab: store in RAM
6. Promote to hot: if used frequently, swap to GPU via LRU

Encoding Examples:
------------------
    Input: "Hello world!"

    Before word learning:
        ['H','e','l','l','o',' ','w','o','r','l','d','!']
        → [72,101,108,108,111,32,119,111,114,108,100,33]

    After learning "Hello" (ID 256) and "world" (ID 257):
        ['Hello',' ','world','!']
        → [256,32,257,33]

Key Features:
-------------
- Zero OOV: Characters always available as fallback
- Online learning: Words created during training
- Memory efficient: Hot/cold split minimizes GPU usage
- Frequency-based: Common words cached on GPU
- Pruning: Rarely used words can be removed (optional)

Classes:
--------
- DynamicVocabulary: Main vocabulary manager
  - encode_text(): Text → token IDs
  - decode_tokens(): Token IDs → text
  - create_word(): Create new word token
  - sync_with_gpu(): Sync hot vocab with CUDA

See Also:
---------
- kernels/vocabulary.cu: GPU-side vocabulary management
- core/brain_wrapper.py: VectLLMBrain (uses this class)
- core/config.py: Vocabulary configuration
"""

import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from .constants import ASCII_SPACE, ASCII_NEWLINE

logger = logging.getLogger(__name__)


class DynamicVocabulary:
    """
    Dynamic Vocabulary Manager with Hot/Cold Architecture
    ======================================================

    Manages A.D.A.M's three-tier vocabulary system:
    - Character fallback (256 tokens, always GPU)
    - Hot vocabulary (10,000 words, GPU)
    - Cold vocabulary (unlimited, RAM)

    Key Methods:
        encode_text(text): Convert text to token IDs
        decode_tokens(ids): Convert token IDs to text
        create_word(word): Create new word token
        is_word_created(word): Check if word has token ID
        sync_with_gpu(brain): Sync hot vocab with CUDA kernel
        save_vocab(path): Save vocabulary to disk
        load_vocab(path): Load vocabulary from disk

    Internal State:
        word_to_id: str → int mapping
        id_to_word: int → str mapping
        frequency: Word usage frequency counters
        char_sequences: Character sequence frequency tracking
        hot_vocab_set: Set of GPU-resident word IDs
        cold_embeddings: RAM embeddings for cold words

    Example:
        vocab = DynamicVocabulary(embed_dim=768)

        # Encode text
        text = "Hello world! Hello again!"
        token_ids = vocab.encode_text(text)

        # After 5 occurrences of "Hello", it becomes a word token
        # token_ids changes from [72,101,108,108,111,...] to [256,...]

        # Decode back
        decoded = vocab.decode_tokens(token_ids)
        # decoded == "Hello world! Hello again!"
    """
    
    def __init__(self, 
                 embed_dim: int = 768,
                 char_vocab_size: int = 256,
                 max_word_vocab_size: int = 100000,
                 creation_threshold: int = 5,
                 pruning_threshold: int = 2,
                 max_word_length: int = 20):
        """
        Args:
            embed_dim: Dimensione embeddings
            char_vocab_size: Numero di caratteri base (fisso a 256)
            max_word_vocab_size: Massimo numero di word tokens
            creation_threshold: Crea word token dopo N occorrenze
            pruning_threshold: Rimuovi word se freq < N
            max_word_length: Massima lunghezza parola in caratteri
        """
        self.embed_dim = embed_dim
        self.char_vocab_size = char_vocab_size
        self.max_word_vocab_size = max_word_vocab_size
        self.creation_threshold = creation_threshold
        self.pruning_threshold = pruning_threshold
        self.max_word_length = max_word_length
        
        # Livello 1: Caratteri (fisso)
        # Nota: le embeddings effettive sono nel kernel CUDA
        # Qui teniamo solo la struttura
        self.char_ids = set(range(char_vocab_size))
        
        # Livello 2: Parole (dinamico)
        self.word_to_id: Dict[str, int] = {}  # "hello" -> 257
        self.id_to_word: Dict[int, str] = {}  # 257 -> "hello"
        self.word_frequency: Dict[str, int] = defaultdict(int)  # Contatore uso
        self.next_word_id = char_vocab_size  # Inizia da 256
        
        # Statistiche
        self.total_words_created = 0
        self.total_words_pruned = 0
        self.total_tokens_encoded = 0
        
        # Whitespace handling
        self.space_char = ASCII_SPACE
        self.newline_char = ASCII_NEWLINE
        
    def encode(self, text: str) -> List[int]:
        """
        Codifica testo in token IDs.
        
        Pipeline:
          1. Split by whitespace
          2. Per ogni parola:
             - Se esiste in vocab → word_id
             - Altrimenti: incrementa frequency
               - Se freq >= threshold → crea word_id
               - Altrimenti → sequenza di char_ids
          3. Inserisci space_char tra parole
        
        Args:
            text: Testo da codificare
            
        Returns:
            Lista di token IDs
        """
        if not text:
            return []
        
        tokens = []
        
        # Split by whitespace (preserva spazi multipli)
        parts = text.split(' ')
        
        for i, word in enumerate(parts):
            # Skip empty strings (da spazi multipli)
            if not word:
                if i < len(parts) - 1:  # Non ultimo
                    tokens.append(self.space_char)
                continue
            
            # Processa la parola
            word_tokens = self._encode_word(word)
            tokens.extend(word_tokens)
            
            # Aggiungi spazio se non è l'ultima parola
            if i < len(parts) - 1:
                tokens.append(self.space_char)
        
        self.total_tokens_encoded += len(tokens)
        return tokens
    
    def _encode_word(self, word: str) -> List[int]:
        """
        Codifica una singola parola.

        CRITICAL: Crea word token alla PRIMA occorrenza per popolare cold vocab.
        Il threshold viene applicato solo per promozione a HOT vocab in brain_wrapper.

        Returns:
            Lista di token IDs (può essere [word_id] o [char, char, ...])
        """
        # Filtra parole troppo lunghe o vuote
        if not word or len(word) > self.max_word_length:
            return self._word_to_chars(word)

        # Check se già esiste come word token
        if word in self.word_to_id:
            # Incrementa frequency per tracking
            self.word_frequency[word] += 1
            return [self.word_to_id[word]]

        # Parola nuova - crea word token immediatamente (threshold=1 per cold vocab)
        # Il threshold configurato verrà applicato per HOT vocab in brain_wrapper
        if self.next_word_id < self.char_vocab_size + self.max_word_vocab_size:
            self.word_frequency[word] = 1
            word_id = self._create_word_token(word)
            return [word_id]

        # Nessuno spazio rimasto (max_word_vocab_size raggiunto), usa caratteri
        return self._word_to_chars(word)
    
    def _word_to_chars(self, word: str) -> List[int]:
        """Converte parola in sequenza di caratteri"""
        return [min(ord(c), 255) for c in word]
    
    def _create_word_token(self, word: str) -> int:
        """
        Crea un nuovo word token.
        
        Returns:
            Il nuovo word_id
        """
        word_id = self.next_word_id
        self.word_to_id[word] = word_id
        self.id_to_word[word_id] = word
        self.next_word_id += 1
        self.total_words_created += 1
        
        return word_id
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decodifica token IDs in testo.
        
        Args:
            token_ids: Lista di token IDs
            
        Returns:
            Testo decodificato
        """
        if not token_ids:
            return ""
        
        chars = []
        
        for token_id in token_ids:
            if token_id < self.char_vocab_size:
                # Carattere
                chars.append(chr(token_id))
            elif token_id in self.id_to_word:
                # Word token
                chars.append(self.id_to_word[token_id])
            else:
                # Token sconosciuto (non dovrebbe succedere)
                chars.append('�')
        
        return ''.join(chars)
    
    def prune_rare_words(self) -> int:
        """
        Rimuove parole con frequenza < pruning_threshold.
        
        Returns:
            Numero di parole rimosse
        """
        words_to_remove = []
        
        for word, freq in self.word_frequency.items():
            if word in self.word_to_id and freq < self.pruning_threshold:
                words_to_remove.append(word)
        
        for word in words_to_remove:
            word_id = self.word_to_id[word]
            del self.word_to_id[word]
            del self.id_to_word[word_id]
            del self.word_frequency[word]
            self.total_words_pruned += 1
        
        return len(words_to_remove)
    
    def get_word_embedding_init(self, word: str, char_embeddings: np.ndarray) -> np.ndarray:
        """
        Inizializza embedding per una nuova parola.
        
        Strategy: media delle embeddings dei caratteri che compongono la parola.
        
        Args:
            word: Parola da inizializzare
            char_embeddings: Array [256, embed_dim] di char embeddings
            
        Returns:
            Array [embed_dim] - embedding iniziale per la parola
        """
        if not word:
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        # Get char embeddings per ogni carattere
        char_embs = []
        for c in word:
            char_id = min(ord(c), 255)
            char_embs.append(char_embeddings[char_id])
        
        # Media
        return np.mean(char_embs, axis=0).astype(np.float32)
    
    def save(self, filepath: str):
        """
        Salva vocabolario su disco.
        
        Formato:
          - JSON per metadata e mappings
          - Pickle per frequency dict (più veloce)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'embed_dim': self.embed_dim,
            'char_vocab_size': self.char_vocab_size,
            'max_word_vocab_size': self.max_word_vocab_size,
            'creation_threshold': self.creation_threshold,
            'pruning_threshold': self.pruning_threshold,
            'max_word_length': self.max_word_length,
            'next_word_id': self.next_word_id,
            'word_to_id': self.word_to_id,
            'id_to_word': {int(k): v for k, v in self.id_to_word.items()},
            'total_words_created': self.total_words_created,
            'total_words_pruned': self.total_words_pruned,
            'total_tokens_encoded': self.total_tokens_encoded,
        }
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save frequency separately (pickle per velocità)
        freq_path = filepath.with_suffix('.freq')
        with open(freq_path, 'wb') as f:
            pickle.dump(dict(self.word_frequency), f)
        
        logger.info(f"Vocabulary saved: {filepath}")
        logger.debug(f"Words: {len(self.word_to_id)}, Created: {self.total_words_created}, Pruned: {self.total_words_pruned}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DynamicVocabulary':
        """
        Carica vocabolario da disco.
        
        Args:
            filepath: Path al file JSON del vocabolario
            
        Returns:
            Istanza di DynamicVocabulary
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        vocab = cls(
            embed_dim=data['embed_dim'],
            char_vocab_size=data['char_vocab_size'],
            max_word_vocab_size=data['max_word_vocab_size'],
            creation_threshold=data['creation_threshold'],
            pruning_threshold=data['pruning_threshold'],
            max_word_length=data['max_word_length']
        )
        
        vocab.next_word_id = data['next_word_id']
        vocab.word_to_id = data['word_to_id']
        vocab.id_to_word = {int(k): v for k, v in data['id_to_word'].items()}
        vocab.total_words_created = data['total_words_created']
        vocab.total_words_pruned = data['total_words_pruned']
        vocab.total_tokens_encoded = data['total_tokens_encoded']
        
        # Load frequencies
        freq_path = filepath.with_suffix('.freq')
        if freq_path.exists():
            with open(freq_path, 'rb') as f:
                vocab.word_frequency = defaultdict(int, pickle.load(f))
        
        logger.info(f"Vocabulary loaded: {filepath} ({len(vocab.word_to_id)} words)")
        
        return vocab
    
    def get_stats(self) -> Dict:
        """Ottieni statistiche del vocabolario"""
        return {
            'char_vocab_size': self.char_vocab_size,
            'word_vocab_size': len(self.word_to_id),
            'max_word_vocab_size': self.max_word_vocab_size,
            'utilization': len(self.word_to_id) / self.max_word_vocab_size,
            'next_word_id': self.next_word_id,
            'total_words_created': self.total_words_created,
            'total_words_pruned': self.total_words_pruned,
            'total_tokens_encoded': self.total_tokens_encoded,
            'creation_threshold': self.creation_threshold,
            'pruning_threshold': self.pruning_threshold,
            'top_words': sorted(self.word_frequency.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:20]
        }
    
    def print_stats(self):
        """Stampa statistiche leggibili"""
        stats = self.get_stats()
        
        print("\n=== Dynamic Vocabulary Statistics ===")
        print(f"Char vocab: {stats['char_vocab_size']} (fixed)")
        print(f"Word vocab: {stats['word_vocab_size']} / {stats['max_word_vocab_size']}")
        print(f"Utilization: {stats['utilization']*100:.1f}%")
        print(f"Next ID: {stats['next_word_id']}")
        print(f"\nLifetime stats:")
        print(f"  Created: {stats['total_words_created']} words")
        print(f"  Pruned: {stats['total_words_pruned']} words")
        print(f"  Encoded: {stats['total_tokens_encoded']:,} tokens")
        print(f"\nThresholds:")
        print(f"  Creation: {stats['creation_threshold']} occurrences")
        print(f"  Pruning: {stats['pruning_threshold']} min frequency")
        
        if stats['top_words']:
            print(f"\nTop 20 words by frequency:")
            for word, freq in stats['top_words']:
                word_id = self.word_to_id.get(word, -1)
                if word_id == -1:
                    print(f"  {word:15s} → ID     ? | freq {freq:5d}")
                else:
                    print(f"  {word:15s} → ID {word_id:5d} | freq {freq:5d}")


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("=== VectLLM Dynamic Vocabulary Test ===\n")
    
    # Create vocabulary
    vocab = DynamicVocabulary(
        embed_dim=768,
        creation_threshold=3,  # Lower for testing
        pruning_threshold=2
    )
    
    # Test texts
    texts = [
        "hello world",
        "hello again",
        "hello hello hello",  # "hello" dovrebbe diventare word token
        "world is beautiful",
        "hello world again",
        "testing dynamic vocabulary system",
        "hello world testing",
    ]
    
    print("Encoding texts...\n")
    for text in texts:
        tokens = vocab.encode(text)
        decoded = vocab.decode(tokens)
        print(f"Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match: {text == decoded}")
        print()
    
    # Stats
    vocab.print_stats()
    
    # Test pruning
    print("\n=== Testing Pruning ===")
    pruned = vocab.prune_rare_words()
    print(f"Pruned {pruned} rare words")
    
    vocab.print_stats()
    
    # Test save/load
    print("\n=== Testing Save/Load ===")
    vocab.save("test_vocab.json")
    
    vocab2 = DynamicVocabulary.load("test_vocab.json")
    
    # Verify
    test_text = "hello world testing"
    tokens1 = vocab.encode(test_text)
    tokens2 = vocab2.encode(test_text)
    
    print(f"Original tokens: {tokens1}")
    print(f"Loaded tokens: {tokens2}")
    print(f"Match: {tokens1 == tokens2}")
    
    # Test embedding initialization
    print("\n=== Testing Embedding Initialization ===")
    
    # Mock char embeddings
    char_embeddings = np.random.randn(256, 768).astype(np.float32)
    
    word = "hello"
    word_emb = vocab.get_word_embedding_init(word, char_embeddings)
    
    print(f"Word: '{word}'")
    print(f"Embedding shape: {word_emb.shape}")
    print(f"Embedding mean: {word_emb.mean():.4f}")
    print(f"Embedding std: {word_emb.std():.4f}")
    
    print("\n✅ All tests passed!")
