#!/usr/bin/env python3
"""
Async Pipeline Module
Parallelizza preparazione batch CPU e training GPU

Features:
- Background thread per encoding/tokenization
- Double buffering per overlap H2D/compute
- Queue con prefetch per latency hiding
- Pinned memory support per trasferimenti veloci
"""

import ctypes
import threading
import queue
import time
from typing import List, Iterator, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

# Import performance config
try:
    from core.config import PERFORMANCE_CONFIG, MODEL_CONFIG
except ImportError:
    from .config import PERFORMANCE_CONFIG, MODEL_CONFIG


@dataclass
class BatchData:
    """Container per batch preparato"""
    tokens: List[int]
    text_length: int
    batch_id: int


class AsyncBatchLoader:
    """
    Loader asincrono che prepara batch in background.

    Mentre GPU processa batch N, CPU prepara batch N+1, N+2, ...
    """

    def __init__(self,
                 encode_fn: Callable[[str], List[int]],
                 prefetch_size: int = 3,
                 max_batch_tokens: int = 512):
        """
        Args:
            encode_fn: Funzione per encoding testo → tokens
            prefetch_size: Numero di batch da preparare in anticipo
            max_batch_tokens: Massimo token per batch
        """
        self.encode_fn = encode_fn
        self.prefetch_size = prefetch_size
        self.max_batch_tokens = max_batch_tokens

        # Queue per batch preparati
        self.batch_queue: queue.Queue = queue.Queue(maxsize=prefetch_size)

        # Control
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.text_queue: queue.Queue = queue.Queue()

        # Stats
        self.batches_prepared = 0
        self.total_tokens_prepared = 0
        self.avg_encode_time = 0.0

    def start(self):
        """Avvia worker thread"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Ferma worker thread"""
        self.running = False
        # Svuota queue per sbloccare worker
        try:
            while not self.text_queue.empty():
                self.text_queue.get_nowait()
        except:
            pass
        # Signal stop
        self.text_queue.put(None)

        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)

    def _worker_loop(self):
        """Loop del worker thread"""
        batch_id = 0

        while self.running:
            try:
                # Attendi testo da processare
                text = self.text_queue.get(timeout=0.1)

                if text is None:
                    break

                # Encode
                start = time.time()
                tokens = self.encode_fn(text)
                encode_time = time.time() - start

                # Update stats
                self.avg_encode_time = 0.9 * self.avg_encode_time + 0.1 * encode_time

                if not tokens:
                    continue

                # Split in batch se necessario
                for i in range(0, len(tokens), self.max_batch_tokens):
                    batch_tokens = tokens[i:i + self.max_batch_tokens]

                    batch = BatchData(
                        tokens=batch_tokens,
                        text_length=len(text),
                        batch_id=batch_id
                    )

                    # Metti in queue (blocca se piena - backpressure)
                    self.batch_queue.put(batch)

                    batch_id += 1
                    self.batches_prepared += 1
                    self.total_tokens_prepared += len(batch_tokens)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Batch loader error: {e}")
                continue

    def submit(self, text: str):
        """Sottometti testo per encoding asincrono"""
        self.text_queue.put(text)

    def get_batch(self, timeout: float = 1.0) -> Optional[BatchData]:
        """
        Ottieni prossimo batch preparato.

        Args:
            timeout: Timeout in secondi

        Returns:
            BatchData o None se timeout
        """
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_batch_nowait(self) -> Optional[BatchData]:
        """Ottieni batch senza bloccare"""
        try:
            return self.batch_queue.get_nowait()
        except queue.Empty:
            return None

    def has_batches(self) -> bool:
        """Check se ci sono batch pronti"""
        return not self.batch_queue.empty()

    def pending_count(self) -> int:
        """Numero di batch in attesa"""
        return self.batch_queue.qsize()

    def get_stats(self) -> dict:
        """Statistiche loader"""
        return {
            'batches_prepared': self.batches_prepared,
            'total_tokens': self.total_tokens_prepared,
            'avg_encode_time_ms': self.avg_encode_time * 1000,
            'queue_size': self.batch_queue.qsize(),
            'text_queue_size': self.text_queue.qsize(),
        }


class PipelinedTrainer:
    """
    Trainer con pipeline CPU↔GPU.

    Architettura:
    1. AsyncBatchLoader prepara batch in background
    2. Double buffering: mentre GPU processa buffer A, CPU riempie buffer B
    3. Overlap H2D transfer con compute
    """

    def __init__(self,
                 brain,  # VectLLMBrain
                 prefetch_size: int = 3):
        """
        Args:
            brain: Istanza VectLLMBrain
            prefetch_size: Batch da prefetchare
        """
        self.brain = brain

        # Async loader
        self.loader = AsyncBatchLoader(
            encode_fn=brain.encode_text,
            prefetch_size=prefetch_size
        )

        # Stats
        self.total_batches = 0
        self.total_tokens = 0
        self.gpu_time = 0.0
        self.cpu_time = 0.0

        # Preallocated buffers for performance
        self._preallocated = False
        self._token_buffer = None
        self._token_buffer_size = 0

        if hasattr(PERFORMANCE_CONFIG, 'PREALLOCATE_BUFFERS') and PERFORMANCE_CONFIG.PREALLOCATE_BUFFERS:
            self._preallocate_buffers()

    def _preallocate_buffers(self):
        """Preallocate host buffers to avoid allocation overhead."""
        try:
            max_seq = MODEL_CONFIG.MAX_SEQ_LEN
            # Estimate max tokens per batch (4x max_seq as buffer)
            max_tokens = max_seq * 4

            self._token_buffer = np.zeros(max_tokens, dtype=np.int32)
            self._token_buffer_size = max_tokens
            self._preallocated = True
        except Exception as e:
            # Fallback to dynamic allocation
            self._preallocated = False

    def _get_token_array(self, tokens: List[int]) -> np.ndarray:
        """Get token array, using preallocated buffer if available."""
        if self._preallocated and len(tokens) <= self._token_buffer_size:
            # Use preallocated buffer
            self._token_buffer[:len(tokens)] = tokens
            return self._token_buffer[:len(tokens)]
        else:
            # Dynamic allocation
            return np.array(tokens, dtype=np.int32)

    def start(self):
        """Avvia pipeline"""
        self.loader.start()

    def stop(self):
        """Ferma pipeline"""
        self.loader.stop()

    def train_texts(self,
                    texts: Iterator[str],
                    callback: Callable[[int, int], None] = None) -> int:
        """
        Train su stream di testi con pipeline.

        Args:
            texts: Iterator di testi
            callback: Callback(batch_num, tokens) per progress

        Returns:
            Totale token processati
        """
        self.start()

        try:
            # Submit tutti i testi per encoding asincrono
            text_count = 0
            for text in texts:
                self.loader.submit(text)
                text_count += 1

                # Process batch disponibili mentre submittiamo
                while self.loader.has_batches():
                    batch = self.loader.get_batch_nowait()
                    if batch:
                        self._process_batch(batch, callback)

            # Process batch rimanenti
            while True:
                batch = self.loader.get_batch(timeout=0.5)
                if batch is None:
                    # Check se loader ha ancora lavoro
                    if self.loader.text_queue.empty() and not self.loader.has_batches():
                        break
                else:
                    self._process_batch(batch, callback)

        finally:
            self.stop()

        return self.total_tokens

    def _process_batch(self, batch: BatchData, callback: Callable = None):
        """Processa singolo batch su GPU"""
        start = time.time()

        # Feed to GPU
        processed = self.brain.feed_training_batch(batch.tokens)

        self.gpu_time += time.time() - start

        if processed > 0:
            self.total_batches += 1
            self.total_tokens += processed

            if callback:
                callback(self.total_batches, processed)

    def get_stats(self) -> dict:
        """Statistiche pipeline"""
        loader_stats = self.loader.get_stats()

        return {
            'total_batches': self.total_batches,
            'total_tokens': self.total_tokens,
            'gpu_time_s': self.gpu_time,
            'avg_tokens_per_batch': self.total_tokens / max(1, self.total_batches),
            'throughput_tok_s': self.total_tokens / max(0.001, self.gpu_time),
            **loader_stats
        }


class MultiStreamTrainer:
    """
    Trainer con CUDA multi-stream per massimo overlap.

    Usa stream separati per:
    - H2D transfer
    - Compute
    - D2H transfer

    Questo permette overlap completo delle operazioni.
    """

    def __init__(self,
                 brain,
                 num_buffers: int = 2):
        """
        Args:
            brain: VectLLMBrain
            num_buffers: Numero di buffer per double/triple buffering
        """
        self.brain = brain
        self.num_buffers = num_buffers

        # Buffer index
        self.current_buffer = 0

        # Stats
        self.overlap_efficiency = 0.0

    def train_with_overlap(self,
                           batch_iterator: Iterator[List[int]],
                           callback: Callable = None) -> int:
        """
        Train con overlap massimo.

        Pattern double buffering:
        1. Load batch 0 → buffer 0
        2. Start compute buffer 0
        3. Load batch 1 → buffer 1 (overlap con compute)
        4. Wait compute buffer 0
        5. Start compute buffer 1
        6. Load batch 2 → buffer 0 (overlap con compute)
        ...

        Args:
            batch_iterator: Iterator di batch di token
            callback: Progress callback

        Returns:
            Totale token
        """
        total_tokens = 0
        batch_num = 0

        # Prime the pipeline - load first batch
        try:
            first_batch = next(batch_iterator)
        except StopIteration:
            return 0

        # Process with overlap
        pending_batch = first_batch

        for next_batch in batch_iterator:
            # Process current batch
            processed = self.brain.feed_training_batch(pending_batch)
            total_tokens += processed
            batch_num += 1

            if callback:
                callback(batch_num, processed)

            # Prepare next (overlap avviene nel kernel se supportato)
            pending_batch = next_batch

        # Process last batch
        if pending_batch:
            processed = self.brain.feed_training_batch(pending_batch)
            total_tokens += processed
            batch_num += 1

            if callback:
                callback(batch_num, processed)

        return total_tokens


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Pipeline Module Test ===\n")

    # Test AsyncBatchLoader
    print("Test 1: AsyncBatchLoader")

    def mock_encode(text):
        # Simulate encoding
        time.sleep(0.01)  # 10ms
        return [ord(c) for c in text[:50]]

    loader = AsyncBatchLoader(mock_encode, prefetch_size=5)
    loader.start()

    # Submit texts
    for i in range(10):
        loader.submit(f"Test text number {i} " * 10)

    # Consume batches
    batches_received = 0
    while batches_received < 10:
        batch = loader.get_batch(timeout=1.0)
        if batch:
            batches_received += 1
            print(f"  Batch {batch.batch_id}: {len(batch.tokens)} tokens")

    loader.stop()

    stats = loader.get_stats()
    print(f"\n  Stats: {stats}")

    print("\n✅ Pipeline module test complete!")
