#!/usr/bin/env python3
"""
VectLLM Statistics and Monitoring
Raccolta e visualizzazione metriche durante training
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metriche di training"""

    # Contatori
    total_cycles: int = 0
    total_tokens: int = 0

    # Loss tracking
    current_loss: float = 0.0
    loss_history: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Validation metrics
    validation_loss: float = 0.0
    best_validation_loss: float = float('inf')
    validation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    validations_without_improvement: int = 0

    # Performance
    tokens_per_second: float = 0.0
    cycles_per_second: float = 0.0

    # Timing
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    # Vocabulary
    vocab_size: int = 0
    vocab_utilization: float = 0.0

    # Model state
    temperature: float = 1.0
    momentum: float = 0.9
    learning_rate: float = 0.0001


class StatsCollector:
    """Raccolta statistiche durante training"""

    def __init__(self, window_size: int = 100, history_sample_rate: int = 1):
        """
        Args:
            window_size: Dimensione finestra per medie mobili
            history_sample_rate: Record history every N updates (1=all, 10=every 10th)
        """
        self.metrics = TrainingMetrics()
        self.window_size = window_size
        self.history_sample_rate = history_sample_rate

        # Buffers per medie mobili
        self.loss_window = deque(maxlen=window_size)
        self.speed_window = deque(maxlen=window_size)

        # Checkpoints
        self.checkpoint_metrics: List[Dict] = []

        # Update counter for downsampling
        self._update_count = 0
        
    def update(self,
               cycles: int = None,
               tokens: int = None,
               loss: float = None,
               vocab_size: int = None,
               vocab_utilization: float = None):
        """Aggiorna metriche"""

        now = time.time()
        dt = now - self.metrics.last_update
        self._update_count += 1

        if cycles is not None:
            self.metrics.total_cycles = cycles
            if dt > 0:
                self.metrics.cycles_per_second = (cycles - self.metrics.total_cycles) / dt

        if tokens is not None:
            delta_tokens = tokens - self.metrics.total_tokens
            self.metrics.total_tokens = tokens
            if dt > 0 and delta_tokens > 0:
                tps = delta_tokens / dt
                self.speed_window.append(tps)
                self.metrics.tokens_per_second = sum(self.speed_window) / len(self.speed_window)

        if loss is not None:
            self.metrics.current_loss = loss
            self.loss_window.append(loss)
            # Downsample history recording for performance
            if self._update_count % self.history_sample_rate == 0:
                self.metrics.loss_history.append((now, loss))

        if vocab_size is not None:
            self.metrics.vocab_size = vocab_size

        if vocab_utilization is not None:
            self.metrics.vocab_utilization = vocab_utilization

        self.metrics.last_update = now
    
    def get_perplexity(self) -> float:
        """Calcola perplexity da loss"""
        import math
        if self.metrics.current_loss > 0:
            return math.exp(self.metrics.current_loss)
        return float('inf')
    
    def get_average_loss(self, window: Optional[int] = None) -> float:
        """Media loss su finestra"""
        if not self.loss_window:
            return 0.0
        
        if window is None:
            window = len(self.loss_window)
        
        recent = list(self.loss_window)[-window:]
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_elapsed_time(self) -> float:
        """Tempo elapsed in secondi"""
        return time.time() - self.metrics.start_time
    
    def get_summary(self) -> Dict:
        """Ottieni summary completo"""
        elapsed = self.get_elapsed_time()
        
        return {
            'cycles': self.metrics.total_cycles,
            'tokens': self.metrics.total_tokens,
            'loss': self.metrics.current_loss,
            'loss_avg': self.get_average_loss(window=100),
            'perplexity': self.get_perplexity(),
            'tokens_per_sec': self.metrics.tokens_per_second,
            'cycles_per_sec': self.metrics.cycles_per_second,
            'vocab_size': self.metrics.vocab_size,
            'vocab_utilization': self.metrics.vocab_utilization,
            'elapsed_time': elapsed,
            'elapsed_hours': elapsed / 3600,
        }
    
    def print_stats(self, prefix: str = ""):
        """Stampa statistiche formattate"""
        stats = self.get_summary()
        elapsed_str = format_time(stats['elapsed_time'])
        
        print(f"{prefix}Cycles: {stats['cycles']:,}")
        print(f"{prefix}Tokens: {stats['tokens']:,}")
        print(f"{prefix}Loss: {stats['loss']:.4f} (avg: {stats['loss_avg']:.4f})")
        print(f"{prefix}Perplexity: {stats['perplexity']:.2f}")
        print(f"{prefix}Speed: {stats['tokens_per_sec']:.0f} tokens/sec")
        print(f"{prefix}Vocab: {stats['vocab_size']:,} words ({stats['vocab_utilization']*100:.1f}%)")
        print(f"{prefix}Elapsed: {elapsed_str}")
    
    def save_checkpoint_stats(self, checkpoint_path: str):
        """Salva snapshot metriche per checkpoint"""
        snapshot = {
            'checkpoint_path': checkpoint_path,
            'timestamp': time.time(),
            'metrics': self.get_summary(),
        }
        self.checkpoint_metrics.append(snapshot)
    
    def get_loss_trend(self, last_n: int = 100) -> str:
        """Ottieni trend loss (↑ ↓ →)"""
        if len(self.loss_window) < 10:
            return "→"
        
        recent = list(self.loss_window)[-last_n:]
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        diff = (second_half - first_half) / first_half
        
        if diff < -0.05:
            return "↓"  # Migliorando
        elif diff > 0.05:
            return "↑"  # Peggiorando
        else:
            return "→"  # Stabile


def format_time(seconds: float) -> str:
    """Formatta tempo in modo leggibile"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def format_number(num: int) -> str:
    """Formatta numero con K/M/B"""
    if num < 1000:
        return str(num)
    elif num < 1_000_000:
        return f"{num/1000:.1f}K"
    elif num < 1_000_000_000:
        return f"{num/1_000_000:.1f}M"
    else:
        return f"{num/1_000_000_000:.1f}B"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Stats Collector Test ===\n")
    
    collector = StatsCollector(window_size=10)
    
    # Simula training
    import random
    
    print("Simulating training...\n")
    
    for i in range(1, 101):
        # Simula metriche
        loss = 5.0 - (i * 0.03) + random.uniform(-0.1, 0.1)
        tokens = i * 100
        cycles = i * 50
        vocab = min(256 + i * 2, 500)
        
        collector.update(
            cycles=cycles,
            tokens=tokens,
            loss=loss,
            vocab_size=vocab,
            vocab_utilization=vocab / 100000
        )
        
        # Stampa ogni 25 steps
        if i % 25 == 0:
            print(f"=== Step {i} ===")
            collector.print_stats(prefix="  ")
            print(f"  Trend: {collector.get_loss_trend()}")
            print()
        
        time.sleep(0.01)  # Simula processing
    
    print("\n✅ Stats test complete!")
