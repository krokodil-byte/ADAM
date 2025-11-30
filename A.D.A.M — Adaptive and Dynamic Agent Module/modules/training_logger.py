"""
Unified Training Logger for A.D.A.M.

Provides consistent logging across all training modules:
- DatasetTrainer
- WikipediaStreamTrainer
- HFDatasetTrainer
- InteractiveChat training
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """Unified logger for all training operations."""

    def __init__(
        self,
        name: str = "adam.training",
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        console: bool = True
    ):
        """
        Initialize training logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for log output
            console: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent propagation to root logger to avoid duplicates
        self.logger.propagate = False

        # Clear existing handlers to avoid duplicates
        self.logger.handlers = []

        # Custom formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.start_time = None
        self.current_task = None
        self._console = console

    # === Core logging methods ===

    def debug(self, msg: str):
        """Debug level message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Info level message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Warning level message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Error level message."""
        self.logger.error(msg)

    # === Training-specific methods ===

    def training_start(self, trainer_type: str, **kwargs):
        """Log training session start."""
        self.start_time = datetime.now()
        self.current_task = trainer_type

        self.info(f"{'=' * 60}")
        self.info(f"Starting {trainer_type} training")
        self.info(f"{'=' * 60}")

        for key, value in kwargs.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.2f}")
            elif isinstance(value, int) and value > 1000:
                self.info(f"  {key}: {value:,}")
            else:
                self.info(f"  {key}: {value}")

    def training_end(self, **kwargs):
        """Log training session end."""
        elapsed = 0
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        self.info(f"{'=' * 60}")
        self.info(f"Training complete!")
        self.info(f"{'=' * 60}")

        for key, value in kwargs.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            elif isinstance(value, int) and value > 1000:
                self.info(f"  {key}: {value:,}")
            else:
                self.info(f"  {key}: {value}")

        if elapsed > 0:
            self.info(f"  elapsed: {elapsed/60:.1f} minutes")

        self.start_time = None
        self.current_task = None

    def pass_start(self, pass_num: int, total_passes: int):
        """Log start of training pass."""
        if pass_num > 1:
            self.info(f"--- Pass {pass_num}/{total_passes} (re-training same batch) ---")
        else:
            self.info(f"--- Pass {pass_num}/{total_passes} ---")

    def pass_end(self, pass_num: int, tokens: int, time_seconds: float):
        """Log end of training pass."""
        speed = tokens / time_seconds if time_seconds > 0 else 0
        self.info(f"Pass {pass_num} complete: {tokens:,} tokens in {time_seconds/60:.1f}m ({speed:.0f} tok/s)")

    def file_start(self, file_idx: int, total_files: int, filename: str, size_chars: int):
        """Log start of file processing."""
        self.info(f"File {file_idx}/{total_files}: {filename} ({size_chars:,} chars)")

    def file_end(self, tokens: int, time_seconds: float, reward: float, vocab_size: int, loss: float = None):
        """Log end of file processing."""
        speed = tokens / time_seconds if time_seconds > 0 else 0
        msg = f"  -> {tokens:,} tokens, {speed:.0f} tok/s, reward={reward:.4f}"
        if loss is not None:
            msg += f" (pseudo-loss={loss:.4f})"
        msg += f", vocab={vocab_size:,}"
        self.info(msg)

    def sample_progress(self, idx: int, total: int, tokens: int, reward: float, loss: float = None):
        """Log sample progress (for HF datasets)."""
        msg = f"  Sample {idx}/{total}: {tokens:,} tokens, reward={reward:.4f}"
        if loss is not None:
            msg += f" (pseudo-loss={loss:.4f})"
        self.info(msg)

    def batch_progress(self, batch_num: int, tokens: int, reward: float, speed: float, loss: float = None):
        """Log batch progress."""
        msg = f"  Batch {batch_num}: {tokens:,} tokens, reward={reward:.4f}"
        if loss is not None:
            msg += f" (pseudo-loss={loss:.4f})"
        msg += f", {speed:.0f} tok/s"
        self.info(msg)

    def article_progress(self, idx: int, title: str, tokens: int, reward: float, vocab: int, loss: float = None):
        """Log Wikipedia article progress."""
        # Truncate title if too long
        if len(title) > 30:
            title = title[:27] + "..."
        msg = f"  [{idx}] {title}: {tokens:,} tok, reward={reward:.4f}"
        if loss is not None:
            msg += f" (pseudo-loss={loss:.4f})"
        msg += f", vocab={vocab:,}"
        self.info(msg)

    def checkpoint_save(self, name: str):
        """Log checkpoint save."""
        self.info(f"  Checkpoint saved: {name}")

    def vocab_sync(self, num_words: int):
        """Log vocabulary sync to GPU."""
        if num_words > 0:
            self.debug(f"  Synced {num_words} new words to GPU")

    def stats_update(self, reward: float, vocab_size: int, tokens_total: int, speed: float = 0, loss: float = None):
        """Log periodic stats update."""
        msg = f"  Stats: reward={reward:.4f}"
        if loss is not None:
            msg += f" (pseudo-loss={loss:.4f})"
        msg += f", vocab={vocab_size:,}, tokens={tokens_total:,}"
        if speed > 0:
            msg += f", {speed:.0f} tok/s"
        self.info(msg)

    def interrupted(self):
        """Log training interruption."""
        self.warning("Training interrupted by user")

    def pipeline_stats(self, throughput: float, gpu_util: float):
        """Log pipeline performance stats."""
        self.info(f"  Pipeline: {throughput:.0f} tok/s, GPU util={gpu_util:.1%}")

    # === Validation methods ===

    def validation_start(self, num_samples: int):
        """Log start of validation."""
        self._flush()
        self.info(f"--- Validation ({num_samples} samples) ---")

    def validation_result(self, val_reward: float, best_reward: float, improved: bool, loss: float = None, best_loss: float = None):
        """Log validation result."""
        status = "NEW BEST" if improved else (f"best reward: {best_reward:.4f}" if best_reward is not None else "")
        msg = f"  Validation reward: {val_reward:.4f} ({status})"
        if loss is not None:
            best_loss_str = f"best loss: {best_loss:.4f}" if best_loss is not None else ""
            msg += f" | pseudo-loss={loss:.4f} {best_loss_str}"
        self.info(msg)

    def validation_early_stop(self, patience: int):
        """Log early stopping triggered."""
        self._flush()
        self.warning(f"Early stopping: no improvement for {patience} validations")

    def validation_complete(self):
        """Log validation complete - returns to normal training."""
        self._flush()

    # === Utility methods ===

    def _flush(self):
        """Flush stdout to ensure clean output."""
        if self._console:
            sys.stdout.flush()

    def separator(self, char: str = "-", length: int = 50):
        """Print a separator line."""
        self.info(char * length)

    def blank_line(self):
        """Print a blank line for readability."""
        self.info("")


# Global logger instance
_global_logger: Optional[TrainingLogger] = None


def configure_root_logger(level: int = logging.WARNING):
    """
    Configure the root logger to reduce noise from other modules.

    By default, sets root logger to WARNING to suppress INFO/DEBUG from
    modules that use logging.getLogger(__name__) directly.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root.handlers = []

    # Add a simple handler for warnings/errors
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter('%(name)s | %(levelname)s | %(message)s'))
    root.addHandler(handler)


def get_logger(
    name: str = "adam.training",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> TrainingLogger:
    """
    Get or create the global training logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console: Whether to output to console

    Returns:
        TrainingLogger instance
    """
    global _global_logger

    if _global_logger is None:
        # Configure root logger to suppress noise from other modules
        configure_root_logger(logging.WARNING)
        _global_logger = TrainingLogger(name, level, log_file, console)

    return _global_logger


def set_log_level(level: int):
    """Set the global logger level."""
    global _global_logger
    if _global_logger:
        _global_logger.logger.setLevel(level)
        for handler in _global_logger.logger.handlers:
            handler.setLevel(level)


def add_file_handler(log_file: Path):
    """Add a file handler to the global logger."""
    global _global_logger
    if _global_logger:
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(_global_logger.logger.level)
        file_handler.setFormatter(formatter)
        _global_logger.logger.addHandler(file_handler)


def reset_logger():
    """Reset the global logger instance. Call this if you need to reconfigure."""
    global _global_logger
    if _global_logger:
        # Clear handlers
        _global_logger.logger.handlers = []
    _global_logger = None
