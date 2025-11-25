#!/usr/bin/env python3
"""
Dataset Training Module
Caricamento e training su dataset di testo

Supporta:
- Plain text files (.txt, .md)
- HuggingFace style datasets (JSONL, Parquet, CSV)
- Text-to-text pairs (instruction/response, input/output, etc.)
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Tuple, Any
import time

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
    from core.pipeline import PipelinedTrainer
    from core.config import TRAINING_CONFIG
    from Utils.checkpoint import CheckpointManager
    from modules.training_logger import get_logger
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..core.pipeline import PipelinedTrainer
    from ..core.config import TRAINING_CONFIG
    from ..utils.checkpoint import CheckpointManager
    from ..modules.training_logger import get_logger

# Try to import pyarrow for Parquet support
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Common column name mappings for text-to-text datasets
COLUMN_MAPPINGS = {
    # Input columns
    'input': ['input', 'instruction', 'prompt', 'question', 'context', 'source', 'text_input'],
    # Output columns
    'output': ['output', 'response', 'completion', 'answer', 'target', 'text_output', 'label'],
    # Single text column
    'text': ['text', 'content', 'sentence', 'document'],
}


class DatasetLoader:
    """Carica dataset di testo da file o directory"""
    
    def __init__(self, 
                 path: str,
                 extensions: List[str] = None,
                 encoding: str = 'utf-8',
                 max_file_size: int = 100 * 1024 * 1024):  # 100MB
        """
        Args:
            path: Path a file o directory
            extensions: Estensioni file da caricare (default: .txt)
            encoding: Encoding file
            max_file_size: Dimensione massima file (bytes)
        """
        self.path = Path(path)
        self.extensions = extensions or ['.txt', '.md', '.text']
        self.encoding = encoding
        self.max_file_size = max_file_size
        
        self.files: List[Path] = []
        self._scan_files()
    
    def _scan_files(self):
        """Scansiona file da caricare"""
        if self.path.is_file():
            self.files = [self.path]
        elif self.path.is_dir():
            for ext in self.extensions:
                self.files.extend(self.path.rglob(f"*{ext}"))
        else:
            raise ValueError(f"Path not found: {self.path}")
        
        # Filter by size
        self.files = [f for f in self.files
                     if f.stat().st_size <= self.max_file_size]

        logger = get_logger()
        logger.info(f"Found {len(self.files)} files")
    
    def load_file(self, filepath: Path) -> str:
        """
        Carica singolo file.
        
        Args:
            filepath: Path al file
            
        Returns:
            Contenuto file
        """
        try:
            with open(filepath, 'r', encoding=self.encoding, errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Error loading {filepath.name}: {e}")
            return ""
    
    def iter_files(self) -> Iterator[tuple]:
        """
        Itera sui file.
        
        Yields:
            (filepath, content)
        """
        for filepath in self.files:
            content = self.load_file(filepath)
            if content:
                yield filepath, content
    
    def get_total_size(self) -> int:
        """Dimensione totale dataset in bytes"""
        return sum(f.stat().st_size for f in self.files)
    
    def get_stats(self) -> dict:
        """Statistiche dataset"""
        total_size = self.get_total_size()

        return {
            'num_files': len(self.files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'average_file_size': total_size / len(self.files) if self.files else 0,
        }


class HFDatasetLoader:
    """
    Loader per dataset stile HuggingFace.

    Supporta:
    - JSONL (.jsonl, .json)
    - Parquet (.parquet)
    - CSV/TSV (.csv, .tsv)

    Auto-detecta colonne per text-to-text:
    - instruction/response
    - input/output
    - prompt/completion
    - question/answer
    """

    def __init__(self,
                 path: str,
                 input_column: str = None,
                 output_column: str = None,
                 text_column: str = None,
                 template: str = None,
                 max_samples: int = None,
                 encoding: str = 'utf-8'):
        """
        Args:
            path: Path al file dataset (JSONL, Parquet, CSV)
            input_column: Nome colonna input (auto-detect se None)
            output_column: Nome colonna output (auto-detect se None)
            text_column: Nome colonna testo singolo (per dataset non-paired)
            template: Template per formattare pairs (default: "{input}\n{output}")
            max_samples: Numero massimo di campioni da caricare
            encoding: Encoding file (per JSONL/CSV)
        """
        self.path = Path(path)
        self.input_column = input_column
        self.output_column = output_column
        self.text_column = text_column
        self.template = template or "{input}\n\n{output}"
        self.max_samples = max_samples
        self.encoding = encoding

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        # Detect format
        self.format = self._detect_format()
        self.logger = get_logger()
        self.logger.info(f"Dataset format: {self.format}")

        # Load and detect columns
        self.samples: List[str] = []
        self._load_dataset()

    def _detect_format(self) -> str:
        """Rileva formato dataset"""
        ext = self.path.suffix.lower()

        if ext in ['.jsonl', '.json']:
            return 'jsonl'
        elif ext == '.parquet':
            if not HAS_PARQUET:
                raise ImportError("Parquet support requires pyarrow: pip install pyarrow")
            return 'parquet'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.tsv':
            return 'tsv'
        else:
            # Try to detect from content
            with open(self.path, 'r', encoding=self.encoding) as f:
                first_line = f.readline().strip()
                if first_line.startswith('{'):
                    return 'jsonl'
                elif ',' in first_line:
                    return 'csv'
                elif '\t' in first_line:
                    return 'tsv'

            raise ValueError(f"Unknown dataset format: {self.path}")

    def _detect_columns(self, sample_data: List[Dict]) -> Tuple[str, str, str]:
        """
        Auto-detect column names from sample data.

        Returns:
            (input_col, output_col, text_col)
        """
        if not sample_data:
            return (None, None, None)

        keys = set(sample_data[0].keys())
        keys_lower = {k.lower(): k for k in keys}

        input_col = None
        output_col = None
        text_col = None

        # Find input column
        if self.input_column:
            input_col = self.input_column
        else:
            for candidate in COLUMN_MAPPINGS['input']:
                if candidate in keys_lower:
                    input_col = keys_lower[candidate]
                    break

        # Find output column
        if self.output_column:
            output_col = self.output_column
        else:
            for candidate in COLUMN_MAPPINGS['output']:
                if candidate in keys_lower:
                    output_col = keys_lower[candidate]
                    break

        # Find text column (for single-column datasets)
        if self.text_column:
            text_col = self.text_column
        else:
            for candidate in COLUMN_MAPPINGS['text']:
                if candidate in keys_lower:
                    text_col = keys_lower[candidate]
                    break

        return (input_col, output_col, text_col)

    def _format_sample(self, row: Dict, input_col: str, output_col: str, text_col: str) -> str:
        """Format a single sample for training"""
        # Text-to-text pair
        if input_col and output_col and input_col in row and output_col in row:
            input_text = str(row[input_col]).strip()
            output_text = str(row[output_col]).strip()
            return self.template.format(input=input_text, output=output_text)

        # Single text column
        if text_col and text_col in row:
            return str(row[text_col]).strip()

        # Fallback: concatenate all string values
        texts = [str(v) for v in row.values() if isinstance(v, str)]
        return '\n'.join(texts)

    def _load_jsonl(self) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(self.path, 'r', encoding=self.encoding) as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break
                try:
                    row = json.loads(line.strip())
                    if isinstance(row, dict):
                        data.append(row)
                except json.JSONDecodeError:
                    continue
        return data

    def _load_parquet(self) -> List[Dict]:
        """Load Parquet file"""
        table = pq.read_table(self.path)
        df = table.to_pandas()

        if self.max_samples:
            df = df.head(self.max_samples)

        return df.to_dict('records')

    def _load_csv(self, delimiter: str = ',') -> List[Dict]:
        """Load CSV/TSV file"""
        data = []
        with open(self.path, 'r', encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for i, row in enumerate(reader):
                if self.max_samples and i >= self.max_samples:
                    break
                data.append(row)
        return data

    def _load_dataset(self):
        """Load dataset and format samples"""
        self.logger.info(f"Loading dataset: {self.path.name}")

        # Load raw data
        if self.format == 'jsonl':
            raw_data = self._load_jsonl()
        elif self.format == 'parquet':
            raw_data = self._load_parquet()
        elif self.format == 'csv':
            raw_data = self._load_csv(',')
        elif self.format == 'tsv':
            raw_data = self._load_csv('\t')
        else:
            raw_data = []

        if not raw_data:
            self.logger.warning("No data loaded")
            return

        # Detect columns
        input_col, output_col, text_col = self._detect_columns(raw_data)

        if input_col and output_col:
            self.logger.info(f"  Columns: {input_col} -> {output_col}")
        elif text_col:
            self.logger.info(f"  Column: {text_col}")
        else:
            self.logger.warning("Could not detect columns, using all text fields")

        # Format samples
        for row in raw_data:
            text = self._format_sample(row, input_col, output_col, text_col)
            if text:
                self.samples.append(text)

        self.logger.info(f"  Loaded {len(self.samples)} samples")

    def iter_samples(self) -> Iterator[Tuple[int, str]]:
        """
        Iterate over samples.

        Yields:
            (index, formatted_text)
        """
        for i, sample in enumerate(self.samples):
            yield i, sample

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        total_chars = sum(len(s) for s in self.samples)

        return {
            'num_samples': len(self.samples),
            'total_chars': total_chars,
            'avg_sample_length': total_chars / len(self.samples) if self.samples else 0,
            'format': self.format,
        }


class HFDatasetTrainer:
    """Trainer per dataset HuggingFace style"""

    def __init__(self,
                 brain: VectLLMBrain,
                 dataset_loader: HFDatasetLoader,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 validation_split: float = None,
                 validation_frequency: int = None,
                 early_stopping_patience: int = None,
                 validate_per_pass: bool = True):
        """
        Args:
            brain: Istanza VectLLMBrain
            dataset_loader: HFDatasetLoader
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
            validation_split: Fraction of data to use for validation (default from config)
            validation_frequency: Validate every N samples (only if validate_per_pass=False)
            early_stopping_patience: Stop after N validations without improvement (default from config)
            validate_per_pass: If True, validate only at end of each pass; if False, validate every N samples
        """
        self.brain = brain
        self.loader = dataset_loader
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.logger = get_logger()

        # Validation settings
        self.validation_split = validation_split or TRAINING_CONFIG.VALIDATION_SPLIT
        self.validation_frequency = validation_frequency or TRAINING_CONFIG.VALIDATION_FREQUENCY
        self.early_stopping_patience = early_stopping_patience or TRAINING_CONFIG.EARLY_STOPPING_PATIENCE
        self.validate_per_pass = validate_per_pass

        # Validation state
        self.train_samples: List[str] = []
        self.val_samples: List[str] = []
        self.best_val_loss = float('inf')
        self.validations_without_improvement = 0

        # Split data
        self._split_data()

    def _split_data(self):
        """Split samples into train/validation sets."""
        all_samples = list(self.loader.samples)

        if len(all_samples) < TRAINING_CONFIG.MIN_VALIDATION_SAMPLES:
            # Not enough samples for validation
            self.train_samples = all_samples
            self.val_samples = []
            self.logger.info(f"Not enough samples for validation ({len(all_samples)} < {TRAINING_CONFIG.MIN_VALIDATION_SAMPLES})")
            return

        # Shuffle for random split
        random.shuffle(all_samples)

        # Split
        val_size = int(len(all_samples) * self.validation_split)
        val_size = max(val_size, 1)  # At least 1 validation sample

        self.val_samples = all_samples[:val_size]
        self.train_samples = all_samples[val_size:]

        self.logger.info(f"Data split: {len(self.train_samples)} train, {len(self.val_samples)} validation")

    def _run_validation(self) -> float:
        """
        Run validation on validation set.

        Returns:
            Average validation loss
        """
        if not self.val_samples:
            return -1.0

        # Defer GPU syncs during validation to avoid contention
        self.brain.begin_validation()

        total_loss = 0.0
        count = 0

        for sample in self.val_samples:
            loss = self.brain.validate_on_text(sample)
            if loss >= 0:
                total_loss += loss
                count += 1

        # Resume normal sync after validation
        self.brain.end_validation()

        if count == 0:
            return -1.0

        avg_loss = total_loss / count

        # Update stats
        self.stats.metrics.validation_loss = avg_loss
        self.stats.metrics.validation_history.append((time.time(), avg_loss))

        # Check for improvement
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.stats.metrics.best_validation_loss = avg_loss
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1
            self.stats.metrics.validations_without_improvement = self.validations_without_improvement

        return avg_loss

    def train(self,
              passes: int = 1,
              auto_save_every: int = 1000,
              verbose: bool = True,
              use_pipeline: bool = True,
              prefetch_size: int = 3,
              enable_validation: bool = True,
              enable_early_stopping: bool = True) -> Dict:
        """
        Train on HuggingFace dataset.

        Args:
            passes: Number of training passes
            auto_save_every: Auto-save every N samples
            verbose: Print progress
            use_pipeline: Use pipelined training for CPU/GPU overlap
            prefetch_size: Number of batches to prefetch (if use_pipeline)
            enable_validation: Run validation during training
            enable_early_stopping: Stop if validation doesn't improve

        Returns:
            Final statistics
        """
        dataset_stats = self.loader.get_stats()
        train_samples_count = len(self.train_samples)

        if verbose:
            self.logger.training_start(
                "HuggingFace Dataset",
                samples=train_samples_count,
                validation_samples=len(self.val_samples),
                format=dataset_stats['format'],
                avg_length=f"{dataset_stats['avg_sample_length']:.0f} chars",
                passes=passes,
                pipeline='enabled' if use_pipeline else 'disabled',
                validation='enabled' if enable_validation and self.val_samples else 'disabled'
            )

        total_tokens = 0
        samples_processed = 0
        start_time = time.time()
        early_stopped = False

        try:
            for pass_num in range(1, passes + 1):
                if verbose:
                    self.logger.pass_start(pass_num, passes)

                pass_start = time.time()
                pass_tokens = 0

                if use_pipeline:
                    # Pipelined training - overlap CPU encoding with GPU training
                    pass_tokens = self._train_pass_pipelined(
                        pass_num, train_samples_count, auto_save_every, verbose, prefetch_size,
                        enable_validation, enable_early_stopping
                    )
                    if pass_num == 1:
                        samples_processed = train_samples_count

                    # Check early stopping
                    if enable_early_stopping and self.validations_without_improvement >= self.early_stopping_patience:
                        if verbose:
                            self.logger.validation_early_stop(self.early_stopping_patience)
                        early_stopped = True
                        total_tokens += pass_tokens
                        break
                else:
                    # Sequential training (legacy)
                    for idx, text in enumerate(self.train_samples):
                        # Train on sample
                        processed = self.brain.train_on_text(text, passes=1)
                        pass_tokens += processed
                        total_tokens += processed

                        if pass_num == 1:
                            samples_processed += 1

                        # Progress update
                        if verbose and (idx + 1) % 100 == 0:
                            brain_stats = self.brain.get_stats()
                            self.logger.sample_progress(
                                idx + 1, train_samples_count,
                                pass_tokens, brain_stats['loss']
                            )

                        # Validation check
                        if enable_validation and self.val_samples and (idx + 1) % self.validation_frequency == 0:
                            if verbose:
                                self.logger.validation_start(len(self.val_samples))
                            val_loss = self._run_validation()
                            if verbose:
                                improved = (self.validations_without_improvement == 0)
                                self.logger.validation_result(val_loss, self.best_val_loss, improved)
                                self.logger.validation_complete()

                            # Early stopping check
                            if enable_early_stopping and self.validations_without_improvement >= self.early_stopping_patience:
                                if verbose:
                                    self.logger.validation_early_stop(self.early_stopping_patience)
                                early_stopped = True
                                break

                        # Auto-save
                        if (idx + 1) % auto_save_every == 0:
                            ckpt_name = f"hf_pass{pass_num}_sample{idx + 1}.ckpt"
                            ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                            if verbose:
                                self.logger.checkpoint_save(ckpt_name)
                            self.brain.save_checkpoint(str(ckpt_path))

                        # Update stats
                        brain_stats = self.brain.get_stats()
                        self.stats.update(
                            cycles=brain_stats['cycles'],
                            tokens=brain_stats['tokens'],
                            loss=brain_stats['loss'],
                            vocab_size=brain_stats['vocab_words']
                        )

                    if early_stopped:
                        break

                total_tokens += pass_tokens
                pass_time = time.time() - pass_start
                if verbose:
                    self.logger.pass_end(pass_num, pass_tokens, pass_time)

                if early_stopped:
                    break

        except KeyboardInterrupt:
            self.logger.interrupted()

        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()

        # Add validation stats
        final_stats['validation_loss'] = self.stats.metrics.validation_loss
        final_stats['best_validation_loss'] = self.best_val_loss
        final_stats['early_stopped'] = early_stopped

        if verbose:
            end_kwargs = {
                'samples': samples_processed,
                'tokens': total_tokens,
                'speed': total_tokens/elapsed if elapsed > 0 else 0,
                'loss': final_stats['loss'],
                'vocab': final_stats['vocab_size']
            }
            if self.val_samples:
                end_kwargs['validation_loss'] = self.stats.metrics.validation_loss
                end_kwargs['best_val_loss'] = self.best_val_loss
            if early_stopped:
                end_kwargs['status'] = 'early_stopped'

            self.logger.training_end(**end_kwargs)

        return final_stats

    def _train_pass_pipelined(self,
                               pass_num: int,
                               train_samples_count: int,
                               auto_save_every: int,
                               verbose: bool,
                               prefetch_size: int,
                               enable_validation: bool = True,
                               enable_early_stopping: bool = True) -> int:
        """
        Train one pass using pipelined CPU/GPU overlap.

        Args:
            pass_num: Current pass number
            train_samples_count: Number of training samples
            auto_save_every: Auto-save interval
            verbose: Verbose output
            prefetch_size: Prefetch queue size
            enable_validation: Run validation during training
            enable_early_stopping: Stop if validation doesn't improve

        Returns:
            Tokens processed in this pass
        """
        # Begin deferred sync - batch all vocab syncs until end of pass
        self.brain.begin_deferred_sync()

        # Create pipelined trainer
        trainer = PipelinedTrainer(self.brain, prefetch_size=prefetch_size)

        # Use train samples only
        texts = self.train_samples

        if not texts:
            return 0

        # Progress callback
        processed_count = [0]
        last_save_idx = [0]
        pass_start = time.time()

        def progress_callback(batch_num: int, tokens: int):
            processed_count[0] += 1
            idx = processed_count[0]

            # Progress update
            if verbose and idx % 100 == 0:
                brain_stats = self.brain.get_stats()
                self.logger.sample_progress(
                    idx, train_samples_count,
                    tokens, brain_stats['loss']
                )

            # Validation check (only if validate_per_pass is False)
            if (enable_validation and self.val_samples and
                not self.validate_per_pass and
                idx % self.validation_frequency == 0):
                if verbose:
                    self.logger.validation_start(len(self.val_samples))
                val_loss = self._run_validation()
                if verbose:
                    improved = (self.validations_without_improvement == 0)
                    self.logger.validation_result(val_loss, self.best_val_loss, improved)
                    self.logger.validation_complete()

            # Auto-save check
            if idx % auto_save_every == 0 and idx > last_save_idx[0]:
                last_save_idx[0] = idx
                ckpt_name = f"hf_pass{pass_num}_sample{idx}.ckpt"
                ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                if verbose:
                    self.logger.checkpoint_save(ckpt_name)
                self.brain.save_checkpoint(str(ckpt_path))

            # Update stats
            brain_stats = self.brain.get_stats()
            self.stats.update(
                cycles=brain_stats['cycles'],
                tokens=brain_stats['tokens'],
                loss=brain_stats['loss'],
                vocab_size=brain_stats['vocab_words']
            )

        # Run pipelined training
        pass_tokens = trainer.train_texts(iter(texts), progress_callback)

        # Validate at end of pass (if validate_per_pass is True)
        if enable_validation and self.val_samples and self.validate_per_pass:
            if verbose:
                self.logger.validation_start(len(self.val_samples))
            val_loss = self._run_validation()
            if verbose:
                improved = (self.validations_without_improvement == 0)
                self.logger.validation_result(val_loss, self.best_val_loss, improved)
                self.logger.validation_complete()

        # End deferred sync - execute all batched vocab syncs
        self.brain.end_deferred_sync()

        # Get pipeline stats
        pipeline_stats = trainer.get_stats()
        if verbose:
            self.logger.pipeline_stats(
                pipeline_stats['throughput_tok_s'],
                pipeline_stats.get('gpu_utilization', 0)
            )

        return pass_tokens


class DatasetTrainer:
    """Training su dataset con progress tracking"""

    def __init__(self,
                 brain: VectLLMBrain,
                 dataset_loader: DatasetLoader,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 validation_split: float = None,
                 validation_frequency: int = None,
                 early_stopping_patience: int = None,
                 validate_per_pass: bool = True):
        """
        Args:
            brain: Istanza VectLLMBrain
            dataset_loader: Loader dataset
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
            validation_split: Fraction of data to use for validation
            validation_frequency: Validate every N files (only if validate_per_pass=False)
            early_stopping_patience: Stop after N validations without improvement
            validate_per_pass: If True, validate only at end of each pass; if False, validate every N files
        """
        self.brain = brain
        self.loader = dataset_loader
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.logger = get_logger()

        # Validation settings
        self.validation_split = validation_split or TRAINING_CONFIG.VALIDATION_SPLIT
        self.validation_frequency = validation_frequency or TRAINING_CONFIG.VALIDATION_FREQUENCY
        self.early_stopping_patience = early_stopping_patience or TRAINING_CONFIG.EARLY_STOPPING_PATIENCE
        self.validate_per_pass = validate_per_pass

        # Validation state
        self.train_files: List[Path] = []
        self.val_files: List[Path] = []
        self.best_val_loss = float('inf')
        self.validations_without_improvement = 0

        # Split files
        self._split_files()

    def _split_files(self):
        """Split files into train/validation sets."""
        all_files = list(self.loader.files)

        if len(all_files) < TRAINING_CONFIG.MIN_VALIDATION_SAMPLES:
            # Not enough files for validation
            self.train_files = all_files
            self.val_files = []
            self.logger.info(f"Not enough files for validation ({len(all_files)} < {TRAINING_CONFIG.MIN_VALIDATION_SAMPLES})")
            return

        # Shuffle for random split
        random.shuffle(all_files)

        # Split
        val_size = int(len(all_files) * self.validation_split)
        val_size = max(val_size, 1)  # At least 1 validation file

        self.val_files = all_files[:val_size]
        self.train_files = all_files[val_size:]

        self.logger.info(f"Data split: {len(self.train_files)} train files, {len(self.val_files)} validation files")

    def _run_validation(self) -> float:
        """
        Run validation on validation files.

        Returns:
            Average validation loss
        """
        if not self.val_files:
            return -1.0

        # Defer GPU syncs during validation to avoid contention
        self.brain.begin_validation()

        total_loss = 0.0
        count = 0

        for filepath in self.val_files:
            content = self.loader.load_file(filepath)
            if content:
                loss = self.brain.validate_on_text(content)
                if loss >= 0:
                    total_loss += loss
                    count += 1

        # Resume normal sync after validation
        self.brain.end_validation()

        if count == 0:
            return -1.0

        avg_loss = total_loss / count

        # Update stats
        self.stats.metrics.validation_loss = avg_loss
        self.stats.metrics.validation_history.append((time.time(), avg_loss))

        # Check for improvement
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.stats.metrics.best_validation_loss = avg_loss
            self.validations_without_improvement = 0
        else:
            self.validations_without_improvement += 1
            self.stats.metrics.validations_without_improvement = self.validations_without_improvement

        return avg_loss

    def train(self,
              passes: int = 1,
              auto_save_every: Optional[int] = None,
              skip_files: int = 0,
              verbose: bool = True,
              enable_validation: bool = True,
              enable_early_stopping: bool = True) -> dict:
        """
        Train su intero dataset.

        Args:
            passes: Numero di passate sul dataset
            auto_save_every: Auto-save ogni N file
            skip_files: Salta primi N file (per resume)
            verbose: Stampa progress
            enable_validation: Run validation during training
            enable_early_stopping: Stop if validation doesn't improve

        Returns:
            Dict con statistiche finali
        """
        dataset_stats = self.loader.get_stats()
        train_files_count = len(self.train_files)

        if verbose:
            self.logger.training_start(
                "Dataset",
                files=train_files_count,
                validation_files=len(self.val_files),
                size=f"{dataset_stats['total_size_mb']:.1f} MB",
                passes=passes,
                auto_save=f"every {auto_save_every} files" if auto_save_every else "disabled",
                validation='enabled' if enable_validation and self.val_files else 'disabled'
            )

        total_tokens = 0
        files_processed = 0
        start_time = time.time()
        early_stopped = False

        try:
            for pass_num in range(1, passes + 1):
                if verbose:
                    self.logger.pass_start(pass_num, passes)

                pass_start = time.time()

                # Begin deferred sync - batch all vocab syncs until end of pass
                self.brain.begin_deferred_sync()

                for file_idx, filepath in enumerate(self.train_files, 1):
                    # Skip if resuming
                    if pass_num == 1 and file_idx <= skip_files:
                        continue

                    content = self.loader.load_file(filepath)
                    if not content:
                        continue

                    if verbose:
                        self.logger.file_start(file_idx, train_files_count, filepath.name, len(content))

                    # Train on file
                    file_start = time.time()
                    processed = self.brain.train_on_text(content, passes=1)
                    file_time = time.time() - file_start

                    total_tokens += processed
                    files_processed += 1

                    # Update stats
                    brain_stats = self.brain.get_stats()
                    self.stats.update(
                        cycles=brain_stats['cycles'],
                        tokens=brain_stats['tokens'],
                        loss=brain_stats['loss'],
                        vocab_size=brain_stats['vocab_words']
                    )

                    if verbose:
                        self.logger.file_end(processed, file_time, brain_stats['loss'], brain_stats['vocab_words'])

                    # Validation check (only if validate_per_pass is False)
                    if (enable_validation and self.val_files and
                        not self.validate_per_pass and
                        file_idx % self.validation_frequency == 0):
                        if verbose:
                            self.logger.validation_start(len(self.val_files))
                        val_loss = self._run_validation()
                        if verbose:
                            improved = (self.validations_without_improvement == 0)
                            self.logger.validation_result(val_loss, self.best_val_loss, improved)
                            self.logger.validation_complete()

                        # Early stopping check
                        if enable_early_stopping and self.validations_without_improvement >= self.early_stopping_patience:
                            if verbose:
                                self.logger.validation_early_stop(self.early_stopping_patience)
                            early_stopped = True
                            break

                    # Auto-save
                    if auto_save_every and file_idx % auto_save_every == 0:
                        ckpt_name = f"dataset_pass{pass_num}_file{file_idx}.ckpt"
                        ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)

                        if verbose:
                            self.logger.checkpoint_save(ckpt_name)

                        self.brain.save_checkpoint(str(ckpt_path))

                # Validate at end of pass (if validate_per_pass is True)
                if enable_validation and self.val_files and self.validate_per_pass and not early_stopped:
                    if verbose:
                        self.logger.validation_start(len(self.val_files))
                    val_loss = self._run_validation()
                    if verbose:
                        improved = (self.validations_without_improvement == 0)
                        self.logger.validation_result(val_loss, self.best_val_loss, improved)
                        self.logger.validation_complete()

                    # Early stopping check
                    if enable_early_stopping and self.validations_without_improvement >= self.early_stopping_patience:
                        if verbose:
                            self.logger.validation_early_stop(self.early_stopping_patience)
                        early_stopped = True

                # End deferred sync - execute all batched vocab syncs
                self.brain.end_deferred_sync()

                if early_stopped:
                    break

                pass_time = time.time() - pass_start

                if verbose:
                    self.logger.pass_end(pass_num, total_tokens, pass_time)

        except KeyboardInterrupt:
            self.logger.interrupted()

        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()

        # Add validation stats
        final_stats['validation_loss'] = self.stats.metrics.validation_loss
        final_stats['best_validation_loss'] = self.best_val_loss
        final_stats['early_stopped'] = early_stopped

        if verbose:
            end_kwargs = {
                'files': files_processed,
                'tokens': total_tokens,
                'loss': final_stats['loss'],
                'vocab': final_stats['vocab_size']
            }
            if self.val_files:
                end_kwargs['validation_loss'] = self.stats.metrics.validation_loss
                end_kwargs['best_val_loss'] = self.best_val_loss
            if early_stopped:
                end_kwargs['status'] = 'early_stopped'

            self.logger.training_end(**end_kwargs)

        return final_stats


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Dataset Training Test ===\n")
    
    # Create test files
    test_dir = Path("./test_dataset")
    test_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        test_file = test_dir / f"test{i}.txt"
        test_file.write_text(f"Test file {i} content. " * 100)
    
    print("Created test dataset")
    
    # Test loader
    loader = DatasetLoader(str(test_dir))
    stats = loader.get_stats()
    
    print(f"\nDataset stats:")
    print(f"  Files: {stats['num_files']}")
    print(f"  Size: {stats['total_size_mb']:.3f} MB")
    
    # Test iteration
    print(f"\nLoading files:")
    for filepath, content in loader.iter_files():
        print(f"  {filepath.name}: {len(content)} chars")
    
    # Cleanup
    for f in test_dir.glob("*.txt"):
        f.unlink()
    test_dir.rmdir()
    
    print("\n✅ Dataset training test complete!")
