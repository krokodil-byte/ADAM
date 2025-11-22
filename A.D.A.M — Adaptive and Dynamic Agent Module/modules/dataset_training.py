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
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Tuple, Any
import time

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
    from Utils.checkpoint import CheckpointManager
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..utils.checkpoint import CheckpointManager

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
        
        print(f"📂 Found {len(self.files)} files")
    
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
            print(f"⚠️  Error loading {filepath.name}: {e}")
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
        print(f"📊 Dataset format: {self.format}")

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
        print(f"📥 Loading dataset: {self.path.name}")

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
            print("   ⚠️  No data loaded")
            return

        # Detect columns
        input_col, output_col, text_col = self._detect_columns(raw_data)

        if input_col and output_col:
            print(f"   Columns: {input_col} → {output_col}")
        elif text_col:
            print(f"   Column: {text_col}")
        else:
            print("   ⚠️  Could not detect columns, using all text fields")

        # Format samples
        for row in raw_data:
            text = self._format_sample(row, input_col, output_col, text_col)
            if text:
                self.samples.append(text)

        print(f"   ✅ Loaded {len(self.samples)} samples")

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
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            dataset_loader: HFDatasetLoader
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
        """
        self.brain = brain
        self.loader = dataset_loader
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

    def train(self,
              passes: int = 1,
              auto_save_every: int = 1000,
              verbose: bool = True) -> Dict:
        """
        Train on HuggingFace dataset.

        Args:
            passes: Number of training passes
            auto_save_every: Auto-save every N samples
            verbose: Print progress

        Returns:
            Final statistics
        """
        dataset_stats = self.loader.get_stats()

        if verbose:
            print("🎓 Starting HuggingFace dataset training")
            print(f"   Samples: {dataset_stats['num_samples']}")
            print(f"   Format: {dataset_stats['format']}")
            print(f"   Avg length: {dataset_stats['avg_sample_length']:.0f} chars")
            print(f"   Passes: {passes}")
            print()

        total_tokens = 0
        samples_processed = 0
        start_time = time.time()

        try:
            for pass_num in range(1, passes + 1):
                if verbose:
                    print(f"=== Pass {pass_num}/{passes} ===")

                pass_start = time.time()
                pass_tokens = 0

                for idx, text in self.loader.iter_samples():
                    # Train on sample
                    processed = self.brain.train_on_text(text, passes=1)
                    pass_tokens += processed
                    total_tokens += processed

                    if pass_num == 1:
                        samples_processed += 1

                    # Progress update
                    if verbose and (idx + 1) % 100 == 0:
                        brain_stats = self.brain.get_stats()
                        elapsed = time.time() - pass_start
                        speed = pass_tokens / elapsed if elapsed > 0 else 0
                        print(f"   Sample {idx + 1}/{dataset_stats['num_samples']}: "
                              f"Loss={brain_stats['loss']:.4f}, "
                              f"Vocab={brain_stats['vocab_words']}, "
                              f"Speed={speed:.0f} tok/s")

                    # Auto-save
                    if (idx + 1) % auto_save_every == 0:
                        ckpt_name = f"hf_pass{pass_num}_sample{idx + 1}.ckpt"
                        ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                        if verbose:
                            print(f"   💾 Auto-saving: {ckpt_name}")
                        self.brain.save_checkpoint(str(ckpt_path))

                    # Update stats
                    brain_stats = self.brain.get_stats()
                    self.stats.update(
                        cycles=brain_stats['cycles'],
                        tokens=brain_stats['tokens'],
                        loss=brain_stats['loss'],
                        vocab_size=brain_stats['vocab_words']
                    )

                pass_time = time.time() - pass_start
                if verbose:
                    print(f"\n   ✅ Pass {pass_num} complete: {pass_tokens:,} tokens in {pass_time/60:.1f}m")

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")

        # Final stats
        elapsed = time.time() - start_time
        final_stats = self.stats.get_summary()

        if verbose:
            print("\n" + "=" * 70)
            print("🎉 HuggingFace dataset training complete!")
            print("=" * 70)
            print(f"Samples: {samples_processed}")
            print(f"Tokens: {total_tokens:,}")
            print(f"Time: {elapsed/60:.1f} minutes")
            print(f"Speed: {total_tokens/elapsed:.0f} tokens/sec")
            print(f"Loss: {final_stats['loss']:.4f}")
            print(f"Vocab: {final_stats['vocab_size']:,} words")
            print()

        return final_stats


class DatasetTrainer:
    """Training su dataset con progress tracking"""
    
    def __init__(self,
                 brain: VectLLMBrain,
                 dataset_loader: DatasetLoader,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            dataset_loader: Loader dataset
            stats_collector: Collector statistiche
            checkpoint_manager: Manager checkpoint
        """
        self.brain = brain
        self.loader = dataset_loader
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
    
    def train(self,
              passes: int = 1,
              auto_save_every: Optional[int] = None,
              skip_files: int = 0,
              verbose: bool = True) -> dict:
        """
        Train su intero dataset.
        
        Args:
            passes: Numero di passate sul dataset
            auto_save_every: Auto-save ogni N file
            skip_files: Salta primi N file (per resume)
            verbose: Stampa progress
            
        Returns:
            Dict con statistiche finali
        """
        dataset_stats = self.loader.get_stats()
        
        if verbose:
            print("🎓 Starting dataset training")
            print(f"   Files: {dataset_stats['num_files']}")
            print(f"   Size: {dataset_stats['total_size_mb']:.1f} MB")
            print(f"   Passes: {passes}")
            if auto_save_every:
                print(f"   Auto-save: every {auto_save_every} files")
            print()
        
        total_tokens = 0
        files_processed = 0
        
        for pass_num in range(1, passes + 1):
            if verbose:
                print(f"=== Pass {pass_num}/{passes} ===")
            
            pass_start = time.time()
            
            for file_idx, (filepath, content) in enumerate(self.loader.iter_files(), 1):
                # Skip if resuming
                if pass_num == 1 and file_idx <= skip_files:
                    continue
                
                if verbose:
                    print(f"\n📄 File {file_idx}/{dataset_stats['num_files']}: {filepath.name}")
                    print(f"   Size: {len(content):,} chars")
                
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
                    tokens_per_sec = processed / file_time if file_time > 0 else 0
                    print(f"   Processed: {processed:,} tokens in {file_time:.1f}s")
                    print(f"   Speed: {tokens_per_sec:.0f} tokens/sec")
                    print(f"   Loss: {brain_stats['loss']:.4f} {self.stats.get_loss_trend()}")
                    print(f"   Vocab: {brain_stats['vocab_words']:,} words")
                
                # Auto-save
                if auto_save_every and file_idx % auto_save_every == 0:
                    ckpt_name = f"dataset_pass{pass_num}_file{file_idx}.ckpt"
                    ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                    
                    if verbose:
                        print(f"   💾 Auto-saving: {ckpt_name}")
                    
                    self.brain.save_checkpoint(str(ckpt_path))
            
            pass_time = time.time() - pass_start
            
            if verbose:
                print(f"\n✅ Pass {pass_num} complete in {pass_time/60:.1f} minutes")
                print(f"   Files: {dataset_stats['num_files']}")
                print(f"   Tokens: {total_tokens:,}")
                print(f"   Avg speed: {total_tokens/pass_time:.0f} tokens/sec")
                print()
        
        # Final stats
        final_stats = self.stats.get_summary()
        
        if verbose:
            print("=" * 70)
            print("🎉 Dataset training complete!")
            print("=" * 70)
            print(f"Files processed: {files_processed}")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Final loss: {final_stats['loss']:.4f}")
            print(f"Final vocab: {final_stats['vocab_size']:,} words")
            print()
        
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
