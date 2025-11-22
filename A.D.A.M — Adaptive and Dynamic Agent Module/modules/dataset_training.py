#!/usr/bin/env python3
"""
Dataset Training Module
Caricamento e training su dataset di testo
"""

import os
from pathlib import Path
from typing import List, Optional, Iterator
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
