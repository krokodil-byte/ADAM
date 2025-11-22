#!/usr/bin/env python3
"""
Checkpoint Management
Gestione salvataggio/caricamento checkpoint
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Informazioni checkpoint"""
    version: int
    magic: str
    char_vocab_size: int
    current_word_vocab_size: int
    max_word_vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_clusters: int
    total_cycles: int
    total_tokens: int
    timestamp: int
    learning_rate: float
    momentum: float
    current_loss: float


class CheckpointManager:
    """Gestione checkpoint con metadata"""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Args:
            checkpoint_dir: Directory checkpoint (default: ./checkpoints)
        """
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, name: str) -> Path:
        """Ottieni path completo checkpoint"""
        if not name.endswith('.ckpt'):
            name = f"{name}.ckpt"
        return self.checkpoint_dir / name
    
    def get_vocab_path(self, checkpoint_path: Path) -> Path:
        """Ottieni path vocabolario associato"""
        return checkpoint_path.with_suffix('.vocab')
    
    def get_metadata_path(self, checkpoint_path: Path) -> Path:
        """Ottieni path metadata JSON"""
        return checkpoint_path.with_suffix('.json')
    
    def save_metadata(self, checkpoint_path: Path, info: CheckpointInfo, extra: Optional[Dict] = None):
        """
        Salva metadata checkpoint in JSON.
        
        Args:
            checkpoint_path: Path al checkpoint
            info: Informazioni checkpoint
            extra: Metadata extra opzionale
        """
        metadata_path = self.get_metadata_path(checkpoint_path)
        
        data = asdict(info)
        if extra:
            data['extra'] = extra
        
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metadata(self, checkpoint_path: Path) -> Optional[Dict]:
        """
        Carica metadata checkpoint.
        
        Args:
            checkpoint_path: Path al checkpoint
            
        Returns:
            Dict con metadata o None se non esiste
        """
        metadata_path = self.get_metadata_path(checkpoint_path)
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_checkpoints(self) -> list:
        """
        Lista tutti i checkpoint disponibili.
        
        Returns:
            Lista di (path, metadata)
        """
        checkpoints = []
        
        for ckpt_path in sorted(self.checkpoint_dir.glob("*.ckpt")):
            metadata = self.load_metadata(ckpt_path)
            checkpoints.append((ckpt_path, metadata))
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Ottieni checkpoint più recente.
        
        Returns:
            Path al checkpoint o None
        """
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def delete_checkpoint(self, checkpoint_path: Path, delete_vocab: bool = True, delete_metadata: bool = True):
        """
        Elimina checkpoint e file associati.
        
        Args:
            checkpoint_path: Path al checkpoint
            delete_vocab: Elimina anche vocabolario
            delete_metadata: Elimina anche metadata
        """
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path.name}")
        
        if delete_vocab:
            vocab_path = self.get_vocab_path(checkpoint_path)
            if vocab_path.exists():
                vocab_path.unlink()
                logger.info(f"Deleted vocab: {vocab_path.name}")
            
            # Also delete .freq file
            freq_path = vocab_path.with_suffix('.freq')
            if freq_path.exists():
                freq_path.unlink()
        
        if delete_metadata:
            metadata_path = self.get_metadata_path(checkpoint_path)
            if metadata_path.exists():
                metadata_path.unlink()
    
    def cleanup_old_checkpoints(self, keep_latest: int = 5):
        """
        Elimina vecchi checkpoint mantenendo solo gli ultimi N.
        
        Args:
            keep_latest: Numero di checkpoint da mantenere
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for ckpt in checkpoints[keep_latest:]:
            self.delete_checkpoint(ckpt)
    
    def print_checkpoint_info(self, checkpoint_path: Path):
        """Stampa informazioni checkpoint"""
        metadata = self.load_metadata(checkpoint_path)
        
        if not metadata:
            print(f"⚠️  No metadata for: {checkpoint_path.name}")
            return
        
        print(f"\n📦 Checkpoint: {checkpoint_path.name}")
        print(f"   Version: {metadata.get('version', 'unknown')}")
        print(f"   Cycles: {metadata.get('total_cycles', 0):,}")
        print(f"   Tokens: {metadata.get('total_tokens', 0):,}")
        print(f"   Loss: {metadata.get('current_loss', 0):.4f}")
        print(f"   Vocab: {metadata.get('current_word_vocab_size', 0):,} words")
        print(f"   Size: {checkpoint_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Vocab info
        vocab_path = self.get_vocab_path(checkpoint_path)
        if vocab_path.exists():
            print(f"   Vocab file: {vocab_path.name} ({vocab_path.stat().st_size / 1024:.1f} KB)")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Checkpoint Manager Test ===\n")
    
    manager = CheckpointManager(Path("./test_checkpoints"))
    
    # Create test checkpoint info
    info = CheckpointInfo(
        version=3,
        magic="VECTLLM3",
        char_vocab_size=256,
        current_word_vocab_size=123,
        max_word_vocab_size=100000,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        num_clusters=256,
        total_cycles=10000,
        total_tokens=50000,
        timestamp=1234567890,
        learning_rate=0.0001,
        momentum=0.9,
        current_loss=2.345
    )
    
    # Test save/load metadata
    ckpt_path = manager.get_checkpoint_path("test_checkpoint")
    ckpt_path.touch()  # Create dummy file
    
    print("Saving metadata...")
    manager.save_metadata(ckpt_path, info, extra={'note': 'test checkpoint'})
    
    print("Loading metadata...")
    loaded = manager.load_metadata(ckpt_path)
    print(f"   Loaded: {loaded['total_cycles']} cycles")
    
    # Print info
    manager.print_checkpoint_info(ckpt_path)
    
    # Cleanup
    manager.delete_checkpoint(ckpt_path)
    manager.checkpoint_dir.rmdir()
    
    print("\n✅ Checkpoint manager test complete!")
