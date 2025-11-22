#!/usr/bin/env python3
"""
Multi-Pass Training Module
Logica training multi-pass con progress tracking
"""

import time
from pathlib import Path
from typing import Optional, Callable

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
    from Utils.checkpoint import CheckpointManager
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..utils.checkpoint import CheckpointManager


class MultiPassTrainer:
    """Training multi-pass con auto-save e monitoring"""
    
    def __init__(self, 
                 brain: VectLLMBrain,
                 stats_collector: Optional[StatsCollector] = None,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            stats_collector: Collector per statistiche (opzionale)
            checkpoint_manager: Manager checkpoint (opzionale)
        """
        self.brain = brain
        self.stats = stats_collector or StatsCollector()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Callbacks
        self.on_pass_start: Optional[Callable] = None
        self.on_pass_end: Optional[Callable] = None
        self.on_checkpoint: Optional[Callable] = None
    
    def train(self,
              text: str,
              passes: int = 1,
              auto_save_every: Optional[int] = None,
              output_checkpoint: Optional[str] = None,
              verbose: bool = True) -> dict:
        """
        Train su testo con multi-pass.
        
        Args:
            text: Testo per training
            passes: Numero di passate
            auto_save_every: Auto-save ogni N passes
            output_checkpoint: Checkpoint finale
            verbose: Stampa progress
            
        Returns:
            Dict con statistiche finali
        """
        if verbose:
            print(f"🧠 Starting training...")
            print(f"   Text size: {len(text):,} chars")
            print(f"   Passes: {passes}")
            if auto_save_every:
                print(f"   Auto-save: every {auto_save_every} passes")
        
        total_tokens = 0
        
        for pass_num in range(1, passes + 1):
            # Callback pre-pass
            if self.on_pass_start:
                self.on_pass_start(pass_num, passes)
            
            if verbose:
                print(f"\n=== Pass {pass_num}/{passes} ===")
            
            pass_start = time.time()
            
            # Train
            processed = self.brain.train_on_text(text, passes=1)
            total_tokens += processed
            
            pass_time = time.time() - pass_start
            
            # Update stats
            brain_stats = self.brain.get_stats()
            self.stats.update(
                cycles=brain_stats['cycles'],
                tokens=brain_stats['tokens'],
                loss=brain_stats['loss'],
                vocab_size=brain_stats['vocab_words'],
                vocab_utilization=brain_stats['vocab_utilization']
            )
            
            # Print progress
            if verbose:
                tokens_per_sec = processed / pass_time if pass_time > 0 else 0
                print(f"   Processed: {processed:,} tokens")
                print(f"   Speed: {tokens_per_sec:.0f} tokens/sec")
                print(f"   Loss: {brain_stats['loss']:.4f}")
                print(f"   Perplexity: {brain_stats['perplexity']:.2f}")
                print(f"   Vocab: {brain_stats['vocab_words']:,} words")
                print(f"   Trend: {self.stats.get_loss_trend()}")
            
            # Callback post-pass
            if self.on_pass_end:
                self.on_pass_end(pass_num, passes, brain_stats)
            
            # Auto-save
            if auto_save_every and pass_num % auto_save_every == 0:
                ckpt_name = f"checkpoint_pass{pass_num}.ckpt"
                ckpt_path = self.checkpoint_manager.get_checkpoint_path(ckpt_name)
                
                if verbose:
                    print(f"   💾 Auto-saving: {ckpt_name}")
                
                self.brain.save_checkpoint(str(ckpt_path))
                self.stats.save_checkpoint_stats(str(ckpt_path))
                
                if self.on_checkpoint:
                    self.on_checkpoint(ckpt_path, pass_num)
        
        # Final save
        if output_checkpoint:
            if verbose:
                print(f"\n💾 Saving final checkpoint: {output_checkpoint}")
            
            ckpt_path = self.checkpoint_manager.get_checkpoint_path(output_checkpoint)
            self.brain.save_checkpoint(str(ckpt_path))
            self.stats.save_checkpoint_stats(str(ckpt_path))
        
        # Final stats
        final_stats = self.stats.get_summary()
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"   Total tokens: {total_tokens:,}")
            print(f"   Final loss: {final_stats['loss']:.4f}")
            print(f"   Final perplexity: {final_stats['perplexity']:.2f}")
            print(f"   Vocab size: {final_stats['vocab_size']:,}")
        
        return final_stats
    
    def train_from_file(self,
                       filepath: str,
                       **kwargs) -> dict:
        """
        Train da file di testo.
        
        Args:
            filepath: Path al file
            **kwargs: Parametri per train()
            
        Returns:
            Statistiche finali
        """
        text = Path(filepath).read_text()
        return self.train(text, **kwargs)
    
    def resume_training(self,
                       checkpoint_path: str,
                       text: str,
                       **kwargs) -> dict:
        """
        Riprendi training da checkpoint.
        
        Args:
            checkpoint_path: Checkpoint da caricare
            text: Nuovo testo per training
            **kwargs: Parametri per train()
            
        Returns:
            Statistiche finali
        """
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        self.brain.load_checkpoint(checkpoint_path)
        
        return self.train(text, **kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Multi-Pass Trainer Test ===\n")
    
    # Mock brain for testing (without GPU)
    class MockBrain:
        def __init__(self):
            self.cycles = 0
            self.tokens = 0
            self.loss = 5.0
            
        def train_on_text(self, text, passes=1):
            import time, random
            time.sleep(0.1)  # Simulate processing
            processed = len(text) * passes
            self.tokens += processed
            self.cycles += processed // 10
            self.loss = max(0.5, self.loss - random.uniform(0.01, 0.05))
            return processed
        
        def get_stats(self):
            return {
                'cycles': self.cycles,
                'tokens': self.tokens,
                'loss': self.loss,
                'perplexity': 2 ** self.loss,
                'vocab_words': 123,
                'vocab_utilization': 0.001,
            }
        
        def save_checkpoint(self, path):
            Path(path).touch()
            print(f"   Saved: {Path(path).name}")
    
    # Create trainer
    brain = MockBrain()
    trainer = MultiPassTrainer(brain)
    
    # Test training
    test_text = "This is a test text for multi-pass training. " * 10
    
    stats = trainer.train(
        text=test_text,
        passes=5,
        auto_save_every=2,
        output_checkpoint="test_final.ckpt",
        verbose=True
    )
    
    print(f"\n📊 Final Statistics:")
    print(f"   Tokens: {stats['tokens']:,}")
    print(f"   Loss: {stats['loss']:.4f}")
    
    print("\n✅ Trainer test complete!")
