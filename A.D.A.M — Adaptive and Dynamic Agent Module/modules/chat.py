#!/usr/bin/env python3
"""
Interactive Chat Module
Modalità chat interattiva con VectLLM
"""

import sys
import threading
import queue
from typing import Optional, List

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
    from core.pipeline import AsyncBatchLoader
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector
    from ..core.pipeline import AsyncBatchLoader


class InteractiveChat:
    """Chat interattiva con il modello"""

    def __init__(self,
                 brain: VectLLMBrain,
                 stats_collector: Optional[StatsCollector] = None,
                 async_training: bool = True):
        """
        Args:
            brain: Istanza VectLLMBrain
            stats_collector: Collector statistiche (opzionale)
            async_training: Usa training asincrono in background
        """
        self.brain = brain
        self.stats = stats_collector or StatsCollector()
        self.history: List[str] = []

        # Async background training
        self.async_training = async_training
        self.training_queue: queue.Queue = queue.Queue()
        self.training_thread: Optional[threading.Thread] = None
        self.training_running = False
        self.pending_training = 0  # Count of messages being trained

    def _start_background_training(self):
        """Start background training thread"""
        if self.training_running:
            return

        self.training_running = True
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()

    def _stop_background_training(self):
        """Stop background training thread"""
        self.training_running = False
        self.training_queue.put(None)  # Signal stop

        if self.training_thread:
            self.training_thread.join(timeout=2.0)

    def _training_worker(self):
        """Background thread for async training"""
        while self.training_running:
            try:
                text = self.training_queue.get(timeout=0.1)

                if text is None:
                    break

                # Train on text
                self.brain.train_on_text(text, passes=1)
                self.pending_training = max(0, self.pending_training - 1)

                # Update stats
                brain_stats = self.brain.get_stats()
                self.stats.update(
                    cycles=brain_stats['cycles'],
                    tokens=brain_stats['tokens'],
                    loss=brain_stats['loss'],
                    reward=brain_stats.get('reward'),
                    topk_reward=brain_stats.get('topk_reward'),
                    venn_reward=brain_stats.get('venn_reward'),
                    vocab_size=brain_stats['vocab_words']
                )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n⚠️  Background training error: {e}")
                self.pending_training = max(0, self.pending_training - 1)
                continue

    def _queue_training(self, text: str):
        """Queue text for background training"""
        if self.async_training:
            self.pending_training += 1
            self.training_queue.put(text)
        else:
            # Synchronous training
            self.brain.train_on_text(text, passes=1)
        
    def start(self):
        """Avvia sessione chat interattiva"""
        print("=" * 70)
        print("🤖 VectLLM Interactive Chat")
        print("=" * 70)
        print()
        print("Commands:")
        print("  /stats    - Show statistics")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear history")
        print("  /save     - Save checkpoint")
        print("  /quit     - Exit chat")
        print()
        print("Type your message and press Enter")
        if self.async_training:
            print("(Background training enabled)")
        print("=" * 70)
        print()

        # Start background training if enabled
        if self.async_training:
            self._start_background_training()

        try:
            while True:
                try:
                    # Show pending training indicator
                    prompt = "You> "
                    if self.pending_training > 0:
                        prompt = f"You ({self.pending_training} training)> "

                    # Get user input
                    user_input = input(prompt).strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith('/'):
                        if self._handle_command(user_input):
                            break  # Exit if quit command
                        continue

                    # Process user input
                    self._process_message(user_input)

                except KeyboardInterrupt:
                    print("\n\n👋 Chat interrupted")
                    break
                except EOFError:
                    print("\n\n👋 Chat ended")
                    break
        finally:
            # Stop background training
            if self.async_training:
                self._stop_background_training()
    
    def _handle_command(self, command: str) -> bool:
        """
        Gestisce comandi speciali.
        
        Returns:
            True se deve uscire
        """
        cmd = command.lower()
        
        if cmd == '/quit' or cmd == '/exit' or cmd == '/q':
            print("\n👋 Goodbye!")
            return True
        
        elif cmd == '/stats':
            self._show_stats()
        
        elif cmd == '/history':
            self._show_history()
        
        elif cmd == '/clear':
            self.history.clear()
            print("✅ History cleared")
        
        elif cmd.startswith('/save'):
            parts = cmd.split()
            filename = parts[1] if len(parts) > 1 else "chat_checkpoint.ckpt"
            self._save_checkpoint(filename)
        
        else:
            print(f"❌ Unknown command: {command}")
            print("   Type /quit to exit")
        
        return False
    
    def _process_message(self, message: str):
        """
        Processa messaggio utente e genera risposta.

        Args:
            message: Messaggio utente
        """
        # Add to history
        self.history.append(f"User: {message}")

        # Queue message for background training (non-blocking)
        self._queue_training(message)

        # Generate response (happens immediately while training runs in background)
        response = self._generate_response(message)

        self.history.append(f"Bot: {response}")
        print(f"Bot> {response}")
        print()

        # Update stats (may not reflect latest training if async)
        if not self.async_training:
            brain_stats = self.brain.get_stats()
            self.stats.update(
                cycles=brain_stats['cycles'],
                tokens=brain_stats['tokens'],
                loss=brain_stats['loss'],
                vocab_size=brain_stats['vocab_words']
            )
    
    def _generate_response(self, user_message: str) -> str:
        """
        Genera risposta usando il modello con continuation bias.

        La generazione si ferma automaticamente quando la confidenza
        del modello scende sotto una soglia (continuation bias).

        Args:
            user_message: Messaggio utente

        Returns:
            Risposta generata
        """
        # Use the brain's generate_text with continuation bias
        response = self.brain.generate_text(user_message)

        # If generation failed or empty, provide fallback
        if not response or len(response.strip()) == 0:
            stats = self.brain.get_stats()
            return f"[Training... Loss: {stats['loss']:.3f}, Vocab: {stats['vocab_words']} words]"

        return response.strip()
    
    def _show_stats(self):
        """Mostra statistiche correnti"""
        print()
        print("📊 Statistics")
        print("-" * 70)
        self.stats.print_stats(prefix="  ")
        print()
    
    def _show_history(self):
        """Mostra cronologia conversazione"""
        print()
        print("📜 Conversation History")
        print("-" * 70)
        
        if not self.history:
            print("  (empty)")
        else:
            for i, msg in enumerate(self.history[-20:], 1):  # Last 20
                print(f"  {msg}")
        
        print()
    
    def _save_checkpoint(self, filename: str):
        """Salva checkpoint"""
        print(f"💾 Saving checkpoint: {filename}")
        success = self.brain.save_checkpoint(filename)
        
        if success:
            print("✅ Checkpoint saved")
        else:
            print("❌ Failed to save checkpoint")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Interactive Chat Test ===\n")
    
    # Mock brain for testing
    class MockBrain:
        def __init__(self):
            self.cycles = 0
            self.tokens = 0
            self.loss = 3.5
            
        def train_on_text(self, text, passes=1):
            self.tokens += len(text)
            self.cycles += 10
            self.loss = max(0.5, self.loss - 0.05)
            return len(text)
        
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
            print(f"   Saved: {path}")
            return True
    
    # Simulate chat with mock
    print("Chat module loaded")
    print("To test interactively, run:")
    print("  python3 cli/vectllm.py chat")
    print()
    
    # Test command parsing
    chat = InteractiveChat(MockBrain())
    
    test_commands = ['/stats', '/history', '/clear', '/save test.ckpt']
    print("Testing commands:")
    for cmd in test_commands:
        print(f"  Command: {cmd}")
        chat._handle_command(cmd)
        print()
    
    print("✅ Chat module test complete!")
