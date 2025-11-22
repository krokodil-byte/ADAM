#!/usr/bin/env python3
"""
Interactive Chat Module
Modalità chat interattiva con VectLLM
"""

import sys
from typing import Optional, List

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.stats import StatsCollector
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.stats import StatsCollector


class InteractiveChat:
    """Chat interattiva con il modello"""
    
    def __init__(self, 
                 brain: VectLLMBrain,
                 stats_collector: Optional[StatsCollector] = None):
        """
        Args:
            brain: Istanza VectLLMBrain
            stats_collector: Collector statistiche (opzionale)
        """
        self.brain = brain
        self.stats = stats_collector or StatsCollector()
        self.history: List[str] = []
        
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
        print("=" * 70)
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("You> ").strip()
                
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
        
        # Train on user message
        self.brain.train_on_text(message, passes=1)
        
        # TODO: Implement actual generation
        # For now, just acknowledge
        response = self._generate_response(message)
        
        self.history.append(f"Bot: {response}")
        print(f"Bot> {response}")
        print()
        
        # Update stats
        brain_stats = self.brain.get_stats()
        self.stats.update(
            cycles=brain_stats['cycles'],
            tokens=brain_stats['tokens'],
            loss=brain_stats['loss'],
            vocab_size=brain_stats['vocab_words']
        )
    
    def _generate_response(self, user_message: str) -> str:
        """
        Genera risposta (placeholder).
        
        TODO: Implementare vera generazione quando sarà pronta.
        Per ora: echo + info model.
        
        Args:
            user_message: Messaggio utente
            
        Returns:
            Risposta generata
        """
        # Placeholder - vera generazione verrà implementata
        stats = self.brain.get_stats()
        
        responses = [
            f"I processed your message. Current loss: {stats['loss']:.3f}",
            f"Interesting! I'm learning from this. Vocab: {stats['vocab_words']} words",
            f"Thanks for the input! Perplexity: {stats['perplexity']:.2f}",
            f"Got it. Training cycles: {stats['cycles']:,}",
        ]
        
        # Simple hash-based selection
        idx = hash(user_message) % len(responses)
        return responses[idx]
    
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
