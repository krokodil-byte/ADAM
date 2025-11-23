#!/usr/bin/env python3
"""
A.D.A.M Command Line Interface
Unified CLI for all A.D.A.M operations

Usage:
    adam              - Open TUI dashboard
    adam train ...    - Train on text
    adam chat ...     - Interactive chat
    adam stats ...    - View statistics
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# Handle imports for both installed package and direct execution
try:
    from core.brain_wrapper import VectLLMBrain
    from core.vocabulary import DynamicVocabulary
    from core.config import (
        MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG,
        set_config_from_preset
    )
    from core.stats import StatsCollector
    from Utils.checkpoint import CheckpointManager
    from modules.training import MultiPassTrainer
    from modules.chat import InteractiveChat
    from modules.dataset_training import DatasetLoader, DatasetTrainer, HFDatasetLoader, HFDatasetTrainer
    from modules.wikipedia_training import WikipediaExtractor, WikipediaTrainer, WikipediaStreamTrainer
    from modules.tui import run_adam_tui
except ImportError:
    from ..core.brain_wrapper import VectLLMBrain
    from ..core.vocabulary import DynamicVocabulary
    from ..core.config import (
        MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG,
        set_config_from_preset
    )
    from ..core.stats import StatsCollector
    from ..Utils.checkpoint import CheckpointManager
    from ..modules.training import MultiPassTrainer
    from ..modules.chat import InteractiveChat
    from ..modules.dataset_training import DatasetLoader, DatasetTrainer, HFDatasetLoader, HFDatasetTrainer
    from ..modules.wikipedia_training import WikipediaExtractor, WikipediaTrainer, WikipediaStreamTrainer
    from ..modules.tui import run_adam_tui

from datetime import datetime


def get_default_checkpoint_path() -> str:
    """Generate default checkpoint path with timestamp"""
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(checkpoint_dir / f"model_{timestamp}.ckpt")


def cmd_init(args):
    """Initialize VectLLM system"""
    print("🚀 Initializing VectLLM...")
    print(f"   GPU: Checking CUDA availability...")
    
    try:
        brain = VectLLMBrain()
        print("✅ CUDA kernel compiled successfully")
        
        # Test initialization
        brain.start()
        stats = brain.get_stats()
        brain.stop()
        
        print("✅ System initialized and tested")
        print(f"   Vocabulary: {stats['vocab_words']} words active")
        print(f"   Model: {MODEL_CONFIG.EMBED_DIM}d embeddings, {MODEL_CONFIG.NUM_LAYERS} layers")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1
    
    return 0


def cmd_train(args):
    """Train on text file"""
    input_file = Path(args.input)
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return 1
    
    print(f"📖 Loading text from: {input_file}")
    text = input_file.read_text()
    print(f"   Size: {len(text):,} chars")
    
    # Apply preset if specified
    if args.preset:
        set_config_from_preset(args.preset)
        print(f"   Config: {args.preset} preset")
    
    # Create brain
    vocab_path = Path(args.vocab) if args.vocab else None
    vocab = None
    if vocab_path and vocab_path.exists():
        print(f"   Loading vocab: {vocab_path}")
        vocab = DynamicVocabulary.load(str(vocab_path))
    
    brain = VectLLMBrain(vocab=vocab)
    
    # Load checkpoint if specified
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"   Loading checkpoint: {ckpt_path}")
            brain.load_checkpoint(str(ckpt_path))
    
    print("🧠 Starting training...")
    brain.start()
    
    try:
        # Training loop
        passes = args.passes
        for pass_num in range(1, passes + 1):
            print(f"\n=== Pass {pass_num}/{passes} ===")
            start_time = time.time()
            
            processed = brain.train_on_text(text, passes=1)
            elapsed = time.time() - start_time
            
            stats = brain.get_stats()
            tokens_per_sec = processed / elapsed if elapsed > 0 else 0
            
            print(f"   Processed: {processed:,} tokens")
            print(f"   Speed: {tokens_per_sec:.0f} tokens/sec")
            print(f"   Loss: {stats['loss']:.4f}")
            print(f"   Perplexity: {stats['perplexity']:.2f}")
            print(f"   Vocab words: {stats['vocab_words']}")
            
            # Auto-checkpoint
            if args.auto_save and pass_num % args.auto_save == 0:
                ckpt_file = f"checkpoint_pass{pass_num}.ckpt"
                print(f"   💾 Auto-saving: {ckpt_file}")
                brain.save_checkpoint(ckpt_file)
        
        # Final save (always save)
        output_path = args.output or get_default_checkpoint_path()
        print(f"\n💾 Saving checkpoint: {output_path}")
        brain.save_checkpoint(output_path)

        # Vocab pruning
        if args.prune_vocab:
            print("\n🧹 Pruning vocabulary...")
            pruned = brain.prune_vocabulary()
            print(f"   Removed: {pruned} rare words")

        print("\n✅ Training complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        output_path = args.output or get_default_checkpoint_path()
        print(f"💾 Saving checkpoint: {output_path}")
        brain.save_checkpoint(output_path)
    finally:
        brain.stop()
    
    return 0


def cmd_generate(args):
    """Generate text interactively"""
    print("🎨 VectLLM Text Generation")
    
    # Load checkpoint
    if not args.checkpoint:
        print("❌ Checkpoint required for generation (--checkpoint)")
        return 1
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return 1
    
    brain = VectLLMBrain()
    print(f"   Loading: {ckpt_path}")
    brain.load_checkpoint(str(ckpt_path))
    
    brain.start()
    
    try:
        print("\n=== Interactive Generation ===")
        print("Enter prompt (or 'quit' to exit)\n")
        
        while True:
            prompt = input("Prompt> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            # Encode prompt
            tokens = brain.encode_text(prompt)
            print(f"Tokens: {tokens}")
            
            # TODO: Implement actual generation
            # For now, just show stats
            stats = brain.get_stats()
            print(f"Loss: {stats['loss']:.4f}, Vocab: {stats['vocab_words']} words")
            print()
    
    except KeyboardInterrupt:
        print("\n")
    finally:
        brain.stop()
    
    return 0


def cmd_stats(args):
    """Show system statistics"""
    
    if args.checkpoint:
        # Load from checkpoint
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return 1
        
        brain = VectLLMBrain()
        brain.load_checkpoint(str(ckpt_path))
        brain.start()
        stats = brain.get_stats()
        brain.stop()
    else:
        # Live stats
        brain = VectLLMBrain()
        brain.start()
        stats = brain.get_stats()
        brain.stop()
    
    print("📊 VectLLM Statistics")
    print("=" * 50)
    print(f"Training Cycles:    {stats['cycles']:,}")
    print(f"Tokens Processed:   {stats['tokens']:,}")
    print(f"Current Loss:       {stats['loss']:.4f}")
    print(f"Perplexity:         {stats['perplexity']:.2f}")
    print()
    print(f"Temperature:        {stats['temperature']:.3f}")
    print(f"Momentum:           {stats['momentum']:.3f}")
    print()
    print(f"Vocab Words:        {stats['vocab_words']:,}")
    print(f"Vocab Utilization:  {stats['vocab_utilization']*100:.1f}%")
    print("=" * 50)
    
    return 0


def cmd_vocab(args):
    """Vocabulary management"""
    
    if args.action == 'stats':
        if not args.file:
            print("❌ Vocab file required (--file)")
            return 1
        
        vocab_path = Path(args.file)
        if not vocab_path.exists():
            print(f"❌ Vocab file not found: {vocab_path}")
            return 1
        
        vocab = DynamicVocabulary.load(str(vocab_path))
        vocab.print_stats()
    
    elif args.action == 'prune':
        if not args.file:
            print("❌ Vocab file required (--file)")
            return 1
        
        vocab_path = Path(args.file)
        if not vocab_path.exists():
            print(f"❌ Vocab file not found: {vocab_path}")
            return 1
        
        vocab = DynamicVocabulary.load(str(vocab_path))
        print(f"Words before: {len(vocab.word_to_id)}")
        
        pruned = vocab.prune_rare_words()
        print(f"Pruned: {pruned} words")
        print(f"Words after: {len(vocab.word_to_id)}")
        
        vocab.save(str(vocab_path))
        print(f"✅ Saved: {vocab_path}")
    
    return 0


def cmd_checkpoint(args):
    """Checkpoint management"""
    
    if args.action == 'info':
        # Show checkpoint info
        print(f"📦 Checkpoint: {args.file}")
        print("   (Info display not yet implemented)")
        # TODO: Read header without loading full checkpoint
    
    elif args.action == 'convert':
        print("⚙️  Checkpoint conversion not yet implemented")
    
    return 0


def cmd_chat(args):
    """Interactive chat mode"""
    print("💬 Starting interactive chat...")
    
    # Load checkpoint if provided
    brain = VectLLMBrain()
    
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return 1
        
        print(f"   Loading checkpoint: {ckpt_path.name}")
        brain.load_checkpoint(str(ckpt_path))
    
    brain.start()
    
    try:
        # Start chat
        chat = InteractiveChat(brain)
        chat.start()
    finally:
        brain.stop()
    
    return 0


def cmd_dataset(args):
    """Train on dataset"""
    print("📂 Dataset training")

    dataset_path = Path(args.path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return 1

    # Apply preset if specified
    if args.preset:
        set_config_from_preset(args.preset)

    # Detect dataset type: HuggingFace (JSONL/Parquet/CSV) or plain text
    is_hf_format = dataset_path.suffix.lower() in ['.jsonl', '.json', '.parquet', '.csv', '.tsv']

    # Create brain
    brain = VectLLMBrain()
    if args.checkpoint:
        print(f"   Loading checkpoint: {args.checkpoint}")
        brain.load_checkpoint(args.checkpoint)

    brain.start()

    try:
        stats = StatsCollector()
        ckpt_manager = CheckpointManager()

        if is_hf_format:
            # HuggingFace style dataset
            print(f"   Format: HuggingFace ({dataset_path.suffix})")

            loader = HFDatasetLoader(
                str(dataset_path),
                input_column=args.input_col,
                output_column=args.output_col,
                text_column=args.text_col,
                template=args.template,
                max_samples=args.max_samples
            )

            trainer = HFDatasetTrainer(
                brain, loader, stats, ckpt_manager,
                validation_split=args.val_split
            )

            trainer.train(
                passes=args.passes,
                auto_save_every=args.auto_save or 1000,
                verbose=True,
                enable_validation=args.validation,
                enable_early_stopping=args.early_stopping
            )
        else:
            # Plain text directory/file
            print(f"   Format: Plain text files")

            loader = DatasetLoader(
                str(dataset_path),
                extensions=args.extensions.split(',') if args.extensions else None
            )

            trainer = DatasetTrainer(
                brain, loader, stats, ckpt_manager,
                validation_split=args.val_split
            )

            trainer.train(
                passes=args.passes,
                auto_save_every=args.auto_save,
                verbose=True,
                enable_validation=args.validation,
                enable_early_stopping=args.early_stopping
            )

        # Save final
        if args.output:
            print(f"\n💾 Saving final checkpoint: {args.output}")
            brain.save_checkpoint(args.output)

    finally:
        brain.stop()

    return 0


def cmd_wikipedia(args):
    """Train on Wikipedia dump or API"""
    print("📚 Wikipedia training")

    # Apply preset if specified
    if args.preset:
        set_config_from_preset(args.preset)

    # Create brain
    brain = VectLLMBrain()
    if args.checkpoint:
        print(f"   Loading checkpoint: {args.checkpoint}")
        brain.load_checkpoint(args.checkpoint)

    brain.start()

    try:
        stats = StatsCollector()
        ckpt_manager = CheckpointManager()

        # Check if dump provided or use API fallback
        if args.dump:
            dump_path = Path(args.dump)
            if not dump_path.exists():
                print(f"⚠️  Dump not found: {dump_path}")
                print(f"   Falling back to Wikipedia API...")
                use_api = True
            else:
                use_api = False
        else:
            use_api = True

        if use_api:
            # Use API streaming
            print(f"   Mode: Wikipedia API (streaming)")
            print(f"   Language: {args.language}")
            print(f"   Batch size: {args.batch_size} articles")

            trainer = WikipediaStreamTrainer(
                brain,
                language=args.language,
                batch_size=args.batch_size,
                stats_collector=stats,
                checkpoint_manager=ckpt_manager,
                validation_articles=args.val_articles
            )

            trainer.train(
                max_articles=args.max_articles,
                passes_per_batch=args.passes,
                auto_save_every=args.auto_save or 100,
                verbose=True,
                enable_validation=args.validation,
                enable_early_stopping=args.early_stopping
            )
        else:
            # Use dump file
            print(f"   Mode: Local dump file")
            extractor = WikipediaExtractor(str(dump_path))

            trainer = WikipediaTrainer(brain, extractor, stats, ckpt_manager)

            trainer.train(
                max_articles=args.max_articles,
                auto_save_every=args.auto_save or 100,
                verbose=True
            )

        # Save final (always save)
        output_path = args.output or get_default_checkpoint_path()
        print(f"\n💾 Saving final checkpoint: {output_path}")
        brain.save_checkpoint(output_path)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        output_path = args.output or get_default_checkpoint_path()
        print(f"💾 Saving checkpoint: {output_path}")
        brain.save_checkpoint(output_path)

    finally:
        brain.stop()

    return 0


def cmd_settings(args):
    """Open A.D.A.M TUI"""
    print("⚙️  Opening A.D.A.M TUI...")
    print("   Use arrow keys to navigate, Enter to select, Q to quit\n")

    try:
        run_adam_tui()
        print("\n✅ A.D.A.M TUI closed")
        return 0
    except Exception as e:
        print(f"❌ TUI error: {e}")
        print("   Note: TUI requires a terminal with curses support")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='adam',
        description='A.D.A.M - Adaptive and Dynamic Agent Module',
        epilog='Run "adam" without arguments to open the TUI dashboard'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # INIT command
    parser_init = subparsers.add_parser('init', help='Initialize VectLLM system')
    
    # TRAIN command
    parser_train = subparsers.add_parser('train', help='Train on text file')
    parser_train.add_argument('input', help='Input text file')
    parser_train.add_argument('-o', '--output', help='Output checkpoint file')
    parser_train.add_argument('-c', '--checkpoint', help='Load from checkpoint')
    parser_train.add_argument('-v', '--vocab', help='Vocabulary file')
    parser_train.add_argument('-p', '--passes', type=int, default=1, 
                             help='Number of training passes (default: 1)')
    parser_train.add_argument('--preset', choices=['default', 'fast_learning', 'stable', 'inference', 'research', 'high_performance', 'memory_efficient', 'max_throughput'],
                             help='Config preset')
    parser_train.add_argument('--auto-save', type=int, metavar='N',
                             help='Auto-save checkpoint every N passes')
    parser_train.add_argument('--prune-vocab', action='store_true',
                             help='Prune vocabulary after training')
    parser_train.add_argument('--validation', action='store_true',
                             help='Enable validation during training')
    parser_train.add_argument('--early-stopping', action='store_true',
                             help='Stop when validation stops improving')
    
    # GENERATE command
    parser_gen = subparsers.add_parser('generate', help='Generate text')
    parser_gen.add_argument('-c', '--checkpoint', required=True, help='Checkpoint file')
    parser_gen.add_argument('--temperature', type=float, help='Sampling temperature')
    
    # STATS command
    parser_stats = subparsers.add_parser('stats', help='Show statistics')
    parser_stats.add_argument('-c', '--checkpoint', help='Load stats from checkpoint')
    
    # VOCAB command
    parser_vocab = subparsers.add_parser('vocab', help='Vocabulary management')
    parser_vocab.add_argument('action', choices=['stats', 'prune'], help='Action')
    parser_vocab.add_argument('-f', '--file', help='Vocabulary file')
    
    # CHECKPOINT command
    parser_ckpt = subparsers.add_parser('checkpoint', help='Checkpoint management')
    parser_ckpt.add_argument('action', choices=['info', 'convert'], help='Action')
    parser_ckpt.add_argument('file', help='Checkpoint file')
    
    # CHAT command
    parser_chat = subparsers.add_parser('chat', help='Interactive chat mode')
    parser_chat.add_argument('-c', '--checkpoint', help='Load from checkpoint')
    
    # DATASET command
    parser_dataset = subparsers.add_parser('dataset', help='Train on dataset')
    parser_dataset.add_argument('path', help='Dataset directory or file (supports .txt, .jsonl, .parquet, .csv)')
    parser_dataset.add_argument('-o', '--output', help='Output checkpoint')
    parser_dataset.add_argument('-c', '--checkpoint', help='Load from checkpoint')
    parser_dataset.add_argument('-p', '--passes', type=int, default=1,
                               help='Number of passes (default: 1)')
    parser_dataset.add_argument('--preset', choices=['default', 'fast_learning', 'stable', 'research', 'high_performance', 'memory_efficient', 'max_throughput'],
                               help='Config preset')
    parser_dataset.add_argument('--auto-save', type=int, metavar='N',
                               help='Auto-save every N files/samples')
    parser_dataset.add_argument('--extensions', type=str,
                               help='File extensions for text files (comma-separated, e.g. .txt,.md)')
    # HuggingFace dataset options
    parser_dataset.add_argument('--input-col', type=str, dest='input_col',
                               help='Input column name (auto-detect if not specified)')
    parser_dataset.add_argument('--output-col', type=str, dest='output_col',
                               help='Output column name (auto-detect if not specified)')
    parser_dataset.add_argument('--text-col', type=str, dest='text_col',
                               help='Text column for single-column datasets')
    parser_dataset.add_argument('--template', type=str,
                               help='Template for input/output pairs (default: "{input}\\n\\n{output}")')
    parser_dataset.add_argument('--max-samples', type=int, dest='max_samples',
                               help='Maximum number of samples to load')
    # Validation options
    parser_dataset.add_argument('--validation', action='store_true',
                               help='Enable validation during training')
    parser_dataset.add_argument('--early-stopping', action='store_true',
                               help='Stop when validation stops improving')
    parser_dataset.add_argument('--val-split', type=float, dest='val_split', default=0.1,
                               help='Validation split fraction (default: 0.1)')
    
    # WIKIPEDIA command
    parser_wiki = subparsers.add_parser('wikipedia', help='Train on Wikipedia (dump or API)')
    parser_wiki.add_argument('dump', nargs='?', default=None,
                            help='Wikipedia dump file (JSONL or XML). If not provided, uses API')
    parser_wiki.add_argument('-o', '--output', help='Output checkpoint')
    parser_wiki.add_argument('-c', '--checkpoint', help='Load from checkpoint')
    parser_wiki.add_argument('-p', '--passes', type=int, default=1,
                            help='Training passes per batch (default: 1)')
    parser_wiki.add_argument('--max-articles', type=int,
                            help='Maximum number of articles to process')
    parser_wiki.add_argument('--auto-save', type=int,
                            help='Auto-save every N articles')
    parser_wiki.add_argument('--preset', choices=['default', 'fast_learning', 'stable', 'research', 'high_performance', 'memory_efficient', 'max_throughput'],
                            help='Config preset')
    parser_wiki.add_argument('--language', type=str, default='en',
                            help='Wikipedia language code (default: en)')
    parser_wiki.add_argument('--batch-size', type=int, default=100,
                            help='Number of articles per batch (default: 100)')
    # Validation options
    parser_wiki.add_argument('--validation', action='store_true',
                            help='Enable validation during training')
    parser_wiki.add_argument('--early-stopping', action='store_true',
                            help='Stop when validation stops improving')
    parser_wiki.add_argument('--val-articles', type=int, dest='val_articles', default=10,
                            help='Number of articles for validation (default: 10)')

    # DASHBOARD/TUI command (alias for running without args)
    parser_dashboard = subparsers.add_parser('dashboard', help='Open A.D.A.M TUI dashboard')

    # Parse arguments
    args = parser.parse_args()

    # If no command, open TUI dashboard
    if not args.command:
        return cmd_settings(args)

    # Execute command
    commands = {
        'init': cmd_init,
        'train': cmd_train,
        'generate': cmd_generate,
        'stats': cmd_stats,
        'vocab': cmd_vocab,
        'checkpoint': cmd_checkpoint,
        'chat': cmd_chat,
        'dataset': cmd_dataset,
        'wikipedia': cmd_wikipedia,
        'dashboard': cmd_settings,
    }
    
    try:
        return commands[args.command](args)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
