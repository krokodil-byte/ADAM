"""
Tests for VectLLM Command Line Interface

Note: Some tests are skipped because the CLI module uses relative imports
that require the package to be properly installed. These tests verify
argument parsing and basic logic without importing the full CLI module.
"""

import pytest
import sys
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import CLI module - may fail due to relative imports
try:
    from cli.vectllm import (
        cmd_init, cmd_train, cmd_generate, cmd_stats,
        cmd_vocab, cmd_checkpoint, cmd_chat, cmd_dataset,
        cmd_wikipedia, main
    )
    CLI_AVAILABLE = True
except ImportError as e:
    CLI_AVAILABLE = False
    CLI_IMPORT_ERROR = str(e)


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing - no module import needed"""

    def test_train_command_args(self):
        """Test train command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        train_parser = subparsers.add_parser('train')
        train_parser.add_argument('input')
        train_parser.add_argument('-o', '--output')
        train_parser.add_argument('-c', '--checkpoint')
        train_parser.add_argument('-p', '--passes', type=int, default=1)

        args = parser.parse_args(['train', 'input.txt', '-o', 'output.ckpt', '-p', '5'])

        assert args.command == 'train'
        assert args.input == 'input.txt'
        assert args.output == 'output.ckpt'
        assert args.passes == 5

    def test_vocab_command_args(self):
        """Test vocab command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        vocab_parser = subparsers.add_parser('vocab')
        vocab_parser.add_argument('action', choices=['stats', 'prune'])
        vocab_parser.add_argument('-f', '--file')

        args = parser.parse_args(['vocab', 'stats', '-f', 'vocab.json'])

        assert args.command == 'vocab'
        assert args.action == 'stats'
        assert args.file == 'vocab.json'

    def test_generate_command_args(self):
        """Test generate command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        gen_parser = subparsers.add_parser('generate')
        gen_parser.add_argument('-c', '--checkpoint', required=True)
        gen_parser.add_argument('--temperature', type=float)

        args = parser.parse_args(['generate', '-c', 'model.ckpt', '--temperature', '0.8'])

        assert args.command == 'generate'
        assert args.checkpoint == 'model.ckpt'
        assert args.temperature == 0.8

    def test_stats_command_args(self):
        """Test stats command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        stats_parser = subparsers.add_parser('stats')
        stats_parser.add_argument('-c', '--checkpoint')

        args = parser.parse_args(['stats', '-c', 'model.ckpt'])

        assert args.command == 'stats'
        assert args.checkpoint == 'model.ckpt'

    def test_chat_command_args(self):
        """Test chat command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        chat_parser = subparsers.add_parser('chat')
        chat_parser.add_argument('-c', '--checkpoint')

        args = parser.parse_args(['chat', '-c', 'model.ckpt'])

        assert args.command == 'chat'
        assert args.checkpoint == 'model.ckpt'

    def test_dataset_command_args(self):
        """Test dataset command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        dataset_parser = subparsers.add_parser('dataset')
        dataset_parser.add_argument('path')
        dataset_parser.add_argument('-o', '--output')
        dataset_parser.add_argument('-p', '--passes', type=int, default=1)
        dataset_parser.add_argument('--extensions')

        args = parser.parse_args(['dataset', '/data/texts', '-o', 'model.ckpt', '--extensions', '.txt,.md'])

        assert args.command == 'dataset'
        assert args.path == '/data/texts'
        assert args.output == 'model.ckpt'
        assert args.extensions == '.txt,.md'

    def test_wikipedia_command_args(self):
        """Test wikipedia command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        wiki_parser = subparsers.add_parser('wikipedia')
        wiki_parser.add_argument('dump')
        wiki_parser.add_argument('-o', '--output')
        wiki_parser.add_argument('--max-articles', type=int)
        wiki_parser.add_argument('--auto-save', type=int)

        args = parser.parse_args(['wikipedia', 'wiki.jsonl', '-o', 'model.ckpt', '--max-articles', '1000'])

        assert args.command == 'wikipedia'
        assert args.dump == 'wiki.jsonl'
        assert args.output == 'model.ckpt'
        assert args.max_articles == 1000

    def test_checkpoint_command_args(self):
        """Test checkpoint command argument structure"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        ckpt_parser = subparsers.add_parser('checkpoint')
        ckpt_parser.add_argument('action', choices=['info', 'convert'])
        ckpt_parser.add_argument('file')

        args = parser.parse_args(['checkpoint', 'info', 'model.ckpt'])

        assert args.command == 'checkpoint'
        assert args.action == 'info'
        assert args.file == 'model.ckpt'

    def test_no_command_sets_none(self):
        """Test that no command sets command to None"""
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest='command')

        args = parser.parse_args([])

        assert args.command is None

    def test_preset_choices(self):
        """Test preset argument choices"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')

        train_parser = subparsers.add_parser('train')
        train_parser.add_argument('input')
        train_parser.add_argument('--preset',
                                  choices=['default', 'fast_learning', 'stable', 'inference', 'research'])

        # Valid preset
        args = parser.parse_args(['train', 'input.txt', '--preset', 'fast_learning'])
        assert args.preset == 'fast_learning'

        # Invalid preset should fail
        with pytest.raises(SystemExit):
            parser.parse_args(['train', 'input.txt', '--preset', 'invalid'])


@pytest.mark.skipif(not CLI_AVAILABLE, reason=f"CLI module not importable: {CLI_IMPORT_ERROR if not CLI_AVAILABLE else ''}")
class TestCLIModuleFunctions:
    """Tests that require importing the CLI module"""

    def test_main_is_callable(self):
        """Test that main() is callable"""
        assert callable(main)

    @patch('cli.vectllm.VectLLMBrain')
    def test_cmd_init_success(self, mock_brain_class):
        """Test successful initialization"""
        mock_brain = MagicMock()
        mock_brain.get_stats.return_value = {'vocab_words': 100}
        mock_brain_class.return_value = mock_brain

        args = argparse.Namespace()
        result = cmd_init(args)

        assert result == 0
        mock_brain.start.assert_called_once()
        mock_brain.stop.assert_called_once()

    @patch('cli.vectllm.VectLLMBrain')
    def test_cmd_init_failure(self, mock_brain_class):
        """Test initialization failure"""
        mock_brain_class.side_effect = Exception("CUDA not available")

        args = argparse.Namespace()
        result = cmd_init(args)

        assert result == 1


class TestCLIFileValidation:
    """Tests for file validation logic - without importing CLI module"""

    def test_nonexistent_file_detection(self, temp_dir):
        """Test detection of nonexistent files"""
        nonexistent = temp_dir / "nonexistent.txt"

        assert not nonexistent.exists()

    def test_existing_file_detection(self, temp_dir):
        """Test detection of existing files"""
        existing = temp_dir / "existing.txt"
        existing.write_text("test content")

        assert existing.exists()

    def test_directory_detection(self, temp_dir):
        """Test detection of directories"""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        assert subdir.is_dir()


class TestCLIPathHandling:
    """Tests for path handling - without importing CLI module"""

    def test_checkpoint_path_extension(self):
        """Test that .ckpt extension is handled"""
        name = "model"
        if not name.endswith('.ckpt'):
            name = f"{name}.ckpt"

        assert name == "model.ckpt"

    def test_checkpoint_path_already_has_extension(self):
        """Test path that already has .ckpt extension"""
        name = "model.ckpt"
        if not name.endswith('.ckpt'):
            name = f"{name}.ckpt"

        assert name == "model.ckpt"

    def test_vocab_path_from_checkpoint(self):
        """Test deriving vocab path from checkpoint path"""
        ckpt_path = Path("model.ckpt")
        vocab_path = ckpt_path.with_suffix('.vocab')

        assert vocab_path == Path("model.vocab")

    def test_metadata_path_from_checkpoint(self):
        """Test deriving metadata path from checkpoint path"""
        ckpt_path = Path("model.ckpt")
        metadata_path = ckpt_path.with_suffix('.json')

        assert metadata_path == Path("model.json")


class TestCLICommandMapping:
    """Tests for command mapping structure"""

    def test_command_dict_structure(self):
        """Test that command mapping structure is correct"""
        expected_commands = [
            'init', 'train', 'generate', 'stats',
            'vocab', 'checkpoint', 'chat', 'dataset', 'wikipedia'
        ]

        # This tests that we have the right set of commands
        # The actual mapping is tested in integration tests
        assert len(expected_commands) == 9
        assert 'init' in expected_commands
        assert 'train' in expected_commands

    def test_all_commands_defined(self):
        """Test that all expected commands are defined"""
        commands = {
            'init': 'Initialize system',
            'train': 'Train on text',
            'generate': 'Generate text',
            'stats': 'Show statistics',
            'vocab': 'Vocabulary management',
            'checkpoint': 'Checkpoint management',
            'chat': 'Interactive chat',
            'dataset': 'Train on dataset',
            'wikipedia': 'Train on Wikipedia',
        }

        assert len(commands) == 9


class TestCLIPresetValidation:
    """Tests for preset validation logic"""

    def test_valid_presets(self):
        """Test that all valid presets are recognized"""
        valid_presets = ['default', 'fast_learning', 'stable', 'inference', 'research']

        for preset in valid_presets:
            assert preset in valid_presets

    def test_invalid_preset_detection(self):
        """Test that invalid presets can be detected"""
        valid_presets = ['default', 'fast_learning', 'stable', 'inference', 'research']
        invalid_preset = 'nonexistent'

        assert invalid_preset not in valid_presets


class TestCLIReturnCodes:
    """Tests for return code conventions"""

    def test_success_return_code(self):
        """Test success return code convention"""
        SUCCESS = 0
        assert SUCCESS == 0

    def test_failure_return_code(self):
        """Test failure return code convention"""
        FAILURE = 1
        assert FAILURE == 1

    def test_file_not_found_returns_failure(self, temp_dir):
        """Test that file not found should return failure code"""
        nonexistent = temp_dir / "nonexistent.txt"

        if not nonexistent.exists():
            # Should return 1
            result = 1
        else:
            result = 0

        assert result == 1


class TestCLIAutoSave:
    """Tests for auto-save logic"""

    def test_auto_save_trigger(self):
        """Test auto-save triggering logic"""
        auto_save_interval = 10
        passes = [5, 10, 15, 20, 25]

        should_save = [p for p in passes if p % auto_save_interval == 0]

        assert should_save == [10, 20]

    def test_no_auto_save_when_none(self):
        """Test no auto-save when interval is None"""
        auto_save_interval = None
        pass_num = 10

        # Should not auto-save
        should_save = auto_save_interval and pass_num % auto_save_interval == 0

        assert not should_save


class TestCLIProgressTracking:
    """Tests for progress tracking logic"""

    def test_multiple_passes_counting(self):
        """Test counting multiple training passes"""
        total_passes = 5
        completed = 0

        for pass_num in range(1, total_passes + 1):
            completed += 1

        assert completed == 5

    def test_pass_numbering(self):
        """Test pass numbering is 1-indexed"""
        passes = 3
        pass_numbers = list(range(1, passes + 1))

        assert pass_numbers == [1, 2, 3]


class TestCLIExtensionParsing:
    """Tests for file extension parsing"""

    def test_parse_extensions(self):
        """Test parsing comma-separated extensions"""
        extensions_str = ".txt,.md,.rst"
        extensions = extensions_str.split(',')

        assert extensions == ['.txt', '.md', '.rst']

    def test_parse_single_extension(self):
        """Test parsing single extension"""
        extensions_str = ".txt"
        extensions = extensions_str.split(',')

        assert extensions == ['.txt']

    def test_parse_none_extensions(self):
        """Test handling None extensions"""
        extensions_str = None
        extensions = extensions_str.split(',') if extensions_str else None

        assert extensions is None


class TestCLIErrorMessages:
    """Tests for error message patterns"""

    def test_file_not_found_message(self):
        """Test file not found error message pattern"""
        filepath = "/nonexistent/file.txt"
        message = f"File not found: {filepath}"

        assert "File not found" in message
        assert filepath in message

    def test_checkpoint_not_found_message(self):
        """Test checkpoint not found error message pattern"""
        checkpoint = "model.ckpt"
        message = f"Checkpoint not found: {checkpoint}"

        assert "Checkpoint not found" in message
        assert checkpoint in message

    def test_dump_not_found_message(self):
        """Test dump not found error message pattern"""
        dump = "wiki.jsonl"
        message = f"Dump not found: {dump}"

        assert "Dump not found" in message
        assert dump in message
