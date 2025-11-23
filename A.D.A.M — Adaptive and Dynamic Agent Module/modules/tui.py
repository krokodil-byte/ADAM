#!/usr/bin/env python3
"""
A.D.A.M TUI - Text User Interface
Interfaccia visuale completa per gestire il modello
"""

import curses
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Handle imports
try:
    from core.config import (
        MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG, RUNTIME_CONFIG,
        PERFORMANCE_CONFIG, GENERATION_CONFIG, VOCAB_OPTIMIZATION_CONFIG
    )
except ImportError:
    from ..core.config import (
        MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG, RUNTIME_CONFIG,
        PERFORMANCE_CONFIG, GENERATION_CONFIG, VOCAB_OPTIMIZATION_CONFIG
    )


class ADAMTUI:
    """TUI completa per A.D.A.M"""

    # Settings file path
    SETTINGS_FILE = Path.home() / ".adam" / "tui_settings.json"

    def __init__(self):
        self.current_menu = 'main'
        self.selected_item = 0
        self.message = ""
        self.message_type = "info"
        self.should_exit = False

        # Stored values for operations
        self.values = {
            'checkpoint': '',
            'input_file': '',
            'output_file': '',
            'passes': 1,
            'preset': 'default',
            'language': 'en',
            'batch_size': 100,
            'max_articles': 0,
            'dataset_path': '',
            'extensions': '.txt,.md',
            # Validation settings
            'enable_validation': True,
            'validation_split': 0.1,
            'validation_articles': 10,
            'early_stopping': True,
            # Vocab optimization settings
            'vocab_opt_enabled': True,
            'max_hot_vocab': 10000,
            'batch_sync_size': 100,
        }

        # Load saved settings
        self._load_settings()

        # Menu definitions
        self.menus = {
            'main': {
                'title': 'A.D.A.M - Main Menu',
                'items': [
                    ('init', '🚀 Initialize System', 'Check CUDA and compile kernels'),
                    ('train', '🧠 Train on Text', 'Train model on text file'),
                    ('wikipedia', '📚 Wikipedia Training', 'Train on Wikipedia articles'),
                    ('dataset', '📂 Dataset Training', 'Train on dataset folder'),
                    ('chat', '💬 Interactive Chat', 'Chat with the model'),
                    ('stats', '📊 View Statistics', 'Show model statistics'),
                    ('settings', '⚙️  Settings', 'Configure model parameters'),
                    ('quit', '🚪 Exit', 'Exit A.D.A.M'),
                ]
            },
            'train': {
                'title': 'Train on Text',
                'items': [
                    ('input', '📄 Input File', 'Select text file to train on'),
                    ('output', '💾 Output Checkpoint', 'Where to save the model'),
                    ('checkpoint', '📦 Load Checkpoint', 'Resume from checkpoint'),
                    ('passes', '🔄 Passes', 'Number of training passes'),
                    ('preset', '⚡ Preset', 'Configuration preset'),
                    ('validation', '✓ Validation', 'Enable validation during training'),
                    ('early_stop', '🛑 Early Stopping', 'Stop when validation stops improving'),
                    ('run', '▶️  Start Training', 'Begin training'),
                    ('back', '← Back', 'Return to main menu'),
                ]
            },
            'wikipedia': {
                'title': 'Wikipedia Training',
                'items': [
                    ('language', '🌍 Language', 'Wikipedia language code'),
                    ('batch', '📦 Batch Size', 'Articles per batch'),
                    ('articles', '📝 Max Articles', 'Maximum articles (0=unlimited)'),
                    ('val_articles', '✓ Validation Articles', 'Articles for validation set'),
                    ('output', '💾 Output Checkpoint', 'Where to save the model'),
                    ('checkpoint', '📦 Load Checkpoint', 'Resume from checkpoint'),
                    ('passes', '🔄 Passes per Batch', 'Training passes per batch'),
                    ('preset', '⚡ Preset', 'Configuration preset'),
                    ('validation', '✓ Validation', 'Enable validation during training'),
                    ('early_stop', '🛑 Early Stopping', 'Stop when validation stops improving'),
                    ('run', '▶️  Start Training', 'Begin Wikipedia training'),
                    ('back', '← Back', 'Return to main menu'),
                ]
            },
            'dataset': {
                'title': 'Dataset Training',
                'items': [
                    ('dataset_path', '📂 Dataset Path', 'File or directory path'),
                    ('output', '💾 Output Checkpoint', 'Where to save the model'),
                    ('checkpoint', '📦 Load Checkpoint', 'Resume from checkpoint'),
                    ('passes', '🔄 Passes', 'Number of training passes'),
                    ('preset', '⚡ Preset', 'Configuration preset'),
                    ('extensions', '📄 Extensions', 'File extensions (.txt,.md)'),
                    ('val_split', '✓ Validation Split', 'Fraction for validation (0.1 = 10%)'),
                    ('validation', '✓ Validation', 'Enable validation during training'),
                    ('early_stop', '🛑 Early Stopping', 'Stop when validation stops improving'),
                    ('run', '▶️  Start Training', 'Begin dataset training'),
                    ('back', '← Back', 'Return to main menu'),
                ]
            },
            'settings': {
                'title': 'Settings',
                'items': [
                    ('model', '🏗️  Model Architecture', 'Layers, dimensions, heads'),
                    ('training', '📈 Training Parameters', 'Learning rate, momentum'),
                    ('generation', '✍️  Generation', 'Continuation bias, temperature, stopping'),
                    ('performance', '⚡ Performance', 'GPU optimizations, pipeline, kernels'),
                    ('vocab_opt', '🔤 Vocab Optimization', 'CPU/GPU hybrid vocab settings'),
                    ('system', '🖥️  System', 'CUDA, checkpoints'),
                    ('save', '💾 Save Settings', 'Save to config file'),
                    ('back', '← Back', 'Return to main menu'),
                ]
            },
            'vocab_opt': {
                'title': 'Vocab Optimization Settings',
                'items': [
                    ('enabled', '✓ Enable Optimization', 'Enable vocab optimization pipeline'),
                    ('max_hot', '🔥 Max Hot Vocab', 'Maximum hot embeddings in GPU'),
                    ('batch_sync', '📦 Batch Sync Size', 'Words per GPU sync batch'),
                    ('back', '← Back', 'Return to settings'),
                ]
            },
            'generation': {
                'title': 'Generation Settings (Continuation Bias)',
                'items': [
                    ('temperature', '🌡️  Temperature', 'Sampling temperature (higher = more random)'),
                    ('min_confidence', '📉 Min Confidence', 'Stop when token probability below this'),
                    ('confidence_decay', '📊 Confidence Decay', 'EMA decay for confidence tracking'),
                    ('low_streak', '🔢 Low Confidence Streak', 'Stop after N low confidence tokens'),
                    ('max_tokens', '📏 Max Tokens', 'Maximum tokens per response'),
                    ('min_tokens', '📐 Min Tokens', 'Minimum before confidence check'),
                    ('stop_newline', '↵ Stop on Newline', 'Stop on double newline'),
                    ('stop_period', '• Stop on Period', 'Stop on sentence end'),
                    ('back', '← Back', 'Return to settings'),
                ]
            },
            'performance': {
                'title': 'Performance Settings',
                'items': [
                    ('gpu_arch', '🖥️ GPU Arch', 'CUDA compute capability (sm_86 for RTX 30xx)'),
                    ('use_cublas', '🔢 Use cuBLAS', 'Enable cuBLAS for matrix operations'),
                    ('use_fused', '🧩 Fused Kernels', 'Enable fused attention+FFN kernels'),
                    ('pipeline', '🔀 Pipeline Mode', 'H2D/compute/D2H overlap'),
                    ('async', '⚡ Async Transfers', 'Asynchronous memory transfers'),
                    ('pinned', '📌 Pinned Memory', 'Use pinned host memory'),
                    ('warp', '🔄 Warp Primitives', 'Use warp-level shuffle operations'),
                    ('target', '🎯 GPU Target', 'Target GPU utilization %'),
                    ('preallocate', '📦 Preallocate', 'Preallocate buffers at init'),
                    ('back', '← Back', 'Return to settings'),
                ]
            },
        }

        # Settings data
        self.settings = self._load_default_settings()

    def _load_default_settings(self) -> Dict[str, Dict[str, Any]]:
        """Load default settings from config"""
        return {
            'model': {
                'num_layers': MODEL_CONFIG.NUM_LAYERS,
                'embed_dim': MODEL_CONFIG.EMBED_DIM,
                'num_heads': MODEL_CONFIG.NUM_HEADS,
                'max_seq_len': MODEL_CONFIG.MAX_SEQ_LEN,
                'max_word_vocab_size': MODEL_CONFIG.MAX_WORD_VOCAB_SIZE,
            },
            'training': {
                'base_lr': TRAINING_CONFIG.BASE_LR,
                'momentum': TRAINING_CONFIG.MOMENTUM,
                'temperature': TRAINING_CONFIG.EXPLORATION_TEMPERATURE,
                'validation_split': TRAINING_CONFIG.VALIDATION_SPLIT,
                'validation_frequency': TRAINING_CONFIG.VALIDATION_FREQUENCY,
                'early_stopping_patience': TRAINING_CONFIG.EARLY_STOPPING_PATIENCE,
                'min_validation_samples': TRAINING_CONFIG.MIN_VALIDATION_SAMPLES,
            },
            'performance': {
                'gpu_arch': RUNTIME_CONFIG.NVCC_ARCH,
                'use_cublas': PERFORMANCE_CONFIG.USE_CUBLAS,
                'use_fused': PERFORMANCE_CONFIG.USE_FUSED_KERNELS,
                'pipeline': PERFORMANCE_CONFIG.PIPELINE_MODE,
                'async': PERFORMANCE_CONFIG.ASYNC_TRANSFERS,
                'pinned': PERFORMANCE_CONFIG.USE_PINNED_MEMORY,
                'warp': PERFORMANCE_CONFIG.USE_WARP_PRIMITIVES,
                'target': PERFORMANCE_CONFIG.GPU_UTILIZATION_TARGET,
                'preallocate': PERFORMANCE_CONFIG.PREALLOCATE_BUFFERS,
            },
            'generation': {
                'temperature': GENERATION_CONFIG.TEMPERATURE,
                'min_confidence': GENERATION_CONFIG.MIN_TOKEN_CONFIDENCE,
                'confidence_decay': GENERATION_CONFIG.CONFIDENCE_DECAY,
                'low_streak': GENERATION_CONFIG.LOW_CONFIDENCE_STREAK,
                'max_tokens': GENERATION_CONFIG.MAX_TOKENS,
                'min_tokens': GENERATION_CONFIG.MIN_TOKENS,
                'stop_newline': GENERATION_CONFIG.STOP_ON_NEWLINE,
                'stop_period': GENERATION_CONFIG.STOP_ON_PERIOD,
            },
            'system': {
                'device_id': RUNTIME_CONFIG.DEVICE_ID,
                'nvcc_arch': RUNTIME_CONFIG.NVCC_ARCH,
            }
        }

    def _load_settings(self):
        """Load settings from file"""
        try:
            if self.SETTINGS_FILE.exists():
                with open(self.SETTINGS_FILE, 'r') as f:
                    saved = json.load(f)
                    # Update only existing keys
                    for key, value in saved.items():
                        if key in self.values:
                            self.values[key] = value
        except Exception:
            pass  # Use defaults if load fails

    def _save_settings_to_file(self, silent=False):
        """Save settings to file"""
        try:
            self.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(self.values, f, indent=2)
            if not silent:
                print(f"✓ Settings saved to {self.SETTINGS_FILE}")
            return True
        except Exception as e:
            if not silent:
                print(f"✗ Failed to save settings: {e}")
            return False

    def run(self):
        """Start TUI"""
        try:
            curses.wrapper(self._main_loop)
        finally:
            # Always save settings on exit
            self._save_settings_to_file()

    def _main_loop(self, stdscr):
        """Main curses loop"""
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()

        # Colors
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)

        while not self.should_exit:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            menu = self.menus[self.current_menu]

            # Draw box
            self._draw_box(stdscr, 0, 0, height - 3, width - 1)

            # Title
            title = f" {menu['title']} "
            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(0, (width - len(title)) // 2, title)
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

            # Menu items
            items = menu['items']
            start_y = 2
            for i, (key, label, desc) in enumerate(items):
                y = start_y + i
                if y >= height - 4:
                    break

                # Highlight selected
                if i == self.selected_item:
                    stdscr.attron(curses.A_REVERSE)

                # Show current value for editable items
                display = f"  {label}"
                value = self._get_item_value(key)
                if value:
                    display += f": {value}"

                stdscr.addstr(y, 2, display[:width-4])

                if i == self.selected_item:
                    stdscr.attroff(curses.A_REVERSE)

            # Description
            if self.selected_item < len(items):
                desc = items[self.selected_item][2]
                stdscr.attron(curses.A_DIM)
                stdscr.addstr(height - 4, 2, f"  {desc}"[:width-4])
                stdscr.attroff(curses.A_DIM)

            # Message
            if self.message:
                color = curses.color_pair(4) if self.message_type == "error" else curses.color_pair(2)
                stdscr.attron(color)
                stdscr.addstr(height - 3, 2, self.message[:width-4])
                stdscr.attroff(color)

            # Footer
            footer = " [↑↓] Navigate  [Enter] Select  [Q] Quit "
            stdscr.attron(curses.A_DIM)
            stdscr.addstr(height - 1, (width - len(footer)) // 2, footer)
            stdscr.attroff(curses.A_DIM)

            stdscr.refresh()

            # Input
            key = stdscr.getch()

            if key == ord('q') or key == ord('Q'):
                if self.current_menu == 'main':
                    self.should_exit = True
                else:
                    self.current_menu = 'main'
                    self.selected_item = 0
            elif key == curses.KEY_UP:
                self.selected_item = (self.selected_item - 1) % len(items)
                self.message = ""
            elif key == curses.KEY_DOWN:
                self.selected_item = (self.selected_item + 1) % len(items)
                self.message = ""
            elif key in (10, 13):  # Enter
                self._handle_selection(stdscr, items[self.selected_item][0])

    def _draw_box(self, stdscr, y, x, h, w):
        """Draw a box"""
        # Corners
        stdscr.addch(y, x, curses.ACS_ULCORNER)
        stdscr.addch(y, w, curses.ACS_URCORNER)
        stdscr.addch(h, x, curses.ACS_LLCORNER)
        stdscr.addch(h, w, curses.ACS_LRCORNER)

        # Lines
        for i in range(x + 1, w):
            stdscr.addch(y, i, curses.ACS_HLINE)
            stdscr.addch(h, i, curses.ACS_HLINE)
        for i in range(y + 1, h):
            stdscr.addch(i, x, curses.ACS_VLINE)
            stdscr.addch(i, w, curses.ACS_VLINE)

    def _get_item_value(self, key: str) -> str:
        """Get display value for menu item"""
        if key == 'input':
            return self.values['input_file'] or "(not set)"
        elif key == 'output':
            return self.values['output_file'] or "(not set)"
        elif key == 'checkpoint':
            return self.values['checkpoint'] or "(none)"
        elif key == 'passes':
            return str(self.values['passes'])
        elif key == 'preset':
            return self.values['preset']
        elif key == 'language':
            return self.values['language']
        elif key == 'batch':
            return str(self.values['batch_size'])
        elif key == 'articles':
            return str(self.values['max_articles']) if self.values['max_articles'] else "unlimited"
        elif key == 'dataset_path':
            return self.values['dataset_path'] or "(not set)"
        elif key == 'extensions':
            return self.values['extensions']
        elif key == 'validation':
            return "enabled" if self.values['enable_validation'] else "disabled"
        elif key == 'early_stop':
            return "enabled" if self.values['early_stopping'] else "disabled"
        elif key == 'val_articles':
            return str(self.values['validation_articles'])
        elif key == 'val_split':
            return f"{self.values['validation_split']:.0%}"
        # Vocab optimization values
        elif key == 'enabled':
            return "enabled" if self.values['vocab_opt_enabled'] else "disabled"
        elif key == 'max_hot':
            return str(self.values['max_hot_vocab'])
        elif key == 'batch_sync':
            return str(self.values['batch_sync_size'])
        return ""

    def _handle_selection(self, stdscr, key: str):
        """Handle menu item selection"""
        # Navigation
        if key == 'back':
            if self.current_menu in ['generation', 'performance', 'vocab_opt']:
                self.current_menu = 'settings'
            else:
                self.current_menu = 'main'
            self.selected_item = 0
            return
        elif key == 'quit':
            self.should_exit = True
            return

        # Settings categories - handle BEFORE menu navigation
        # so we call _edit_settings instead of navigating to their menus
        if key == 'model':
            self._edit_settings(stdscr, 'model')
            return
        elif key == 'training':
            self._edit_settings(stdscr, 'training')
            return
        elif key == 'generation':
            self._edit_settings(stdscr, 'generation')
            return
        elif key == 'performance':
            self._edit_settings(stdscr, 'performance')
            return
        elif key == 'system':
            self._edit_settings(stdscr, 'system')
            return
        elif key == 'vocab_opt':
            self.current_menu = 'vocab_opt'
            self.selected_item = 0
            return
        elif key == 'save':
            self._save_settings()
            return

        # Menu navigation (for non-settings menus)
        if key in self.menus:
            self.current_menu = key
            self.selected_item = 0
            return

        # Actions
        if key == 'init':
            self._run_command(stdscr, 'init')
        elif key == 'chat':
            self._run_command(stdscr, 'chat')
        elif key == 'stats':
            self._run_command(stdscr, 'stats')

        # Inputs
        elif key == 'input':
            value = self._input_dialog(stdscr, "Input File", self.values['input_file'])
            if value is not None:
                self.values['input_file'] = value
        elif key == 'output':
            value = self._input_dialog(stdscr, "Output Checkpoint", self.values['output_file'])
            if value is not None:
                self.values['output_file'] = value
        elif key == 'checkpoint':
            value = self._input_dialog(stdscr, "Load Checkpoint", self.values['checkpoint'])
            if value is not None:
                self.values['checkpoint'] = value
        elif key == 'passes':
            value = self._input_dialog(stdscr, "Passes", str(self.values['passes']))
            if value is not None:
                try:
                    self.values['passes'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"
        elif key == 'preset':
            presets = ['default', 'fast_learning', 'stable', 'high_performance', 'max_throughput', 'memory_efficient']
            idx = self._select_dialog(stdscr, "Select Preset", presets)
            if idx >= 0:
                self.values['preset'] = presets[idx]
                # Apply preset immediately to global config
                from core.config import set_config_from_preset
                set_config_from_preset(presets[idx])
                self.message = f"Preset '{presets[idx]}' applied"
                self.message_type = "success"
        elif key == 'language':
            value = self._input_dialog(stdscr, "Language Code", self.values['language'])
            if value is not None:
                self.values['language'] = value
        elif key == 'batch':
            value = self._input_dialog(stdscr, "Batch Size", str(self.values['batch_size']))
            if value is not None:
                try:
                    self.values['batch_size'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"
        elif key == 'articles':
            value = self._input_dialog(stdscr, "Max Articles (0=unlimited)", str(self.values['max_articles']))
            if value is not None:
                try:
                    self.values['max_articles'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"

        # Dataset inputs
        elif key == 'dataset_path':
            value = self._input_dialog(stdscr, "Dataset Path", self.values['dataset_path'])
            if value is not None:
                self.values['dataset_path'] = value
        elif key == 'extensions':
            value = self._input_dialog(stdscr, "Extensions (comma-sep)", self.values['extensions'])
            if value is not None:
                self.values['extensions'] = value

        # Validation inputs
        elif key == 'validation':
            options = ['enabled', 'disabled']
            idx = self._select_dialog(stdscr, "Validation", options)
            if idx >= 0:
                self.values['enable_validation'] = (idx == 0)
        elif key == 'early_stop':
            options = ['enabled', 'disabled']
            idx = self._select_dialog(stdscr, "Early Stopping", options)
            if idx >= 0:
                self.values['early_stopping'] = (idx == 0)
        elif key == 'val_articles':
            value = self._input_dialog(stdscr, "Validation Articles", str(self.values['validation_articles']))
            if value is not None:
                try:
                    self.values['validation_articles'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"
        elif key == 'val_split':
            value = self._input_dialog(stdscr, "Validation Split (0.1 = 10%)", str(self.values['validation_split']))
            if value is not None:
                try:
                    self.values['validation_split'] = float(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"

        # Vocab optimization inputs
        elif key == 'enabled':
            options = ['enabled', 'disabled']
            idx = self._select_dialog(stdscr, "Vocab Optimization", options)
            if idx >= 0:
                self.values['vocab_opt_enabled'] = (idx == 0)
        elif key == 'max_hot':
            value = self._input_dialog(stdscr, "Max Hot Vocab", str(self.values['max_hot_vocab']))
            if value is not None:
                try:
                    self.values['max_hot_vocab'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"
        elif key == 'batch_sync':
            value = self._input_dialog(stdscr, "Batch Sync Size", str(self.values['batch_sync_size']))
            if value is not None:
                try:
                    self.values['batch_sync_size'] = int(value)
                except ValueError:
                    self.message = "Invalid number"
                    self.message_type = "error"

        # Run commands
        elif key == 'run':
            if self.current_menu == 'train':
                self._run_train(stdscr)
            elif self.current_menu == 'wikipedia':
                self._run_wikipedia(stdscr)
            elif self.current_menu == 'dataset':
                self._run_dataset(stdscr)

    def _input_dialog(self, stdscr, title: str, default: str = "") -> Optional[str]:
        """Show input dialog"""
        height, width = stdscr.getmaxyx()
        dialog_w = min(60, width - 4)
        dialog_h = 5
        start_y = (height - dialog_h) // 2
        start_x = (width - dialog_w) // 2

        curses.curs_set(1)
        buffer = default

        while True:
            # Draw dialog
            stdscr.attron(curses.color_pair(5))
            for i in range(dialog_h):
                stdscr.addstr(start_y + i, start_x, " " * dialog_w)
            stdscr.addstr(start_y, start_x + 2, f" {title} ")
            stdscr.attroff(curses.color_pair(5))

            # Input field
            stdscr.addstr(start_y + 2, start_x + 2, "> " + buffer[:dialog_w-6] + "_")
            stdscr.addstr(start_y + 4, start_x + 2, "[Enter] OK  [Esc] Cancel")

            stdscr.refresh()

            key = stdscr.getch()
            if key == 27:  # Esc
                curses.curs_set(0)
                return None
            elif key in (10, 13):  # Enter
                curses.curs_set(0)
                return buffer
            elif key in (127, 263):  # Backspace
                buffer = buffer[:-1]
            elif 32 <= key <= 126:
                buffer += chr(key)

    def _select_dialog(self, stdscr, title: str, options: List[str]) -> int:
        """Show selection dialog"""
        height, width = stdscr.getmaxyx()
        dialog_w = min(40, width - 4)
        dialog_h = len(options) + 4
        start_y = (height - dialog_h) // 2
        start_x = (width - dialog_w) // 2

        selected = 0

        while True:
            # Draw dialog
            stdscr.attron(curses.color_pair(5))
            for i in range(dialog_h):
                stdscr.addstr(start_y + i, start_x, " " * dialog_w)
            stdscr.addstr(start_y, start_x + 2, f" {title} ")
            stdscr.attroff(curses.color_pair(5))

            # Options
            for i, opt in enumerate(options):
                y = start_y + 2 + i
                if i == selected:
                    stdscr.attron(curses.A_REVERSE)
                stdscr.addstr(y, start_x + 4, opt[:dialog_w-8])
                if i == selected:
                    stdscr.attroff(curses.A_REVERSE)

            stdscr.refresh()

            key = stdscr.getch()
            if key == 27:  # Esc
                return -1
            elif key in (10, 13):  # Enter
                return selected
            elif key == curses.KEY_UP:
                selected = (selected - 1) % len(options)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(options)

    def _edit_settings(self, stdscr, category: str):
        """Edit settings for a category"""
        height, width = stdscr.getmaxyx()
        settings = self.settings[category]
        keys = list(settings.keys())
        selected = 0

        while True:
            stdscr.clear()
            self._draw_box(stdscr, 0, 0, height - 3, width - 1)

            # Title
            title = f" Settings: {category.title()} "
            stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(0, (width - len(title)) // 2, title)
            stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

            # Settings
            for i, key in enumerate(keys):
                y = 2 + i
                if i == selected:
                    stdscr.attron(curses.A_REVERSE)

                value = settings[key]
                stdscr.addstr(y, 2, f"  {key}: {value}"[:width-4])

                if i == selected:
                    stdscr.attroff(curses.A_REVERSE)

            # Footer
            footer = " [↑↓] Navigate  [Enter] Edit  [Q] Back "
            stdscr.attron(curses.A_DIM)
            stdscr.addstr(height - 1, (width - len(footer)) // 2, footer)
            stdscr.attroff(curses.A_DIM)

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_UP:
                selected = (selected - 1) % len(keys)
            elif key == curses.KEY_DOWN:
                selected = (selected + 1) % len(keys)
            elif key in (10, 13):
                setting_key = keys[selected]
                current = str(settings[setting_key])
                new_value = self._input_dialog(stdscr, setting_key, current)
                if new_value is not None:
                    try:
                        # Convert to appropriate type
                        old_value = settings[setting_key]
                        if isinstance(old_value, bool):
                            settings[setting_key] = new_value.lower() in ('true', '1', 'yes')
                        elif isinstance(old_value, int):
                            settings[setting_key] = int(new_value)
                        elif isinstance(old_value, float):
                            settings[setting_key] = float(new_value)
                        else:
                            settings[setting_key] = new_value

                        # Apply performance settings to global config
                        if category == 'performance':
                            self._apply_performance_settings()
                    except ValueError:
                        pass

    def _apply_performance_settings(self):
        """Apply performance settings to global config objects"""
        perf = self.settings['performance']

        # Apply to RUNTIME_CONFIG
        RUNTIME_CONFIG.NVCC_ARCH = perf['gpu_arch']

        # Apply to PERFORMANCE_CONFIG
        PERFORMANCE_CONFIG.USE_CUBLAS = perf['use_cublas']
        PERFORMANCE_CONFIG.USE_FUSED_KERNELS = perf['use_fused']
        PERFORMANCE_CONFIG.PIPELINE_MODE = perf['pipeline']
        PERFORMANCE_CONFIG.ASYNC_TRANSFERS = perf['async']
        PERFORMANCE_CONFIG.USE_PINNED_MEMORY = perf['pinned']
        PERFORMANCE_CONFIG.USE_WARP_PRIMITIVES = perf['warp']
        PERFORMANCE_CONFIG.GPU_UTILIZATION_TARGET = perf['target']
        PERFORMANCE_CONFIG.PREALLOCATE_BUFFERS = perf['preallocate']

    def _save_settings(self):
        """Save settings to file"""
        config_path = Path.home() / ".config" / "adam" / "settings.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            # Also save TUI values
            if self._save_settings_to_file(silent=True):
                self.message = f"✓ Saved settings to ~/.adam/"
                self.message_type = "success"
            else:
                self.message = f"✓ Config saved, TUI values failed"
                self.message_type = "success"
        except Exception as e:
            self.message = f"✗ Save failed: {e}"
            self.message_type = "error"

    def _run_command(self, stdscr, cmd: str):
        """Run a CLI command"""
        curses.endwin()

        # Build command
        if cmd == 'init':
            os.system('adam init')
        elif cmd == 'chat':
            ckpt = f"-c {self.values['checkpoint']}" if self.values['checkpoint'] else ""
            os.system(f"adam chat {ckpt}")
        elif cmd == 'stats':
            ckpt = f"-c {self.values['checkpoint']}" if self.values['checkpoint'] else ""
            os.system(f"adam stats {ckpt}")

        input("\nPress Enter to continue...")
        stdscr.clear()
        stdscr.refresh()

    def _run_train(self, stdscr):
        """Run training command"""
        if not self.values['input_file']:
            self.message = "✗ Input file required"
            self.message_type = "error"
            return

        curses.endwin()

        # Build command
        cmd = f"adam train {self.values['input_file']}"
        if self.values['output_file']:
            cmd += f" -o {self.values['output_file']}"
        if self.values['checkpoint']:
            cmd += f" -c {self.values['checkpoint']}"
        cmd += f" -p {self.values['passes']}"
        cmd += f" --preset {self.values['preset']}"
        if self.values['enable_validation']:
            cmd += " --validation"
        if self.values['early_stopping']:
            cmd += " --early-stopping"
        # Vocab optimization
        if not self.values['vocab_opt_enabled']:
            cmd += " --no-vocab-opt"
        if self.values['max_hot_vocab'] != 10000:
            cmd += f" --max-hot-vocab {self.values['max_hot_vocab']}"
        if self.values['batch_sync_size'] != 100:
            cmd += f" --batch-sync-size {self.values['batch_sync_size']}"

        print(f"\n$ {cmd}\n")
        os.system(cmd)

        input("\nPress Enter to continue...")
        stdscr.clear()
        stdscr.refresh()

    def _run_wikipedia(self, stdscr):
        """Run Wikipedia training"""
        curses.endwin()

        # Build command
        cmd = "adam wikipedia"
        if self.values['output_file']:
            cmd += f" -o {self.values['output_file']}"
        if self.values['checkpoint']:
            cmd += f" -c {self.values['checkpoint']}"
        cmd += f" -p {self.values['passes']}"
        cmd += f" --language {self.values['language']}"
        cmd += f" --batch-size {self.values['batch_size']}"
        if self.values['max_articles']:
            cmd += f" --max-articles {self.values['max_articles']}"
        cmd += f" --preset {self.values['preset']}"
        cmd += f" --val-articles {self.values['validation_articles']}"
        if self.values['enable_validation']:
            cmd += " --validation"
        if self.values['early_stopping']:
            cmd += " --early-stopping"
        # Vocab optimization
        if not self.values['vocab_opt_enabled']:
            cmd += " --no-vocab-opt"
        if self.values['max_hot_vocab'] != 10000:
            cmd += f" --max-hot-vocab {self.values['max_hot_vocab']}"
        if self.values['batch_sync_size'] != 100:
            cmd += f" --batch-sync-size {self.values['batch_sync_size']}"

        print(f"\n$ {cmd}\n")
        os.system(cmd)

        input("\nPress Enter to continue...")
        stdscr.clear()
        stdscr.refresh()

    def _run_dataset(self, stdscr):
        """Run dataset training"""
        if not self.values['dataset_path']:
            self.message = "✗ Dataset path required"
            self.message_type = "error"
            return

        curses.endwin()

        # Build command
        cmd = f"adam dataset {self.values['dataset_path']}"
        if self.values['output_file']:
            cmd += f" -o {self.values['output_file']}"
        if self.values['checkpoint']:
            cmd += f" -c {self.values['checkpoint']}"
        cmd += f" -p {self.values['passes']}"
        cmd += f" --preset {self.values['preset']}"
        if self.values['extensions']:
            cmd += f" --extensions {self.values['extensions']}"
        cmd += f" --val-split {self.values['validation_split']}"
        if self.values['enable_validation']:
            cmd += " --validation"
        if self.values['early_stopping']:
            cmd += " --early-stopping"
        # Vocab optimization
        if not self.values['vocab_opt_enabled']:
            cmd += " --no-vocab-opt"
        if self.values['max_hot_vocab'] != 10000:
            cmd += f" --max-hot-vocab {self.values['max_hot_vocab']}"
        if self.values['batch_sync_size'] != 100:
            cmd += f" --batch-sync-size {self.values['batch_sync_size']}"

        print(f"\n$ {cmd}\n")
        os.system(cmd)

        input("\nPress Enter to continue...")
        stdscr.clear()
        stdscr.refresh()


def run_settings_tui():
    """Legacy function - runs full TUI now"""
    tui = ADAMTUI()
    tui.run()
    return tui.settings


def run_adam_tui():
    """Run the full A.D.A.M TUI"""
    tui = ADAMTUI()
    tui.run()


if __name__ == '__main__':
    run_adam_tui()
