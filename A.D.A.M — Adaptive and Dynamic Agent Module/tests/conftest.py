"""
Shared pytest fixtures for VectLLM tests
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "Hello world. This is a test. Hello again. Testing dynamic vocabulary."


@pytest.fixture
def sample_texts():
    """Multiple sample texts for vocabulary testing"""
    return [
        "hello world",
        "hello again",
        "hello hello hello",
        "world is beautiful",
        "hello world again",
        "testing dynamic vocabulary system",
        "hello world testing",
    ]


@pytest.fixture
def mock_brain(mocker):
    """Mock VectLLMBrain for testing without CUDA"""
    class MockBrain:
        def __init__(self):
            self.started = False
            self.total_trained = 0
            self.vocab_words = 0

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def train_on_text(self, text, passes=1):
            tokens = len(text)
            self.total_trained += tokens * passes
            return tokens * passes

        def get_stats(self):
            return {
                'cycles': self.total_trained,
                'tokens': self.total_trained,
                'loss': 2.5,
                'perplexity': 12.18,
                'temperature': 1.0,
                'momentum': 0.9,
                'vocab_words': self.vocab_words,
                'vocab_utilization': 0.001,
            }

        def encode_text(self, text):
            return [ord(c) for c in text]

        def save_checkpoint(self, path):
            Path(path).touch()

        def load_checkpoint(self, path):
            pass

        def prune_vocabulary(self):
            return 5

    return MockBrain()


@pytest.fixture
def checkpoint_info():
    """Sample checkpoint info for testing"""
    from Utils.checkpoint import CheckpointInfo

    return CheckpointInfo(
        version=3,
        magic="VECTLLM3",
        char_vocab_size=256,
        current_word_vocab_size=100,
        max_word_vocab_size=100000,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        num_clusters=256,
        total_cycles=5000,
        total_tokens=25000,
        timestamp=1234567890,
        learning_rate=0.0001,
        momentum=0.9,
        current_loss=2.5
    )
