"""
Tests for VectLLM Dynamic Vocabulary System
"""

import pytest
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vocabulary import DynamicVocabulary


class TestDynamicVocabularyInit:
    """Tests for DynamicVocabulary initialization"""

    def test_default_initialization(self):
        """Test default initialization values"""
        vocab = DynamicVocabulary()

        assert vocab.embed_dim == 768
        assert vocab.char_vocab_size == 256
        assert vocab.max_word_vocab_size == 100000
        assert vocab.creation_threshold == 5
        assert vocab.pruning_threshold == 2
        assert vocab.max_word_length == 20

    def test_custom_initialization(self):
        """Test custom initialization values"""
        vocab = DynamicVocabulary(
            embed_dim=512,
            char_vocab_size=128,
            max_word_vocab_size=50000,
            creation_threshold=3,
            pruning_threshold=1,
            max_word_length=15
        )

        assert vocab.embed_dim == 512
        assert vocab.char_vocab_size == 128
        assert vocab.max_word_vocab_size == 50000
        assert vocab.creation_threshold == 3
        assert vocab.pruning_threshold == 1
        assert vocab.max_word_length == 15

    def test_initial_state(self):
        """Test initial state of vocabulary"""
        vocab = DynamicVocabulary()

        assert len(vocab.word_to_id) == 0
        assert len(vocab.id_to_word) == 0
        assert vocab.next_word_id == 256
        assert vocab.total_words_created == 0
        assert vocab.total_words_pruned == 0
        assert vocab.total_tokens_encoded == 0


class TestVocabularyEncode:
    """Tests for encoding functionality"""

    def test_encode_empty_string(self):
        """Test encoding empty string"""
        vocab = DynamicVocabulary()
        tokens = vocab.encode("")

        assert tokens == []

    def test_encode_single_char(self):
        """Test encoding single character"""
        vocab = DynamicVocabulary()
        tokens = vocab.encode("A")

        assert tokens == [65]  # ASCII 'A'

    def test_encode_simple_word(self):
        """Test encoding simple word (chars only, below threshold)"""
        vocab = DynamicVocabulary(creation_threshold=5)
        tokens = vocab.encode("hello")

        # Should be character tokens
        expected = [104, 101, 108, 108, 111]  # h e l l o
        assert tokens == expected

    def test_encode_preserves_spaces(self):
        """Test that encoding preserves spaces between words"""
        vocab = DynamicVocabulary()
        tokens = vocab.encode("hello world")

        # Should have space (32) between words
        assert 32 in tokens

    def test_encode_multiple_spaces(self):
        """Test encoding text with multiple spaces"""
        vocab = DynamicVocabulary()
        tokens = vocab.encode("hello  world")

        # Should preserve multiple spaces
        space_count = tokens.count(32)
        assert space_count == 2

    def test_word_creation_after_threshold(self):
        """Test that word token is created after reaching threshold"""
        vocab = DynamicVocabulary(creation_threshold=3)

        # First two encodes - chars
        tokens1 = vocab.encode("hello")
        tokens2 = vocab.encode("hello")
        assert len(tokens1) == 5  # Still chars

        # Third encode - should create word token
        tokens3 = vocab.encode("hello")
        assert len(tokens3) == 1  # Now a single word token
        assert tokens3[0] >= 256  # Word token ID

    def test_word_token_reused(self):
        """Test that created word token is reused"""
        vocab = DynamicVocabulary(creation_threshold=2)

        # Create word token
        vocab.encode("test")
        tokens1 = vocab.encode("test")  # Creates token

        # Use again
        tokens2 = vocab.encode("test")

        assert tokens1 == tokens2
        assert len(tokens1) == 1

    def test_long_word_uses_chars(self):
        """Test that words longer than max_word_length use chars"""
        vocab = DynamicVocabulary(creation_threshold=1, max_word_length=5)

        tokens = vocab.encode("longword")  # 8 chars > 5 max

        assert len(tokens) == 8  # All chars, no word token

    def test_total_tokens_encoded_updated(self):
        """Test that total_tokens_encoded is updated"""
        vocab = DynamicVocabulary()

        vocab.encode("hello")
        first_count = vocab.total_tokens_encoded

        vocab.encode("world")
        second_count = vocab.total_tokens_encoded

        assert second_count > first_count


class TestVocabularyDecode:
    """Tests for decoding functionality"""

    def test_decode_empty_list(self):
        """Test decoding empty list"""
        vocab = DynamicVocabulary()
        text = vocab.decode([])

        assert text == ""

    def test_decode_char_tokens(self):
        """Test decoding character tokens"""
        vocab = DynamicVocabulary()
        tokens = [72, 101, 108, 108, 111]  # H e l l o
        text = vocab.decode(tokens)

        assert text == "Hello"

    def test_decode_preserves_roundtrip(self):
        """Test that encode-decode roundtrip preserves text"""
        vocab = DynamicVocabulary()
        original = "Hello World"

        tokens = vocab.encode(original)
        decoded = vocab.decode(tokens)

        assert decoded == original

    def test_decode_word_tokens(self):
        """Test decoding word tokens"""
        vocab = DynamicVocabulary(creation_threshold=2)

        # Create word token
        vocab.encode("test")
        tokens = vocab.encode("test")

        # Decode
        decoded = vocab.decode(tokens)

        assert decoded == "test"

    def test_decode_unknown_token(self):
        """Test decoding unknown token returns replacement char"""
        vocab = DynamicVocabulary()

        # Use a word token ID that doesn't exist
        tokens = [300]  # No word with this ID
        decoded = vocab.decode(tokens)

        assert decoded == "�"

    def test_decode_mixed_tokens(self):
        """Test decoding mix of char and word tokens"""
        vocab = DynamicVocabulary(creation_threshold=2)

        # Create word token for "hello"
        vocab.encode("hello")
        vocab.encode("hello")

        # Now encode mixed text
        tokens = vocab.encode("hello world")
        decoded = vocab.decode(tokens)

        assert decoded == "hello world"


class TestVocabularyPruning:
    """Tests for vocabulary pruning"""

    def test_prune_rare_words(self):
        """Test pruning removes rare words"""
        vocab = DynamicVocabulary(creation_threshold=2, pruning_threshold=3)

        # Create word with low frequency
        vocab.encode("rare")
        vocab.encode("rare")  # freq = 2, creates token

        # Create word with high frequency
        for _ in range(5):
            vocab.encode("common")

        # Prune
        pruned = vocab.prune_rare_words()

        # "rare" should be pruned (freq 2 < threshold 3)
        assert "rare" not in vocab.word_to_id
        assert pruned >= 1

    def test_prune_updates_statistics(self):
        """Test that pruning updates statistics"""
        vocab = DynamicVocabulary(creation_threshold=2, pruning_threshold=5)

        # Create and prune
        vocab.encode("test")
        vocab.encode("test")

        initial_pruned = vocab.total_words_pruned
        vocab.prune_rare_words()

        assert vocab.total_words_pruned > initial_pruned

    def test_prune_empty_vocabulary(self):
        """Test pruning empty vocabulary"""
        vocab = DynamicVocabulary()
        pruned = vocab.prune_rare_words()

        assert pruned == 0


class TestVocabularySaveLoad:
    """Tests for save/load functionality"""

    def test_save_creates_files(self, temp_dir):
        """Test that save creates JSON and freq files"""
        vocab = DynamicVocabulary()
        vocab.encode("test test test")

        filepath = temp_dir / "vocab.json"
        vocab.save(str(filepath))

        assert filepath.exists()
        assert filepath.with_suffix('.freq').exists()

    def test_load_restores_state(self, temp_dir):
        """Test that load restores vocabulary state"""
        # Create and save
        vocab1 = DynamicVocabulary(creation_threshold=2)
        vocab1.encode("hello")
        vocab1.encode("hello")  # Creates token

        filepath = temp_dir / "vocab.json"
        vocab1.save(str(filepath))

        # Load
        vocab2 = DynamicVocabulary.load(str(filepath))

        assert vocab2.word_to_id == vocab1.word_to_id
        assert vocab2.next_word_id == vocab1.next_word_id
        assert vocab2.total_words_created == vocab1.total_words_created

    def test_saved_vocab_produces_same_tokens(self, temp_dir):
        """Test that loaded vocab produces same tokens"""
        vocab1 = DynamicVocabulary(creation_threshold=2)

        # Create word token
        vocab1.encode("test")
        tokens1 = vocab1.encode("test")

        # Save and load
        filepath = temp_dir / "vocab.json"
        vocab1.save(str(filepath))
        vocab2 = DynamicVocabulary.load(str(filepath))

        # Compare
        tokens2 = vocab2.encode("test")
        assert tokens1 == tokens2

    def test_save_json_format(self, temp_dir):
        """Test that saved file is valid JSON"""
        vocab = DynamicVocabulary()
        filepath = temp_dir / "vocab.json"
        vocab.save(str(filepath))

        # Should be valid JSON
        with open(filepath) as f:
            data = json.load(f)

        assert "embed_dim" in data
        assert "word_to_id" in data
        assert "next_word_id" in data


class TestVocabularyEmbeddings:
    """Tests for embedding initialization"""

    def test_get_word_embedding_init(self):
        """Test word embedding initialization"""
        vocab = DynamicVocabulary(embed_dim=768)

        # Mock char embeddings
        char_embeddings = np.random.randn(256, 768).astype(np.float32)

        word_emb = vocab.get_word_embedding_init("hello", char_embeddings)

        assert word_emb.shape == (768,)
        assert word_emb.dtype == np.float32

    def test_embedding_init_empty_word(self):
        """Test embedding initialization for empty word"""
        vocab = DynamicVocabulary(embed_dim=768)
        char_embeddings = np.random.randn(256, 768).astype(np.float32)

        word_emb = vocab.get_word_embedding_init("", char_embeddings)

        assert word_emb.shape == (768,)
        assert np.allclose(word_emb, 0)

    def test_embedding_is_mean_of_chars(self):
        """Test that embedding is mean of character embeddings"""
        vocab = DynamicVocabulary(embed_dim=4)

        # Simple char embeddings
        char_embeddings = np.zeros((256, 4), dtype=np.float32)
        char_embeddings[ord('a')] = [1, 0, 0, 0]
        char_embeddings[ord('b')] = [0, 1, 0, 0]

        word_emb = vocab.get_word_embedding_init("ab", char_embeddings)

        expected = np.array([0.5, 0.5, 0, 0], dtype=np.float32)
        assert np.allclose(word_emb, expected)


class TestVocabularyStats:
    """Tests for statistics"""

    def test_get_stats(self):
        """Test get_stats returns correct structure"""
        vocab = DynamicVocabulary()
        stats = vocab.get_stats()

        assert "char_vocab_size" in stats
        assert "word_vocab_size" in stats
        assert "max_word_vocab_size" in stats
        assert "utilization" in stats
        assert "total_words_created" in stats
        assert "total_words_pruned" in stats
        assert "total_tokens_encoded" in stats
        assert "top_words" in stats

    def test_utilization_calculation(self):
        """Test utilization is calculated correctly"""
        vocab = DynamicVocabulary(max_word_vocab_size=100, creation_threshold=1)

        # Create some words
        vocab.encode("word1")
        vocab.encode("word2")

        stats = vocab.get_stats()
        expected_util = 2 / 100

        assert stats["utilization"] == expected_util

    def test_top_words_sorted_by_frequency(self):
        """Test that top_words are sorted by frequency"""
        vocab = DynamicVocabulary(creation_threshold=100)  # High threshold to keep tracking

        # Create words with different frequencies
        for _ in range(10):
            vocab.encode("common")
        for _ in range(5):
            vocab.encode("medium")
        for _ in range(1):
            vocab.encode("rare")

        stats = vocab.get_stats()
        top = stats["top_words"]

        # First should be "common"
        assert top[0][0] == "common"
        assert top[0][1] == 10


class TestVocabularyEdgeCases:
    """Tests for edge cases"""

    def test_max_vocab_size_limit(self):
        """Test that vocabulary respects max size limit"""
        vocab = DynamicVocabulary(
            max_word_vocab_size=2,
            creation_threshold=1
        )

        # Create max words
        vocab.encode("word1")
        vocab.encode("word2")

        # Try to create more
        tokens = vocab.encode("word3")

        # Should use chars instead of creating new word
        assert len(tokens) == 5  # All chars

    def test_special_characters(self):
        """Test handling of special characters"""
        vocab = DynamicVocabulary()

        text = "hello\tworld\n"
        tokens = vocab.encode(text)
        decoded = vocab.decode(tokens)

        assert decoded == text

    def test_unicode_characters_truncated(self):
        """Test that unicode chars > 255 are truncated"""
        vocab = DynamicVocabulary()

        # Character with code > 255
        tokens = vocab.encode("日")  # Unicode char

        # Should be truncated to 255
        assert all(t <= 255 for t in tokens)
