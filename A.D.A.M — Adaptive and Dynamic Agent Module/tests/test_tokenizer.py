"""
Tests for VectLLM Tokenizer
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.tokenizer import CharTokenizer, WhitespacePreservingTokenizer


class TestCharTokenizerInit:
    """Tests for CharTokenizer initialization"""

    def test_default_initialization(self):
        """Test default initialization values"""
        tokenizer = CharTokenizer()

        assert tokenizer.vocab_size == 256
        assert tokenizer.space_token == 32
        assert tokenizer.newline_token == 10


class TestCharTokenizerEncode:
    """Tests for CharTokenizer encoding"""

    def test_encode_empty_string(self):
        """Test encoding empty string"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("")

        assert tokens == []

    def test_encode_single_char(self):
        """Test encoding single character"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("A")

        assert tokens == [65]

    def test_encode_word(self):
        """Test encoding a word"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("Hello")

        expected = [72, 101, 108, 108, 111]
        assert tokens == expected

    def test_encode_with_spaces(self):
        """Test encoding text with spaces"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("Hello World")

        assert 32 in tokens  # Space
        assert len(tokens) == 11

    def test_encode_special_chars(self):
        """Test encoding special characters"""
        tokenizer = CharTokenizer()

        # Newline
        tokens = tokenizer.encode("\n")
        assert tokens == [10]

        # Tab
        tokens = tokenizer.encode("\t")
        assert tokens == [9]

    def test_encode_punctuation(self):
        """Test encoding punctuation"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("Hello, World!")

        assert 44 in tokens  # Comma
        assert 33 in tokens  # Exclamation

    def test_encode_numbers(self):
        """Test encoding numbers"""
        tokenizer = CharTokenizer()
        tokens = tokenizer.encode("123")

        expected = [49, 50, 51]
        assert tokens == expected

    def test_encode_truncates_high_unicode(self):
        """Test that high unicode values are truncated to 255"""
        tokenizer = CharTokenizer()

        # Character with high code point
        tokens = tokenizer.encode("日")  # Code point > 255

        assert all(t <= 255 for t in tokens)


class TestCharTokenizerDecode:
    """Tests for CharTokenizer decoding"""

    def test_decode_empty_list(self):
        """Test decoding empty list"""
        tokenizer = CharTokenizer()
        text = tokenizer.decode([])

        assert text == ""

    def test_decode_single_token(self):
        """Test decoding single token"""
        tokenizer = CharTokenizer()
        text = tokenizer.decode([65])

        assert text == "A"

    def test_decode_word(self):
        """Test decoding a word"""
        tokenizer = CharTokenizer()
        tokens = [72, 101, 108, 108, 111]
        text = tokenizer.decode(tokens)

        assert text == "Hello"

    def test_decode_filters_invalid_tokens(self):
        """Test that invalid tokens are filtered"""
        tokenizer = CharTokenizer()

        # Include invalid token (negative)
        tokens = [-1, 65, 300]  # Invalid, 'A', Invalid
        text = tokenizer.decode(tokens)

        assert text == "A"  # Only valid token decoded


class TestCharTokenizerRoundtrip:
    """Tests for encode-decode roundtrip"""

    def test_roundtrip_simple(self):
        """Test simple roundtrip"""
        tokenizer = CharTokenizer()
        original = "Hello"

        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)

        assert decoded == original

    def test_roundtrip_with_spaces(self):
        """Test roundtrip with spaces"""
        tokenizer = CharTokenizer()
        original = "Hello World"

        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)

        assert decoded == original

    def test_roundtrip_special_chars(self):
        """Test roundtrip with special characters"""
        tokenizer = CharTokenizer()
        original = "Hello\nWorld\tTest"

        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)

        assert decoded == original

    def test_roundtrip_punctuation(self):
        """Test roundtrip with punctuation"""
        tokenizer = CharTokenizer()
        original = "Hello, World! How are you?"

        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)

        assert decoded == original


class TestCharTokenizerCountTokens:
    """Tests for token counting"""

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string"""
        tokenizer = CharTokenizer()
        count = tokenizer.count_tokens("")

        assert count == 0

    def test_count_tokens_equals_length(self):
        """Test that token count equals string length"""
        tokenizer = CharTokenizer()
        text = "Hello World"
        count = tokenizer.count_tokens(text)

        assert count == len(text)

    def test_count_tokens_with_special_chars(self):
        """Test counting with special characters"""
        tokenizer = CharTokenizer()
        text = "Hello\nWorld"
        count = tokenizer.count_tokens(text)

        assert count == 11


class TestCharTokenizerFileOperations:
    """Tests for file operations"""

    def test_encode_file(self, temp_dir):
        """Test encoding file"""
        tokenizer = CharTokenizer()

        # Create test file
        filepath = temp_dir / "test.txt"
        filepath.write_text("Hello World")

        tokens = tokenizer.encode_file(str(filepath))

        expected = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]
        assert tokens == expected

    def test_decode_to_file(self, temp_dir):
        """Test decoding to file"""
        tokenizer = CharTokenizer()

        tokens = [72, 101, 108, 108, 111]
        filepath = temp_dir / "output.txt"

        tokenizer.decode_to_file(tokens, str(filepath))

        assert filepath.exists()
        assert filepath.read_text() == "Hello"

    def test_file_roundtrip(self, temp_dir):
        """Test file encoding-decoding roundtrip"""
        tokenizer = CharTokenizer()

        original = "Hello World\nThis is a test."
        input_file = temp_dir / "input.txt"
        output_file = temp_dir / "output.txt"

        input_file.write_text(original)

        tokens = tokenizer.encode_file(str(input_file))
        tokenizer.decode_to_file(tokens, str(output_file))

        assert output_file.read_text() == original


class TestWhitespacePreservingTokenizer:
    """Tests for WhitespacePreservingTokenizer"""

    def test_initialization(self):
        """Test initialization includes additional tokens"""
        tokenizer = WhitespacePreservingTokenizer()

        assert tokenizer.tab_token == 9
        assert tokenizer.cr_token == 13

    def test_inherits_from_char_tokenizer(self):
        """Test that it inherits CharTokenizer functionality"""
        tokenizer = WhitespacePreservingTokenizer()

        # Basic encoding should work
        tokens = tokenizer.encode("Hello")
        assert tokens == [72, 101, 108, 108, 111]


class TestNormalizeWhitespace:
    """Tests for whitespace normalization"""

    def test_normalize_preserves_newlines(self):
        """Test normalization preserving newlines"""
        tokenizer = WhitespacePreservingTokenizer()
        text = "Hello    World\n\nMultiple   spaces"

        normalized = tokenizer.normalize_whitespace(text, preserve_newlines=True)

        # Should preserve newlines but normalize other spaces
        assert "\n\n" in normalized
        assert "    " not in normalized

    def test_normalize_removes_newlines(self):
        """Test normalization removing newlines"""
        tokenizer = WhitespacePreservingTokenizer()
        text = "Hello\nWorld"

        normalized = tokenizer.normalize_whitespace(text, preserve_newlines=False)

        assert "\n" not in normalized
        assert normalized == "Hello World"

    def test_normalize_multiple_spaces(self):
        """Test normalization of multiple spaces"""
        tokenizer = WhitespacePreservingTokenizer()
        text = "Hello     World"

        normalized = tokenizer.normalize_whitespace(text)

        assert normalized == "Hello World"

    def test_normalize_tabs(self):
        """Test normalization of tabs"""
        tokenizer = WhitespacePreservingTokenizer()
        text = "Hello\t\tWorld"

        normalized = tokenizer.normalize_whitespace(text)

        assert "\t" not in normalized

    def test_normalize_empty_string(self):
        """Test normalization of empty string"""
        tokenizer = WhitespacePreservingTokenizer()
        normalized = tokenizer.normalize_whitespace("")

        assert normalized == ""

    def test_normalize_only_whitespace(self):
        """Test normalization of only whitespace"""
        tokenizer = WhitespacePreservingTokenizer()
        normalized = tokenizer.normalize_whitespace("   \t\n   ")

        # Should become empty or minimal
        assert normalized.strip() == ""


class TestTokenizerEdgeCases:
    """Tests for edge cases"""

    def test_very_long_text(self):
        """Test encoding very long text"""
        tokenizer = CharTokenizer()
        text = "A" * 10000

        tokens = tokenizer.encode(text)

        assert len(tokens) == 10000

    def test_all_ascii_chars(self):
        """Test encoding all printable ASCII characters"""
        tokenizer = CharTokenizer()

        # All printable ASCII
        text = "".join(chr(i) for i in range(32, 127))
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text

    def test_binary_data(self):
        """Test encoding binary-like data"""
        tokenizer = CharTokenizer()

        # Control characters
        text = "\x00\x01\x02\x03"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        assert decoded == text
