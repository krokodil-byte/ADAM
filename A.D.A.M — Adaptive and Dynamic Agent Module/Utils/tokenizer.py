#!/usr/bin/env python3
"""
Character-Level Tokenizer
Tokenizer semplice a livello carattere con gestione spazi
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# ASCII Constants
ASCII_SPACE = 32
ASCII_NEWLINE = 10
ASCII_TAB = 9
ASCII_CR = 13


class CharTokenizer:
    """
    Tokenizer character-level base.
    Ogni carattere ASCII (0-255) è un token.
    """

    def __init__(self):
        self.vocab_size = 256
        self.space_token = ASCII_SPACE
        self.newline_token = ASCII_NEWLINE
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text a token IDs (char-level).
        
        Args:
            text: Testo da codificare
            
        Returns:
            Lista di token IDs (0-255)
        """
        return [min(ord(c), 255) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs a text.
        
        Args:
            tokens: Lista di token IDs
            
        Returns:
            Testo decodificato
        """
        return ''.join(chr(t) for t in tokens if 0 <= t <= 255)
    
    def count_tokens(self, text: str) -> int:
        """Conta numero di token nel testo"""
        return len(text)
    
    def encode_file(self, filepath: str) -> List[int]:
        """
        Encode intero file a tokens.
        
        Args:
            filepath: Path al file di testo
            
        Returns:
            Lista di token IDs
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return self.encode(text)
    
    def decode_to_file(self, tokens: List[int], filepath: str):
        """
        Decode tokens e salva su file.
        
        Args:
            tokens: Lista di token IDs
            filepath: Path al file output
        """
        text = self.decode(tokens)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)


class WhitespacePreservingTokenizer(CharTokenizer):
    """
    Tokenizer che preserva spazi multipli e whitespace.
    Utile per codice o testo formattato.
    """
    
    def __init__(self):
        super().__init__()
        self.tab_token = 9  # ASCII tab
        self.cr_token = 13  # Carriage return
    
    def normalize_whitespace(self, text: str, preserve_newlines: bool = True) -> str:
        """
        Normalizza whitespace mantenendo struttura.
        
        Args:
            text: Testo input
            preserve_newlines: Se True, mantiene newlines
            
        Returns:
            Testo normalizzato
        """
        if preserve_newlines:
            # Mantieni newlines, normalizza altri whitespace
            lines = text.split('\n')
            normalized_lines = [' '.join(line.split()) for line in lines]
            return '\n'.join(normalized_lines)
        else:
            # Normalizza tutto
            return ' '.join(text.split())


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("=== Tokenizer Test ===\n")
    
    # Test 1: Basic encoding
    tokenizer = CharTokenizer()
    
    text = "Hello, World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    print(f"Match: {text == decoded}\n")
    
    # Test 2: Special characters
    text2 = "Hello\nWorld\tTest"
    tokens2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(tokens2)
    
    print(f"Text with special chars: '{repr(text2)}'")
    print(f"Tokens: {tokens2}")
    print(f"Decoded: '{repr(decoded2)}'")
    print(f"Match: {text2 == decoded2}\n")
    
    # Test 3: Whitespace preserving
    ws_tokenizer = WhitespacePreservingTokenizer()
    
    text3 = "Hello    World\n\nMultiple   spaces"
    normalized = ws_tokenizer.normalize_whitespace(text3)
    
    print(f"Original: '{text3}'")
    print(f"Normalized: '{normalized}'\n")
    
    # Test 4: Token counting
    long_text = "This is a longer text for token counting test."
    count = tokenizer.count_tokens(long_text)
    
    print(f"Text: '{long_text}'")
    print(f"Token count: {count}")
    print(f"Length: {len(long_text)}")
    print(f"Match: {count == len(long_text)}\n")
    
    # Test 5: Unicode handling
    unicode_text = "Café résumé naïve 日本語"
    tokens_unicode = tokenizer.encode(unicode_text)
    decoded_unicode = tokenizer.decode(tokens_unicode)
    
    print(f"Unicode text: '{unicode_text}'")
    print(f"Tokens: {tokens_unicode[:20]}...")  # First 20
    print(f"Decoded: '{decoded_unicode}'")
    print(f"Note: Non-ASCII chars may be truncated to byte range\n")
    
    print("✅ Tokenizer tests complete!")
