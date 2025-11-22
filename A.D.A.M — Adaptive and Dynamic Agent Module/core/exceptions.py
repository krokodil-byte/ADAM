#!/usr/bin/env python3
"""
VectLLM Custom Exceptions
Centralized exception classes for better error handling
"""


class VectLLMError(Exception):
    """Base exception for all VectLLM errors"""
    pass


class ConfigurationError(VectLLMError):
    """Configuration-related errors"""
    pass


class CheckpointError(VectLLMError):
    """Checkpoint loading/saving errors"""
    pass


class CheckpointNotFoundError(CheckpointError):
    """Checkpoint file not found"""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Checkpoint file is corrupted or invalid"""
    pass


class VocabularyError(VectLLMError):
    """Vocabulary-related errors"""
    pass


class VocabularyFullError(VocabularyError):
    """Vocabulary has reached maximum size"""
    pass


class TokenizationError(VectLLMError):
    """Tokenization errors"""
    pass


class CUDAError(VectLLMError):
    """CUDA/GPU related errors"""
    pass


class CUDANotAvailableError(CUDAError):
    """CUDA is not available on this system"""
    pass


class KernelCompilationError(CUDAError):
    """Failed to compile CUDA kernel"""
    pass


class TrainingError(VectLLMError):
    """Training-related errors"""
    pass


class DatasetError(VectLLMError):
    """Dataset loading/processing errors"""
    pass


class FileNotFoundError(DatasetError):
    """Input file not found"""
    pass


class InvalidFormatError(DatasetError):
    """Invalid file format"""
    pass
