"""
VectLLM Modules
High-level training and interaction modules
"""

# Import diretti (per evitare problemi con relative imports)
# Use: from modules.training import MultiPassTrainer

__all__ = [
    'MultiPassTrainer',
    'InteractiveChat',
    'DatasetTrainer',
    'WikipediaTrainer',
]
