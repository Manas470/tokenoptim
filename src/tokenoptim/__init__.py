"""
tokenoptim — LLM token optimization toolkit.

Reduce prompt & output tokens by 40-75% across Claude, OpenAI, and local models.
"""

import logging

# Library best practice: add NullHandler so callers control log output.
# Without this, Python emits "No handlers could be found for logger tokenoptim"
# warnings in applications that don't configure logging.
logging.getLogger("tokenoptim").addHandler(logging.NullHandler())

from tokenoptim.core.compressor import PromptCompressor
from tokenoptim.core.output_style import OutputCompressor, CompressionLevel
from tokenoptim.core.memory import MemoryManager
from tokenoptim.core.counter import TokenCounter
from tokenoptim.core.retry import RetryConfig
from tokenoptim.core.cache import ResponseCache
from tokenoptim.client import OptimizedClient
from tokenoptim.async_client import AsyncOptimizedClient

__version__ = "0.2.0"
__author__ = "Manas Mourya"

__all__ = [
    "PromptCompressor",
    "OutputCompressor",
    "CompressionLevel",
    "MemoryManager",
    "TokenCounter",
    "RetryConfig",
    "ResponseCache",
    "OptimizedClient",
    "AsyncOptimizedClient",
]
