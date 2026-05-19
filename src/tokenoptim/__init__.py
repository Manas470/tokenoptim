"""
tokenoptim — cut LLM token costs by 40-75%.

Works as a CLI tool (no API key needed) or Python SDK.
"""
from __future__ import annotations

import logging

from tokenoptim.async_client import AsyncOptimizedClient
from tokenoptim.client import OptimizedClient
from tokenoptim.core.cache import ResponseCache
from tokenoptim.core.compressor import PromptCompressor
from tokenoptim.core.counter import TokenCounter
from tokenoptim.core.memory import MemoryManager
from tokenoptim.core.output_style import CompressionLevel, OutputCompressor
from tokenoptim.core.retry import RetryConfig

logging.getLogger("tokenoptim").addHandler(logging.NullHandler())

__version__ = "0.2.0"

__all__ = [
    "AsyncOptimizedClient",
    "CompressionLevel",
    "MemoryManager",
    "OptimizedClient",
    "OutputCompressor",
    "PromptCompressor",
    "ResponseCache",
    "RetryConfig",
    "TokenCounter",
    "__version__",
]
