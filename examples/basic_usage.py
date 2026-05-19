"""
Example 1: Basic prompt compression — no LLM needed.

Run: python examples/basic_usage.py
"""

from tokenoptim.core.compressor import PromptCompressor

# A typical over-verbose prompt
verbose_prompt = """
Could you please help me understand the concept of tokenization in large language models?
I would like you to explain how it works, perhaps with a few examples if possible.
It would be really helpful if you could maybe also mention what different tokenization
strategies exist, and perhaps explain the trade-offs between them, if that makes sense.
I think it is important to note that I'm fairly new to this topic, so please make sure
to keep the explanation accessible. Hope this helps me understand this better.
Thank you so much in advance for your help with this!
"""

print("=== tokenoptim — Prompt Compression Demo ===\n")

for level in ["light", "medium", "full"]:
    compressor = PromptCompressor(level=level)
    compressed, stats = compressor.compress(verbose_prompt)
    print(f"[{level.upper()}]")
    print(f"  Original : {verbose_prompt.strip()[:80]}...")
    print(f"  Compressed: {compressed[:80]}...")
    print(f"  Stats    : {stats}")
    print()
