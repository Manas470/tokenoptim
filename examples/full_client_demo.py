"""
Example 2: Full OptimizedClient with Anthropic — compression + memory + token tracking.

Requirements: pip install tokenoptim[anthropic]
Set: ANTHROPIC_API_KEY environment variable

Run: python examples/full_client_demo.py
"""

import os
from tokenoptim import OptimizedClient
from tokenoptim.providers import AnthropicProvider

client = OptimizedClient(
    provider=AnthropicProvider(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-haiku-4-5-20251001",  # cheapest/fastest
    ),
    compress_prompts=True,
    prompt_compression_level="medium",
    output_level="full",          # caveman-style output, ~60% output reduction
    memory_enabled=True,
    memory_max_turns=10,
    token_budget=50_000,
    system="You are an expert in distributed systems.",
)

client.status()

# Turn 1
print("--- Turn 1 ---")
resp = client.chat(
    "Could you please help me understand what Apache Spark is and how it works? "
    "I would really like to understand the core concepts in detail."
)
print("Assistant:", resp["content"])
if resp["_prompt_stats"]:
    print(f"(Prompt compressed: {resp['_prompt_stats']})")

# Turn 2 — memory is active, context is preserved
print("\n--- Turn 2 ---")
resp2 = client.chat("How does PySpark compare to pandas for large datasets?")
print("Assistant:", resp2["content"])

# Toggle memory OFF — next call is stateless
print("\n--- Toggling memory OFF ---")
client.toggle_memory()

print("\n--- Turn 3 (stateless) ---")
resp3 = client.chat("What is a Spark executor?")
print("Assistant:", resp3["content"])

# Final report
print("\n")
client.counter.report()
