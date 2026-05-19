"""
Example 3: PySpark batch compression — compress 1M prompts at scale.

Requirements: pip install tokenoptim[spark]

Run: python examples/spark_batch_demo.py
"""

from pyspark.sql import SparkSession
from tokenoptim.spark import SparkTokenOptimizer

# Sample prompts (replace with your real dataset path)
SAMPLE_PROMPTS = [
    "Could you please help me understand what tokenization means in the context of large language models?",
    "I would really like you to explain, perhaps with examples, how transformer architectures work.",
    "In order to understand RAG better, could you provide a detailed explanation of retrieval-augmented generation?",
    "Please make sure to explain the difference between fine-tuning and prompt engineering in detail.",
    "I think it is important to understand how attention mechanisms work — could you elaborate?",
]

spark = SparkSession.builder \
    .appName("tokenoptim-demo") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Create DataFrame
df = spark.createDataFrame(
    [(i, p) for i, p in enumerate(SAMPLE_PROMPTS * 10)],
    schema=["id", "prompt"],
)

print(f"Processing {df.count()} prompts...")

# Compress with PySpark
optimizer = SparkTokenOptimizer(level="full", spark=spark)
df_compressed = optimizer.compress_dataframe(df, prompt_col="prompt")

# Show results
df_compressed.select("id", "prompt", "compressed_prompt", "reduction_pct").show(5, truncate=60)

# Print savings report
optimizer.savings_report(df, df_compressed)

spark.stop()
