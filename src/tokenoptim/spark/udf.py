"""
PySpark integration for batch prompt compression.

Use this to compress millions of prompts at scale before feeding them to an LLM.
Token reduction applies across the whole dataset — significant cost impact at enterprise scale.

Example
-------
>>> from pyspark.sql import SparkSession
>>> from tokenoptim.spark import SparkTokenOptimizer
>>>
>>> spark = SparkSession.builder.appName("tokenoptim").getOrCreate()
>>> optimizer = SparkTokenOptimizer(level="medium")
>>>
>>> df = spark.read.parquet("s3://my-bucket/prompts/")
>>> df_compressed = optimizer.compress_dataframe(df, prompt_col="prompt")
>>> df_compressed.show()
>>> optimizer.savings_report(df, df_compressed)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Avoid import-time PySpark dependency


# ---------------------------------------------------------------------------
# UDF factories
# ---------------------------------------------------------------------------

def compress_prompts_udf(level: str = "medium"):
    """
    Return a PySpark UDF that compresses a prompt string column.

    Parameters
    ----------
    level : str
        "light" | "medium" | "full"

    Usage
    -----
    >>> from pyspark.sql import functions as F
    >>> compress = compress_prompts_udf(level="full")
    >>> df = df.withColumn("compressed_prompt", compress(F.col("prompt")))
    """
    try:
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    except ImportError as e:
        raise ImportError(
            "pyspark not installed. Run: pip install pyspark"
        ) from e

    # Capture level in closure (avoid importing inside broadcast)
    _level = level

    @udf(
        StructType([
            StructField("compressed", StringType(), False),
            StructField("original_approx_tokens", IntegerType(), False),
            StructField("compressed_approx_tokens", IntegerType(), False),
            StructField("reduction_pct", IntegerType(), False),
        ])
    )
    def _compress(text: Optional[str]):
        if text is None:
            return ("", 0, 0, 0)
        # Import inside UDF — each executor gets its own Python process
        from tokenoptim.core.compressor import PromptCompressor
        compressor = PromptCompressor(level=_level)
        compressed, stats = compressor.compress(text)
        return (
            compressed,
            stats.original_approx_tokens,
            stats.compressed_approx_tokens,
            int(stats.token_reduction_pct),
        )

    return _compress


def batch_compress(
    texts: list[str],
    level: str = "medium",
    n_workers: int = 4,
) -> list[tuple[str, dict]]:
    """
    Compress a list of prompt strings in parallel using Python's ThreadPoolExecutor.

    Useful for smaller batches without a full Spark cluster.

    Parameters
    ----------
    texts : list[str]
        Raw prompts to compress.
    level : str
        Compression level.
    n_workers : int
        Thread pool size.

    Returns
    -------
    list of (compressed_text, stats_dict) tuples.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tokenoptim.core.compressor import PromptCompressor

    compressor = PromptCompressor(level=level)

    def _compress_one(text: str):
        compressed, stats = compressor.compress(text)
        return compressed, {
            "original_tokens": stats.original_approx_tokens,
            "compressed_tokens": stats.compressed_approx_tokens,
            "reduction_pct": stats.token_reduction_pct,
        }

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_compress_one, texts))

    return results


# ---------------------------------------------------------------------------
# High-level SparkTokenOptimizer
# ---------------------------------------------------------------------------

class SparkTokenOptimizer:
    """
    High-level helper for compressing prompt DataFrames with PySpark.

    Parameters
    ----------
    level : str
        Compression level passed to PromptCompressor.
    spark : SparkSession | None
        Optional SparkSession. If None, uses the active session.

    Example
    -------
    >>> optimizer = SparkTokenOptimizer(level="full")
    >>> df_out = optimizer.compress_dataframe(df, prompt_col="prompt")
    >>> optimizer.savings_report(df, df_out)
    """

    def __init__(self, level: str = "medium", spark=None) -> None:
        self.level = level
        self._spark = spark

    def _get_spark(self):
        if self._spark:
            return self._spark
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession()
        except ImportError as e:
            raise ImportError("pyspark not installed: pip install pyspark") from e

    def compress_dataframe(
        self,
        df,
        prompt_col: str = "prompt",
        output_col: str = "compressed_prompt",
        keep_stats: bool = True,
    ):
        """
        Add a compressed column to a Spark DataFrame.

        Parameters
        ----------
        df : pyspark.sql.DataFrame
        prompt_col : str
            Name of the column containing raw prompts.
        output_col : str
            Name of the output compressed column.
        keep_stats : bool
            If True, also adds columns: original_tokens, compressed_tokens, reduction_pct.

        Returns
        -------
        pyspark.sql.DataFrame
        """
        from pyspark.sql import functions as F

        compress_udf = compress_prompts_udf(level=self.level)
        df_out = df.withColumn("_tok_result", compress_udf(F.col(prompt_col)))
        df_out = df_out.withColumn(output_col, F.col("_tok_result.compressed"))

        if keep_stats:
            df_out = (
                df_out
                .withColumn("original_tokens", F.col("_tok_result.original_approx_tokens"))
                .withColumn("compressed_tokens", F.col("_tok_result.compressed_approx_tokens"))
                .withColumn("reduction_pct", F.col("_tok_result.reduction_pct"))
            )

        return df_out.drop("_tok_result")

    def savings_report(self, df_before, df_after, prompt_col: str = "prompt") -> None:
        """
        Print a cost savings summary comparing original vs compressed DataFrames.
        """
        from pyspark.sql import functions as F

        if "original_tokens" not in df_after.columns:
            print("Run compress_dataframe with keep_stats=True to generate a report.")
            return

        agg = df_after.agg(
            F.sum("original_tokens").alias("total_original"),
            F.sum("compressed_tokens").alias("total_compressed"),
            F.avg("reduction_pct").alias("avg_reduction_pct"),
            F.count("*").alias("num_prompts"),
        ).collect()[0]

        saved = agg["total_original"] - agg["total_compressed"]
        print("\n━━━━ tokenoptim PySpark Savings Report ━━━━")
        print(f"Prompts processed   : {agg['num_prompts']:,}")
        print(f"Total original tokens: {agg['total_original']:,}")
        print(f"Total compressed     : {agg['total_compressed']:,}")
        print(f"Tokens saved         : {saved:,} ({agg['avg_reduction_pct']:.1f}% avg)")
        # Rough cost estimate at $3/1M tokens (Claude Sonnet input)
        cost_saved = (saved / 1_000_000) * 3.0
        print(f"Est. cost saved      : ${cost_saved:.4f} at $3/1M tokens")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
