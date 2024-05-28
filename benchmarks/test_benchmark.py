import polars as pl
import random
from to_sparse import to_sparse

def test_performance(benchmark):
    # Precompute some data useful for the benchmark but that should not be
    df = pl.DataFrame({
        str(i): [random.random() > 0.9 for i in range(1000)]
        for i in range(100)})

    # Benchmark the execution of the function
    benchmark(lambda: df.select(to_sparse(pl.col("*"))))
