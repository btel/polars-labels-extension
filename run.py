import polars as pl
from to_sparse import pig_latinnify, sum_i64, to_sparse

df = pl.DataFrame({
    'french': ['ce', 'nest', 'pas', 'latin', 'cochon'],
    'english': ['this', 'is', 'not', 'pig', 'latin'],
})
result = df.with_columns(pig_latinnify(pl.col("*")))
print(result)

df = pl.DataFrame({'a': [1, 5, 2], 'b': [3, None, -1], 'c': [4, 1, 2]})
print(df.select(sum_i64(pl.col("*"))))

df = pl.DataFrame({'1': [1, 0, 1], '2': [0, 1, 1]})
print(df.select(to_sparse(pl.col("*")).list.eval(pl.col("").str.to_integer())))

df = pl.DataFrame({'a': [True, False, True], 'b': [False, True, True]})
print(df.select(to_sparse(pl.col("*"))))
