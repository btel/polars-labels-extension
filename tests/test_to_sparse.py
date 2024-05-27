import polars as pl
from to_sparse import to_sparse

def test_integer():
    df = pl.DataFrame({'1': [1, 0, 1], '2': [0, 1, 1]})
    df_sparse = (df.select(to_sparse(pl.col("*")).list.eval(pl.col("").str.to_integer())))
    assert df_sparse[0, 0].to_list()  == [1]
    assert df_sparse[1, 0].to_list()  == [2]
    assert df_sparse[2, 0].to_list()  == [1, 2]

def test_boolean():
    df = pl.DataFrame({'1': [True, False, True], '2': [False, True, True]})
    df_sparse = (df.select(to_sparse(pl.col("*")).list.eval(pl.col("").str.to_integer())))
    assert df_sparse[0, 0].to_list()  == [1]
    assert df_sparse[1, 0].to_list()  == [2]
    assert df_sparse[2, 0].to_list()  == [1, 2]   
