# pylint: disable=missing-docstring,no-self-use
from typing import Tuple, Dict

import pytest

import pandas as pd
import pandas.util.testing as tm

from shimoku_dataframer import DataFrameMaker
from shimoku_dataframer.exceptions import UnsupportedDataType

assert pd.__version__ == '0.23.4'


def check_df(df: pd.DataFrame, shape: Tuple[int, int], cols: Dict[str, str]):
    assert df.shape == shape
    assert list(df.columns) == list(cols.keys())


class TestDataFrameMaker(object):

    def test__init(self):

        dm = DataFrameMaker()
        assert dm.seed == 1
        dm = DataFrameMaker(seed=444)
        assert dm.seed == 444

    def test_make_df(self):

        dm = DataFrameMaker()
        with pytest.raises(UnsupportedDataType):
            dm.make_df(nrows=100, cols={'id': 'foo'})

        n = 100
        t_cols = {'id': 'int'}
        df = dm.make_df(nrows=n, cols=t_cols)
        check_df(df, (100, 1), t_cols)

        df_res = pd.read_pickle('tests/df1.pkl')

        tm.assert_frame_equal(df, df_res)


        n = 1000
        t_cols = {
            'ts': 'timestamp',
            'dt': 'date',
            'ints': 'int',
            'floats': 'float',
            's': 'str',
            'cs': 'constant_str',
            'ci': 'constant_int',
            'en': 'enum',
        }
        df = dm.make_df(nrows=n, cols=t_cols, str_len=12, enum_len=5)
        check_df(df, (n, len(t_cols)), t_cols)

        df_res = pd.read_pickle('tests/df2.pkl')

        tm.assert_frame_equal(df, df_res)
