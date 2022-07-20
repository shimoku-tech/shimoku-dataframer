"""Create a dataframe with fixtures for unit testing.
"""
from typing import Dict

import pandas as pd

from .column_factory import column_grabber


class DataFrameMaker:
    """Main Dataframe creator.
    """

    def __init__(self, seed: int = 1):
        self.seed = seed

    def make_df(self, nrows: int = 100, cols: Dict[str, str] = {'id': 'int'},
                str_len: int = 14, enum_len: int = 4) -> pd.DataFrame:
        """Build a DataFrame of the specified dimensions.

        :param nrows: number of rows in the DataFrame.
        :param cols: A dictionary mapping column names to types. Accepted data
        types are the following:
            'timestamp': datetime64 with minute precision.
            'date': datetime64 with day precision.
            'int': np.int64.
            'float': np.float64.
            'str': strings.
            'constant_str': a column of a single repeated constant string.
            'constant_int': a column of a single repeated constant integer.
            'enum': a column of values ranging from 0 to a small integer.
        """

        df = pd.DataFrame()

        for col, dtype in cols.items():
            ColMaker = column_grabber(dtype)
            if dtype in ['str', 'constant_str']:
                cm = ColMaker(nrows, self.seed, str_len)
            elif dtype == 'enum':
                cm = ColMaker(nrows, self.seed, enum_len)
            else:
                cm = ColMaker(nrows, self.seed)

            df[col] = cm.make_col()

        return df
