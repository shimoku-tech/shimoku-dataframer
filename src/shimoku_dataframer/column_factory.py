"""This module contains a factory of data columns to be used to build
dataframes.
"""
# pylint: disable=too-few-public-methods,useless-super-delegation
from abc import ABC, abstractmethod
import string

import numpy as np

from .exceptions import UnsupportedDataType

CHARS = [l for l in string.ascii_letters + string.digits]

# pylint: disable=too-many-return-statements
def column_grabber(dtype: str):
    """Returns a column maker for the specified data type. Accepted data types
    are the following:
        'timestamp': datetime64 with minute precision.
        'date': datetime64 with day precision.
        'int': np.int64.
        'float': np.float64.
        'str': strings.
        'constant_str': a column of a single repeated constant string.
        'constant_int': a column of a single repeated constant integer.
        'enum': a column of values ranging from 0 to a small integer.
    """

    if dtype == "timestamp":
        return TimestampColumnMaker
    elif dtype == "date":
        return DateColumnMaker
    elif dtype == "int":
        return IntColumnMaker
    elif dtype == "float":
        return FloatColumnMaker
    elif dtype == "str":
        return StringColumnMaker
    elif dtype == "constant_str":
        return ConstantStringColumnMaker
    elif dtype == "constant_int":
        return ConstantIntColumnMaker
    elif dtype == "enum":
        return EnumColumnMaker

    raise UnsupportedDataType(
        f"Data type '{dtype}' currently not supported."
    )


class ColumnMakerBaseClass(ABC):
    """Abstract base class for column makers.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        self.nrows = nrows
        self.seed = seed

    @abstractmethod
    def make_col(self) -> np.array:
        """This method implements the generation of column data and returns it
        as a numpy array.
        """
        pass


class IntColumnMaker(ColumnMakerBaseClass):
    """Make a column of integers.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        super().__init__(nrows, seed)

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        return np.random.randint(0, 10**5, self.nrows, dtype=np.int64)


class FloatColumnMaker(ColumnMakerBaseClass):
    """Make a column of floats.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        super().__init__(nrows, seed)

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        return np.random.randn(self.nrows)


class TimestampColumnMaker(ColumnMakerBaseClass):
    """Make a column of timestamps.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        super().__init__(nrows, seed)

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        start_seed = np.datetime64("2017-01-01 00:00")
        np.random.seed(self.seed)
        jumps = np.random.randint(1, 10**6, size=self.nrows)

        return np.array([start_seed + j for j in jumps])


class DateColumnMaker(ColumnMakerBaseClass):
    """Make a column of dates.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        super().__init__(nrows, seed)

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        start_seed = np.datetime64("2017-01-01")
        np.random.seed(self.seed)
        jumps = np.random.randint(1, 10**4, size=self.nrows)

        return np.array([start_seed + j for j in jumps])

class StringColumnMaker(ColumnMakerBaseClass):
    """Make a column of strings of fixed length.
    """

    def __init__(self, nrows: int, seed: int, str_len: int = 12):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        :param str_len: length of the desired strings.
        """
        super().__init__(nrows, seed)
        self.str_len = str_len

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        return np.array([''.join(np.random.choice(CHARS,
                                                  self.str_len))
                         for _ in range(self.nrows)])


class ConstantStringColumnMaker(ColumnMakerBaseClass):
    """Make a column with a repeated constant string.
    """

    def __init__(self, nrows: int, seed: int, str_len: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        :param str_len: length of the desired constant string.
        """
        super().__init__(nrows, seed)
        self.str_len = str_len

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        return np.array(self.nrows*[''.join(np.random.choice(CHARS,
                                                             self.str_len))])


class ConstantIntColumnMaker(ColumnMakerBaseClass):
    """Make a column with a constant integer value.
    """

    def __init__(self, nrows: int, seed: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        """
        super().__init__(nrows, seed)

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        return np.repeat(np.random.randint(0, 10**5, dtype=np.int64),
                         self.nrows)


class EnumColumnMaker(ColumnMakerBaseClass):
    """Make a column with categorical enum values.
    """

    def __init__(self, nrows: int, seed: int, enum_vals: int):
        """Save column metadata.

        :param nrows: number of rows.
        :param seed: fix numpy random seed value to this.
        :param enum_vals: take values up to this.
        """
        super().__init__(nrows, seed)
        self.enum_vals = enum_vals

    def make_col(self) -> np.array:
        """Compute column data and return it as a numpy array.
        """
        np.random.seed(self.seed)
        val_pool = np.arange(0, self.enum_vals)
        return np.random.choice(val_pool, size=self.nrows)
