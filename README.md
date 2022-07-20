# Shimoku Dataframer

A small library for creating pandas DataFrame fixtures.

This library generates pandas dataframes with prescribed columns and types, and filled up rows. It can therefore be used to generate arbitrary data for fixtures to be used in unit tests.

# Installation

> pip install shimoku-dataframer

# Usage

Shimoku Dataframer allows you, by passing a dictionary mapping column names to data typesand data, to generate a fixture dataframe.

### Supported data types:

Data types are to be passed as strings:

- `'timestamp'`: np.datetime64 with minute precision.
- `'date'`: np.datetime64 with day precision.
- `'int'`: np.int64.
- `'float'`: np.float64.
- `'str'`: strings.
- `'constant_str'`: a column of a single repeated constant string.
- `'constant_int'`: a column of a single repeated constant integer.
- `'enum'`: a column of values ranging from 0 to a small integer.

After fixing a numpy random seed, the generated fixture is constant and can be used for testing purposes.


# Examples

If no parameters are passed, a dataframe with a single column named `'id'` and containing integers is created.

```python
from shimoku_dataframer import DataFrameMaker

maker = DataFrameMaker(seed=1)  # seed fixes the numpy random seed.
df = maker.make_df(nrows=5)
```

yields

index |id
------|-------
0     | 98539
1     | 77708
2     | 5192
3     | 98047
4     | 50057


In order to use any of the supported types, pass them as a dictionary as follows.

```python
from shimoku_dataframer import DataFrameMaker

columns = {
    'a': 'str',
    'b': 'float',
    'c': 'int'
}

maker = DataFrameMaker(seed=1)
df = maker.make_df(nrows=3, cols=columns)
```

a | b | c
--------------|--------------|--------------
LRmijlfpaqbmhT |  1.624345 |  98539
8gzYuLsul8QCDo | -0.611756 |  77708
YexxPX3EGwnPjh | -0.528172 |   5192


### License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

### Credits

By BitPhy and Alberto Camara
