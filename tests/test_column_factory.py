# pylint: disable=missing-docstring,no-self-use,bad-whitespace,bad-continuation
import pytest

import numpy as np

from shimoku_dataframer.column_factory import column_grabber
from shimoku_dataframer.column_factory import IntColumnMaker, FloatColumnMaker, EnumColumnMaker
from shimoku_dataframer.column_factory import TimestampColumnMaker, DateColumnMaker
from shimoku_dataframer.column_factory import StringColumnMaker, ConstantStringColumnMaker
from shimoku_dataframer.column_factory import ConstantIntColumnMaker

from shimoku_dataframer.exceptions import UnsupportedDataType

def test_column_grabber():

    cm = column_grabber('timestamp')
    assert cm == TimestampColumnMaker

    cm = column_grabber('date')
    assert cm == DateColumnMaker

    cm = column_grabber('int')
    assert cm == IntColumnMaker

    cm = column_grabber('float')
    assert cm == FloatColumnMaker

    cm = column_grabber('str')
    assert cm == StringColumnMaker

    cm = column_grabber('constant_str')
    assert cm == ConstantStringColumnMaker

    cm = column_grabber('constant_int')
    assert cm == ConstantIntColumnMaker

    cm = column_grabber('enum')
    assert cm == EnumColumnMaker

    with pytest.raises(UnsupportedDataType):
        column_grabber('foo')


class TestIntColumnMaker(object):

    def test__init(self):
        cm = IntColumnMaker(100, 401)
        assert cm.nrows == 100
        assert cm.seed == 401

    def test_make_col(self):
        cm = IntColumnMaker(100, 400)
        s = cm.make_col()
        assert len(s) == 100
        exp_res = np.array(
            [69071, 67902, 46009, 65676, 70643, 82327, 93446, 96501, 28808,
             79039, 2246, 24091, 51558, 14498, 29185, 47886, 66641, 93509,
             21015, 87758, 20228, 97123, 21681, 62834, 35052, 70452, 73521,
             84163, 93641, 66465, 52180, 58774, 47389, 43261, 3947, 43500,
             86956, 81793, 1264, 57466, 16499, 88436, 12647, 37381, 80421,
             71588, 75587, 970, 67416, 54062, 832, 79393, 77652, 57771,
             50024, 9394, 78031, 31460, 15194, 37010, 99283, 71831, 75609,
             59395, 28928, 6869, 17481, 13584, 28508, 60369, 60002, 10214,
             52368, 12265, 52906, 43671, 42519, 63824, 58018, 1102, 8095,
             97951, 75946, 78792, 37072, 27476, 49566, 70231, 73306, 59725,
             88864, 30594, 94373, 75155, 30940, 90242, 40160, 31587, 8712,
             24498]
        )
        np.testing.assert_array_equal(s, exp_res)

class TestFloatColumnMaker(object):

    def test__init(self):
        cm = FloatColumnMaker(100, 403)
        assert cm.nrows == 100
        assert cm.seed == 403

    def test_make_col(self):
        cm = FloatColumnMaker(100, 404)
        s = cm.make_col()
        assert len(s) == 100
        exp_res = np.array(
            [-1.18269089,  1.33746165, -0.31430504,  0.5566431 , -0.20929933,
             -0.01180037,  1.23493345, -0.04341894, -0.68178724,  1.20069197,
              1.34394438, -0.834109  ,  1.0359977 ,  1.6657991 ,  0.94947856,
              1.06700952,  0.39448078,  0.60035462, -1.35330447, -0.96449618,
             -0.35066941,  0.32114751, -1.00624144,  0.74879582,  0.03032371,
              0.1818493 , -0.49743505, -0.33721919,  0.4873507 ,  0.02626495,
             -0.17983642, -0.55894996,  0.54115242, -0.27699667, -0.11066764,
              0.30976783,  1.76463343, -0.26426718,  0.8739962 ,  1.0820967 ,
             -0.44475137, -0.14789828,  1.6292685 ,  1.10013229, -0.40411208,
              0.52642534,  1.62882925, -0.47027466, -0.42043333, -1.29220809,
              1.98623785,  0.23549476, -0.95989601,  0.00338232, -0.44951812,
             -0.83197963, -1.06399132,  0.90569629,  0.90134097,  2.20378598,
             -0.73025773,  0.66868426, -0.04096123,  0.13807827,  0.27802423,
             -0.18573573, -1.36662486,  0.19413865, -1.65499799,  0.17347657,
             -0.56490422,  2.47041146, -0.5928873 , -1.25291136, -0.56030953,
             -0.00727728, -1.13354003,  0.05320708, -2.44622148,  1.98755155,
              0.7500889 , -1.72827758,  1.93577379, -0.73249951,  0.76423933,
              0.59785635,  0.38259364,  2.13389131,  2.52744425, -0.12602527,
             -0.50884558, -1.20580224,  0.13675123,  0.13892423, -0.41673309,
              0.83939946, -0.29615155, -1.35064133, -0.6735806 , -0.63160002]
        )
        np.testing.assert_array_almost_equal(s, exp_res)


class TestTimestampColumnMaker(object):

    def test__init(self):
        cm = TimestampColumnMaker(100, 203)
        assert cm.nrows == 100
        assert cm.seed == 203

    def test_make_col(self):
        cm = TimestampColumnMaker(50, 104)
        s = cm.make_col()
        assert len(s) == 50
        exp_res = np.array(
            ['2018-10-05T21:26', '2017-08-09T05:22', '2017-03-22T03:38',
             '2018-08-13T16:28', '2017-12-25T20:31', '2017-04-29T03:51',
             '2018-02-06T20:48', '2018-10-28T09:29', '2018-04-22T19:21',
             '2017-01-10T20:21', '2018-05-07T02:25', '2017-11-11T11:16',
             '2018-05-11T08:56', '2017-06-13T19:58', '2018-06-20T05:49',
             '2018-11-17T08:04', '2017-01-07T08:33', '2017-06-07T20:57',
             '2017-07-10T03:27', '2018-09-01T09:32', '2018-08-24T15:52',
             '2018-11-17T03:15', '2017-06-28T23:27', '2017-09-24T02:09',
             '2018-01-27T23:02', '2018-08-21T09:10', '2018-02-17T16:18',
             '2018-03-01T00:25', '2018-06-01T17:14', '2017-11-29T04:48',
             '2017-10-26T21:36', '2017-09-13T00:06', '2017-02-26T03:47',
             '2017-07-13T08:09', '2018-07-30T00:23', '2017-11-26T18:15',
             '2018-10-20T15:32', '2017-08-08T11:44', '2018-04-07T17:06',
             '2018-04-09T04:42', '2017-05-05T06:22', '2018-04-01T11:11',
             '2018-10-03T01:55', '2018-10-21T03:19', '2017-09-17T15:43',
             '2018-07-25T20:29', '2017-08-28T15:11', '2018-11-24T06:54',
             '2017-06-05T08:36', '2017-06-18T16:55'], dtype='datetime64[m]'
        )
        np.testing.assert_array_equal(s, exp_res)


class TestDateColumnMaker(object):

    def test__init(self):
        cm = DateColumnMaker(100, 208)
        assert cm.nrows == 100
        assert cm.seed == 208

    def test_make_col(self):
        cm = DateColumnMaker(50, 107)
        s = cm.make_col()
        assert len(s) == 50
        exp_res = np.array(
            ['2022-10-13', '2029-07-07', '2029-06-18', '2035-08-14',
             '2026-04-28', '2027-06-16', '2021-02-28', '2030-01-24',
             '2021-08-21', '2020-08-19', '2018-07-09', '2032-01-05',
             '2035-01-28', '2038-12-01', '2030-01-30', '2031-05-01',
             '2036-01-28', '2041-01-31', '2035-01-19', '2023-11-18',
             '2032-12-03', '2031-05-26', '2036-05-01', '2042-04-03',
             '2040-05-18', '2043-12-12', '2035-10-30', '2028-01-02',
             '2027-01-26', '2042-02-06', '2020-07-16', '2024-01-15',
             '2019-09-25', '2024-12-18', '2040-01-29', '2017-07-12',
             '2039-12-19', '2023-05-08', '2035-10-28', '2043-12-20',
             '2030-11-23', '2041-10-25', '2029-06-08', '2041-11-18',
             '2037-09-11', '2032-11-22', '2040-08-28', '2021-08-23',
             '2019-01-09', '2041-05-31'], dtype='datetime64[D]'
        )
        np.testing.assert_array_equal(s, exp_res)


class TestStringColumnMaker(object):

    def test__init(self):
        cm = StringColumnMaker(100, 208, 10)
        assert cm.nrows == 100
        assert cm.seed == 208
        assert cm.str_len == 10

    def test_make_col(self):
        cm = StringColumnMaker(50, 107, 7)
        s = cm.make_col()
        assert len(s) == 50
        for l in s:
            assert len(l) == 7
        exp_res = np.array(
            ['WzgolIP', 'MUpqsIv', 'CTP0GPi', 'edOW1Bp', '62jxtg2', 'FRzWBX8',
             'ColjKBu', 'Rvy0cnV', '5z47rM8', '4fREHVR', 'HhHNtu9', 'pOaUmCj',
             '9BulEQ6', 'RPS41gA', 'nExuCh5', 'ZciycYh', 'O5c76V2', 'EKDFpLC',
             'wfJqwkR', 'ieZXv1B', 'IzAwY6N', 'K6fegoI', 'JEB5Tu8', 'JL7Bcbr',
             'a6bztx8', 'IQghAdM', 'znweJFG', 'DwWrP4B', 'Db7Fl1j', 'ZZzhobu',
             '8Piz841', 'zUE5K9o', 'YKjZnb2', 'CgYHiTi', 'Nuv3yO8', 'fPowjeb',
             'KoM2uox', 'R35P1lB', 'udZUeXt', 'mjM5IFa', 'hH3MMip', '85jbGzV',
             '6Qn907i', 'CFel2rp', 'yn1qp8p', 'ziaw6oY', 'T04LxDh', 'BtTeAEK',
             'VDuZG6Y', '2sP9840'], dtype='<U7'
        )
        np.testing.assert_array_equal(s, exp_res)


class TestConstantStringColumnMaker(object):

    def test__init(self):
        cm = ConstantStringColumnMaker(100, 204, 10)
        assert cm.nrows == 100
        assert cm.seed == 204
        assert cm.str_len == 10

    def test_make_col(self):
        cm = ConstantStringColumnMaker(50, 70, 8)
        s = cm.make_col()
        assert len(s) == 50
        for l in s:
            assert len(l) == 8
            assert l == 'owYy87a0'

        exp_res = np.array(
            ['owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0',
             'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0', 'owYy87a0'],
            dtype='<U8'
        )
        np.testing.assert_array_equal(s, exp_res)


class TestConstantIntColumnMaker(object):

    def test__init(self):
        cm = ConstantIntColumnMaker(10, 24)
        assert cm.nrows == 10
        assert cm.seed == 24

    def test_make_col(self):
        cm = ConstantIntColumnMaker(20, 70)
        s = cm.make_col()
        assert len(s) == 20
        for k in s:
            assert k == 89422

        exp_res = np.array(
            [89422, 89422, 89422, 89422, 89422, 89422, 89422, 89422, 89422,
             89422, 89422, 89422, 89422, 89422, 89422, 89422, 89422, 89422,
             89422, 89422]
        )
        np.testing.assert_array_equal(s, exp_res)


class TestEnumColumnMaker(object):

    def test__init(self):
        cm = EnumColumnMaker(100, 2, 4)
        assert cm.nrows == 100
        assert cm.seed == 2
        assert cm.enum_vals == 4

    def test_make_col(self):
        cm = EnumColumnMaker(100, 2, 4)
        s = cm.make_col()
        assert len(s) == 100
        for k in s:
            assert k in range(0, 4)

        exp_res = np.array(
            [0, 3, 1, 0, 2, 3, 2, 3, 0, 3, 2, 1, 3, 3, 1, 3, 3, 3, 2, 0, 0, 0,
             1, 3, 3, 2, 0, 2, 3, 3, 3, 2, 2, 1, 2, 0, 3, 3, 1, 0, 0, 2, 2, 3,
             3, 1, 3, 2, 0, 0, 2, 0, 2, 0, 2, 3, 3, 1, 3, 3, 2, 0, 2, 1, 2, 3,
             1, 1, 0, 3, 1, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2, 2, 3, 0, 3, 0, 2, 2,
             0, 0, 2, 2, 0, 2, 0, 1, 2, 1, 2, 2]
        )
        np.testing.assert_array_equal(s, exp_res)
