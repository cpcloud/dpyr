import os
import operator

from typing import Union, Type, Callable

import pytest
import _pytest as pt

import pandas as pd
import pandas.util.testing as tm

import ibis
import ibis.expr.types as ir
import ibis.expr.datatypes as dt

from dpyr import (
    anti_join,
    cast,
    desc,
    distinct,
    do,
    groupby,
    head,
    inner_join,
    join,
    left_join,
    log,
    max,
    mean,
    min,
    mutate,
    n,
    nullif,
    nunique,
    outer_join,
    right_join,
    select,
    semi_join,
    sift,
    sort_by,
    std,
    sum,
    summarize,
    transmute,
    var,
    X, Y,

    exp,
    ln,
    log2,
    log10,
    floor,
    ceil,
    abs,
    round,
    sign,
    sqrt,

    lower,
    upper,
)

from dpyr.core import Unary


@pytest.fixture(scope='module')
def df() -> pd.DataFrame:
    path = os.environ.get('DIAMONDS_CSV', 'diamonds.csv')
    return pd.read_csv(path, index_col=None)


@pytest.fixture(scope='module')
def batting_df() -> pd.DataFrame:
    path = os.environ.get('BATTING_CSV', 'batting.csv')
    return pd.read_csv(path, index_col=None)


@pytest.fixture(
    params=[
        # ibis.postgres.connect(
            # database=os.environ.get('TEST_POSTGRES_DB', 'ibis_testing')
        # ),
        ibis.sqlite.connect(
            os.environ.get('TEST_SQLITE_DB', 'ibis_testing.db')
        ),
        # ibis.pandas.connect({
            # 'diamonds': df(), 'other_diamonds': df(), 'batting': batting_df()
        # })
    ],
    scope='module',
)
def client(request: pt.fixtures.FixtureRequest) -> ibis.client.Client:
    return request.param


@pytest.fixture
def diamonds(client: ibis.client.Client) -> ir.TableExpr:
    return client.table('diamonds').head(1000)


@pytest.fixture
def batting(client: ibis.client.Client) -> ir.TableExpr:
    return client.table('batting')


@pytest.fixture
def awards_players(client: ibis.client.Client) -> ir.TableExpr:
    return client.table('awards_players')


@pytest.fixture
def other_diamonds(client: ibis.client.Client) -> ir.TableExpr:
    return client.table('diamonds').view().head(1000)


def test_compound_expression(diamonds: ir.TableExpr) -> None:
    expected = diamonds[diamonds.price * diamonds.price / 2.0 >= 100]
    expected = expected.groupby('cut').aggregate([
        expected.carat.max().name('max_carat'),
        expected.carat.mean().name('mean_carat'),
        expected.carat.min().name('min_carat'),
        expected.x.count().name('n'),
        expected.carat.std().name('std_carat'),
        expected.carat.sum().name('sum_carat'),
        expected.carat.var().name('var_carat'),
    ])
    expected = expected.mutate(
        foo=expected.mean_carat,
        bar=expected.var_carat
    ).sort_by([ibis.desc('foo'), 'bar']).head()

    result = (
        diamonds >> sift(X.price * X.price / 2.0 >= 100)
                 >> groupby(X.cut)
                 >> summarize(
                     max_carat=max(X.carat),
                     mean_carat=mean(X.carat),
                     min_carat=min(X.carat),
                     n=n(X.x),
                     std_carat=std(X.carat),
                     sum_carat=sum(X.carat),
                     var_carat=var(X.carat),
                    )
                 >> mutate(foo=X.mean_carat, bar=X.var_carat)
                 >> sort_by(desc(X.foo), X.bar)
                 >> head(5)
    )
    assert result.equals(expected)
    tm.assert_frame_equal(expected.execute(), result >> do())


@pytest.mark.parametrize(
    'join_func',
    [
        inner_join,
        left_join,
        pytest.mark.xfail(right_join, raises=KeyError),
        outer_join,
        semi_join,
        anti_join,
    ]
)
def test_join(
    diamonds: ir.TableExpr,
    other_diamonds: ir.TableExpr,
    join_func: Type[join]
) -> None:
    result = (
        diamonds >> join_func(other_diamonds, on=X.cut == Y.cut)
                 >> select(X.x, Y.y)
    )
    join_func_name = join_func.__name__  # type: str
    joined = getattr(diamonds, join_func_name)(
        other_diamonds, diamonds.cut == other_diamonds.cut
    )
    expected = joined[diamonds.x, other_diamonds.y]
    assert result.equals(expected)


@pytest.mark.parametrize(
    'column',
    [
        'carat',
        'cut',
        'color',
        'clarity',
        'depth',
        'table',
        'price',
        'x',
        'y',
        'z',
        0,
    ] + list(range(1, 10))
)
def test_pull(diamonds: ir.TableExpr, column: Union[str, int]) -> None:
    result = diamonds >> X[column]
    expected = diamonds[column]
    assert result.equals(expected)
    tm.assert_series_equal(expected.execute(), result >> do())


def test_do(diamonds: ir.TableExpr) -> None:
    tm.assert_frame_equal(diamonds.execute(), diamonds >> do())


def test_simple_arithmetic(diamonds: ir.TableExpr) -> None:
    result = diamonds >> mean(X.carat) + 1
    expected = diamonds.carat.mean() + 1
    assert result.equals(expected)
    assert float(expected.execute()) == float(result >> do())


def test_mutate(diamonds: ir.TableExpr) -> None:
    result = diamonds >> mutate(new_column=X.carat + 1)
    expected = diamonds.mutate(new_column=lambda x: x.carat + 1)
    assert result.equals(expected)
    tm.assert_frame_equal(expected.execute(), result >> do())


def test_transmute(diamonds: ir.TableExpr) -> None:
    result = diamonds >> transmute(new_column=X.carat * 2)
    expected = diamonds[[(diamonds.carat * 2).name('new_column')]]
    assert result.equals(expected)
    tm.assert_frame_equal(expected.execute(), result >> do())


@pytest.mark.parametrize('to', ['string', dt.string])
def test_cast(
    diamonds: ir.TableExpr, to: Union[str, dt.DataType]
) -> None:
    result = diamonds >> cast(X.carat + 1, to=to)
    expected = (diamonds.carat + 1).cast(to)
    assert result.equals(expected)
    tm.assert_series_equal(expected.execute(), result >> do())


@pytest.mark.parametrize(
    'column',
    [
        'carat',
        'cut',
        'color',
        'clarity',
        'depth',
        'table',
        'price',
        'x',
        'y',
        'z',
    ]
)
def test_distinct(diamonds: ir.TableExpr, column: str) -> None:
    result = diamonds >> distinct(X[column])
    expected = diamonds[column].distinct()
    assert result.equals(expected)
    tm.assert_series_equal(expected.execute(), result >> do())


@pytest.mark.parametrize(
    'column',
    [
        'carat',
        'cut',
        'color',
        'clarity',
        'depth',
        'table',
        'price',
        'x',
        'y',
        'z',
    ]
)
def test_nunique(diamonds: ir.TableExpr, column: str) -> None:
    result = diamonds >> nunique(X[column])
    expected = diamonds[column].nunique()
    assert result.equals(expected)
    assert expected.execute() == result >> do()


@pytest.mark.parametrize(
    'func',
    [
        exp,
        ln,
        log2,
        log10,
        floor,
        ceil,
        abs,
        sign,
        sqrt,
    ]
)
def test_unary_math(diamonds: ir.TableExpr, func: Type[Unary]) -> None:
    result = diamonds >> func(cast(X.carat, to=dt.Decimal(19, 7)))
    expected = getattr(diamonds.carat.cast(dt.Decimal(19, 7)), func.__name__)()
    assert result.equals(expected)
    tm.assert_series_equal(result >> do(), expected.execute())


@pytest.mark.parametrize(
    'func',
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.pow,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ]
)
def test_binary_math(diamonds: ir.TableExpr, func: Callable) -> None:
    result = diamonds >> func(X.carat, X.z)
    expected = func(diamonds.carat, diamonds.z)
    assert result.equals(expected)
    tm.assert_series_equal(result >> do(), expected.execute())


@pytest.mark.parametrize(
    'base',
    [-2, -1, 1, 2],
)
def test_log(diamonds: ir.TableExpr, base: int) -> None:
    result_expr = diamonds >> log(nullif(X.carat, 0), base)
    expected_expr = diamonds.carat.nullif(0).log(base)
    assert result_expr.equals(expected_expr)
    result_df = result_expr >> do()
    expected_df = expected_expr.execute()
    tm.assert_series_equal(result_df, expected_df)


@pytest.mark.parametrize('places', list(range(-5, 6)))
def test_round(diamonds: ir.TableExpr, places: int) -> None:
    result = diamonds >> round(X.carat, places)
    expected = diamonds.carat.round(places)
    assert result.equals(expected)
    tm.assert_series_equal(result >> do(), expected.execute())


@pytest.mark.parametrize('func', [lower, upper])
def test_unary_string(diamonds: ir.TableExpr, func: Type[Unary]) -> None:
    result = diamonds >> func(X.cut)
    expected = getattr(diamonds.cut, func.__name__)()
    assert result.equals(expected)
    tm.assert_series_equal(result >> do(), expected.execute())


def test_column_slice(batting: ir.TableExpr) -> None:
    result = batting >> select(
        X.playerID, X.yearID, X.teamID, X.G, X['AB':'H']
    )
    columns = batting.columns
    expected = batting[
        ['playerID', 'yearID', 'teamID', 'G'] + [
            columns[i]
            for i in range(columns.index('AB'), columns.index('H') + 1)
        ]
    ]
    assert result.equals(expected)
