from typing import Union

import pytest

from dpyr.core import Value, X, Y, Literal


@pytest.mark.parametrize(
    ('variable', 'expected'),
    [
        (X, 'X'),
        (Y, 'Y'),
    ]
)
def test_X_Y(variable: Value, expected: str) -> None:
    assert repr(variable) == expected


def test_attribute_repr() -> None:
    assert repr(X.foo) == 'X.foo'


@pytest.mark.parametrize('value', [1, 'a', 'bar baz'])
def test_item_repr(value: Union[int, str]) -> None:
    assert repr(X.foo[value]) == "X.foo[{!r}]".format(value)


@pytest.mark.parametrize('value', [1, 'a', 3.42])
def test_literal_repr(value: Union[int, str, float]) -> None:
    assert repr(Literal(value)) == repr(value)
