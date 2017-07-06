from typing import Union

import pytest

from dpyr.core import X, Y, Literal


def test_X_Y() -> None:
    assert repr(X) == 'X'
    assert repr(Y) == 'Y'


def test_attribute_repr() -> None:
    assert repr(X.foo) == 'X.foo'


def test_item_repr() -> None:
    assert repr(X.foo['a']) == "X.foo['a']"


@pytest.mark.parametrize('value', [1, 'a', 3.42])
def test_literal_repr(value: Union[int, str, float]) -> None:
    assert repr(Literal(value)) == repr(value)
