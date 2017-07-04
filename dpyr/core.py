import abc
import operator

from typing import Union, Optional, Dict, List

import numpy as np
import pandas as pd

import ibis.expr.types as ir


class Keyed:
    """Objects that can be accessed by ``__getitem__`` or ``__getattr__``.
    """

    __slots__ = ()

    def __getitem__(self, name: Union[str, int]) -> 'Item':
        return Item(name, self)  # type: ignore

    def __getattr__(self, name: str) -> 'Attribute':
        if name.startswith('_') and name.endswith('_'):
            raise AttributeError(name)
        return Attribute(name, self)  # type: ignore


Operand = Union[
    'Value', str, int, float, np.str_, np.bytes_, np.integer, np.floating
]
Scope = Dict[Operand, ir.Expr]


class Shiftable(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, other: ir.Expr) -> ir.Expr:
        pass

    def __rrshift__(self, other: ir.Expr) -> ir.Expr:
        return self(other)


class Resolvable(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        pass


class Value(Keyed, Resolvable, Shiftable):

    """A generic value class forming the basis for dpyr expressions.

    Parameters
    ----------
    name : str
        The name of the expression
    expr : Value
        The expression
    """

    __slots__ = 'name', 'expr'

    def __init__(self, name: Optional[str], expr: Optional[Operand]) -> None:
        self.name = name
        self.expr = expr

    def __hash__(self) -> int:
        return hash((self.name, self.expr))

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        return expr

    def __call__(self, expr: ir.Expr) -> ir.Expr:
        return self.resolve(expr, {X: expr})

    def __add__(self, other: Operand) -> 'Add':
        return Add(self, other)

    def __sub__(self, other: Operand) -> 'Sub':
        return Sub(self, other)

    def __mul__(self, other: Operand) -> 'Mul':
        return Mul(self, other)

    def __truediv__(self, other: Operand) -> 'Div':
        return Div(self, other)

    def __div__(self, other: Operand) -> 'Div':
        return Div(self, other)

    def __floordiv__(self, other: Operand) -> 'FloorDiv':
        return FloorDiv(self, other)

    def __pow__(self, other: Operand) -> 'Pow':
        return Pow(self, other)

    def __mod__(self, other: Operand) -> 'Mod':
        return Mod(self, other)

    def __eq__(self, other: Operand) -> 'Eq':  # type: ignore
        return Eq(self, other)

    def __ne__(self, other: Operand) -> 'Ne':  # type: ignore
        return Ne(self, other)

    def __lt__(self, other: Operand) -> 'Lt':
        return Lt(self, other)

    def __le__(self, other: Operand) -> 'Le':
        return Le(self, other)

    def __gt__(self, other: Operand) -> 'Gt':
        return Gt(self, other)

    def __ge__(self, other: Operand) -> 'Ge':
        return Ge(self, other)

    def __invert__(self) -> 'Not':
        return Not(self)

    def __neg__(self) -> 'Negate':
        return Negate(self)


class Binary(Value, metaclass=abc.ABCMeta):

    """A class that implements :meth:`dpyr.Value.resolve` for binary
    operations.
    """

    __slots__ = 'left', 'right'

    def __init__(self, left: Operand, right: Operand) -> None:
        self.left = left
        self.right = right

    @abc.abstractmethod
    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        pass

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        if isinstance(self.left, Resolvable):
            left = self.left.resolve(expr, scope)
        else:
            left = self.left

        if isinstance(self.right, Resolvable):
            right = self.right.resolve(expr, scope)
        else:
            right = self.right

        return self.operate(left, right)


class Unary(Value, metaclass=abc.ABCMeta):

    __slots__ = ()

    def __init__(self, operand: Operand) -> None:
        super().__init__(None, operand)

    @abc.abstractmethod
    def operate(self, expr: ir.ValueExpr) -> ir.ValueExpr:
        pass

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        assert self.expr is not None
        return self.operate(self.expr.resolve(expr, scope))


class Not(Unary):

    __slots__ = ()

    def operate(self, expr: ir.BooleanValue) -> ir.BooleanValue:
        return ~expr


class Negate(Unary):

    __slots__ = ()

    def operate(self, expr: ir.ValueExpr) -> ir.ValueExpr:
        return -expr


class Add(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left + right


class Sub(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left - right


class Mul(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left * right


class Div(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left / right


class FloorDiv(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left // right


class Pow(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left ** right


class Mod(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left % right


class Eq(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left == right


class Ne(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left != right


class Lt(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left < right


class Le(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left <= right


class Gt(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left > right


class Ge(Binary):

    __slots__ = ()

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        return left >= right


class Getter(Value):

    __slots__ = ()

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        assert self.expr is not None

        try:
            result = scope[self.expr]
        except KeyError:
            result = expr

        return result[self.name]


class Attribute(Getter):

    __slots__ = ()

    def __repr__(self) -> str:
        return '{0.expr}.{0.name}'.format(self)


class Item(Getter):

    __slots__ = ()

    def __repr__(self) -> str:
        return '{0.expr}[{0.name!r}]'.format(self)


X = Value('X', None)
Y = Value('Y', None)


class Verb(Keyed, Shiftable, Resolvable):

    __slots__ = ()

    def resolve(self, other: ir.Expr, scope: Scope) -> ir.Expr:
        return self(other)


class Reduction(Value):

    __slots__ = 'column', 'where', 'func',

    def __init__(self, column: Value, where: Optional[Value]=None) -> None:
        self.column = column
        self.where = where
        self.func = operator.attrgetter(type(self).__name__.lower())

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.ValueExpr:
        where = self.where
        column = self.column.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where
        )


class SpreadReduction(Reduction):

    __slots__ = 'how',

    def __init__(
        self, column: Value, where: Optional[Value]=None, how: str='sample'
    ) -> None:
        super().__init__(column, where=where)
        self.how = how

    def __call__(self, expr: ir.ColumnExpr) -> ir.ValueExpr:
        where = self.where
        scope = {X: expr}
        column = self.column.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where,
            how=self.how
        )


JoinKey = Union[Value, List[Value], List[str]]


class On:

    __slots__ = 'right', 'on',

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        self.right = right
        self.on = on

    def resolve(
        self, left: ir.TableExpr, scope: Scope
    ) -> Union[ir.BooleanColumn, List[Union[ir.BooleanColumn, str]]]:
        if isinstance(self.on, Value):
            return self.on.resolve(left, scope)
        else:
            return self.on


Result = Union[
    pd.DataFrame, pd.Series, str, float, int, np.integer, np.floating
]
