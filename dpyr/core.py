import abc
import operator

from typing import Union, Optional, Dict, List
from typing import Any  # noqa: F401
from typing import Tuple  # noqa: F401

import numpy as np

import ibis
import ibis.expr.types as ir


class Keyed:
    """Objects that can be accessed by ``__getitem__`` or ``__getattr__``."""

    __slots__ = ()

    def __getitem__(self, name: Union[str, int, slice]) -> 'Item':
        if isinstance(name, slice):
            return ColumnSlice(self, name.start, name.stop)
        return Item(self, name)

    def __getattr__(self, name: str) -> 'Attribute':
        if name.startswith('_') and name.endswith('_'):
            raise AttributeError(name)
        return Attribute(self, name)


RawScalar = (
    str, int, float, np.str_, np.bytes_, np.integer, np.floating, type(None),
    slice
)
Scalar = Optional[Union[
    str, int, float, np.str_, np.bytes_, np.integer, np.floating, slice
]]
Operand = Optional[Union[
    'Value', str, int, float, np.str_, np.bytes_, np.integer, np.floating,
    None, slice
]]
Scope = Dict['Value', ir.Expr]


class Shiftable(metaclass=abc.ABCMeta):
    """Objects whose ``__call__`` method is invoked by right shifting."""

    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, other: ir.Expr) -> ir.Expr:
        pass

    def __rrshift__(self, other: ir.Expr) -> ir.Expr:
        return self(other)


class Resolvable(metaclass=abc.ABCMeta):
    """Objects that resolve to an ibis expression."""

    __slots__ = ()

    @abc.abstractmethod
    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        pass


class Value(Keyed, Shiftable, Resolvable):
    """A generic value class forming the basis for dpyr expressions.

    Parameters
    ----------
    expr : Optional[Operand]
        The expression
    name : Optional[str]
        The name of the expression
    """

    __slots__ = 'exprs', 'name'

    def __init__(self, *exprs: Optional[Operand]) -> None:
        self.exprs = tuple(
            Literal(expr) if isinstance(expr, RawScalar) else expr
            for expr in exprs
        )  # type: Tuple[Value, ...]
        self.name = type(self).__name__

    @property
    def expr(self) -> 'Value':
        return self.exprs[0]

    def __repr__(self) -> str:
        arguments = [expr for expr in self.exprs if expr is not None]

        if not arguments:
            return self.name
        return '{}({})'.format(self.name, ', '.join(map(repr, arguments)))

    def __hash__(self) -> int:
        exprs = self.exprs  # type: Tuple[Any, ...]
        return hash(exprs + (self.name,))

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

    def __truediv__(self, other: Operand) -> 'TrueDiv':
        return TrueDiv(self, other)

    def __div__(self, other: Operand) -> 'TrueDiv':
        return TrueDiv(self, other)

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


class Literal(Value):

    __slots__ = 'exprs', 'name'

    def __init__(self, value: Optional[Scalar]) -> None:
        self.exprs = (value,)
        self.name = str(value)  # type: str

    def __repr__(self) -> str:
        return repr(self.exprs[0])

    def resolve(self, expr: ir.ScalarExpr, scope: Scope) -> Scalar:
        return self.expr


class Binary(Value):
    """A class that implements :meth:`dpyr.Value.resolve` for binary
    operations such as ``+``, ``*``, etc.
    """

    __slots__ = ()

    @property
    def left(self) -> Value:
        assert len(self.exprs) == 2
        return self.exprs[0]

    @property
    def right(self) -> Value:
        assert len(self.exprs) == 2
        return self.exprs[1]

    def operate(self, left: ir.ValueExpr, right: ir.ValueExpr) -> ir.ValueExpr:
        function = getattr(operator, type(self).__name__.lower())
        return function(left, right)

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        left = self.left.resolve(expr, scope)
        right = self.right.resolve(expr, scope)
        return self.operate(left, right)


class Unary(Value):
    """A class implementing :meth:`dpyr.Value.resolve` for unary operations
    such as ``~`` and ``-``.
    """

    __slots__ = ()

    def operate(self, expr: ir.ValueExpr) -> ir.ValueExpr:
        method = getattr(expr, type(self).__name__)
        return method()

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
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


class Sub(Binary):

    __slots__ = ()


class Mul(Binary):

    __slots__ = ()


class TrueDiv(Binary):

    __slots__ = ()


class FloorDiv(Binary):

    __slots__ = ()


class Pow(Binary):

    __slots__ = ()


class Mod(Binary):

    __slots__ = ()


class Eq(Binary):

    __slots__ = ()


class Ne(Binary):

    __slots__ = ()


class Lt(Binary):

    __slots__ = ()


class Le(Binary):

    __slots__ = ()


class Gt(Binary):

    __slots__ = ()


class Ge(Binary):

    __slots__ = ()


class Getter(Value):
    """Parent class implementing resolution for :class:`dpyr.core.Attribute`
    and :class:`dpyr.core.Item` objects."""

    __slots__ = ()

    @property
    def index(self) -> Value:
        return self.exprs[1]

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.ColumnExpr:
        try:
            result = scope[self.expr]
        except KeyError:
            result = expr

        return result[self.index.resolve(expr, scope)]


class Attribute(Getter):

    __slots__ = ()

    def __init__(self, expr: Keyed, name: str) -> None:
        super().__init__(expr, name)

    def __repr__(self) -> str:
        return '{}.{}'.format(self.exprs[0], self.exprs[1].exprs[0])


class Item(Getter):

    __slots__ = ()

    def __init__(self, expr: Keyed, index: Union[str, int]) -> None:
        super().__init__(expr, index)

    def __repr__(self) -> str:
        return '{}[{!r}]'.format(self.exprs[0], self.exprs[1].exprs[0])


class ColumnSlice(Item):

    __slots__ = 'start', 'stop'

    def __init__(self, expr: Keyed, start: str, stop: Optional[str]) -> None:
        self.start = start
        self.stop = stop

    def __repr__(self) -> str:
        if self.stop is None:
            return '{}:'.format(self.start)
        return '{}:{}'.format(self.start, self.stop)

    def resolve(self, expr: ir.Expr, scope: Scope) -> List[ir.ColumnExpr]:
        schema = expr.schema()
        columns = schema.names
        name_locs = schema._name_locs
        start = name_locs[self.start]
        stop = name_locs.get(self.stop, len(expr.columns) - 1) + 1  # inclusive
        return [expr[columns[i]] for i in range(start, stop)]


X = Value()
X.name = 'X'
Y = Value()
Y.name = 'Y'


class Verb(Keyed, Shiftable, Resolvable):
    """Operations whose ``resolve`` method is equal to their ``__call__``
    method.
    """

    __slots__ = ()

    def resolve(self, other: ir.Expr, scope: Scope) -> ir.Expr:
        return self(other)


class Reduction(Value):

    __slots__ = 'func',

    def __init__(self, column: Value, where: Optional[Value]=None) -> None:
        super().__init__(column, where)
        self.func = operator.attrgetter(type(self).__name__.lower())

    @property
    def where(self) -> Resolvable:
        return self.exprs[1]

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.ValueExpr:
        where = self.where
        column = self.expr.resolve(expr, scope)
        return self.func(column)(where=where.resolve(expr, scope))


class SpreadReduction(Reduction):

    __slots__ = 'how',

    def __init__(
        self, column: Value, where: Optional[Value]=None, how: str='sample'
    ) -> None:
        super().__init__(column, where=where)
        self.how = how

    def __call__(self, expr: ir.ColumnExpr) -> ir.ValueExpr:
        where = self.where
        scope = {X: expr}  # type: Scope
        column = self.expr.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where,
            how=self.how
        )


JoinKey = Union[Value, List[Value], str, List[str]]


class On:

    """Class representing the condition of a join expression.

    Parameters
    ----------
    right : ir.TableExpr
        The right side relation to join against
    on : JoinKey
        The join condition
    """

    __slots__ = 'right', 'on',

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        self.right = right
        self.on = [
            Literal(expr) if isinstance(expr, str) else expr
            for expr in ibis.util.promote_list(on)
        ]

    def resolve(
        self, left: ir.TableExpr, scope: Scope
    ) -> List[Union[ir.BooleanColumn, str]]:
        return [on.resolve(left, scope) for on in self.on]
