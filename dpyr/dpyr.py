import abc
import operator

from typing import Union, Optional, Dict, Callable, List

import pandas as pd

import ibis
import ibis.expr.types as ir
import ibis.expr.operations as ops


class Keyed:
    """Objects that can be accessed by ``__getitem__`` or ``__getattr__``.

    Parameters
    ----------
    name : str
    """

    __slots__ = ()

    def __getitem__(self, name: str) -> 'Item':
        return Item(name, self)

    def __getattr__(self, name: str) -> 'Attribute':
        if name.startswith('_') and name.endswith('_'):
            raise AttributeError(name)
        return Attribute(name, self)


class BinaryOperations:

    """A mixin class implementing binary operations."""

    __slots__ = ()

    def __add__(self, other: 'Value') -> 'Add':
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __div__(self, other):
        return Div(self, other)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

    def __pow__(self, other):
        return Pow(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __eq__(self, other):
        return Eq(self, other)

    def __ne__(self, other):
        return Ne(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Ge(self, other)


Scope = Optional[Union[Dict[Keyed, ir.Expr], Dict['Value', ir.Expr]]]


class Value(Keyed, BinaryOperations):

    """A generic value class forming the basis for dpyr expressions.

    Parameters
    ----------
    name : str
    parent : Value
    """

    __slots__ = 'name', 'parent'

    def __init__(
        self,
        name: Optional[str]=None,
        parent: Optional[Union[Keyed, 'Value']]=None
    ) -> None:
        self.name = name
        self.parent = parent

    def __hash__(self) -> int:
        return hash((self.name, self.parent))

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        return scope.get(self, expr)


class Binary(Value, metaclass=abc.ABCMeta):

    """A class that implements :meth:`dpyr.Value.resolve` for binary
    operations.
    """

    __slots__ = 'left', 'right'

    def __init__(self, left: Value, right: Value) -> None:
        self.left = left
        self.right = right

    @abc.abstractmethod
    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        pass

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        try:
            left = self.left.resolve(expr, scope)
        except AttributeError:
            left = self.left
        try:
            right = self.right.resolve(expr, scope)
        except AttributeError:
            right = self.right
        return self.operate(left, right)

    def __call__(self, other: ir.Expr) -> ir.Expr:
        return self.resolve(other, {X: other})


class Unary(Value):

    __slots__ = 'operand',

    def __init__(self, operand: Value) -> None:
        self.operand = operand

    @abc.abstractmethod
    def operate(self, expr: ir.Expr) -> ir.Expr:
        pass

    def resolve(self, operand: ir.Expr, scope: Scope) -> ir.Expr:
        return self.operate(self.operand.resolve(operand, scope))


class Add(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left + right


class Sub(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left - right


class Mul(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left * right


class Div(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left / right


class FloorDiv(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left // right


class Pow(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left ** right


class Mod(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left % right


class Eq(Binary):

    __slots__ = ()

    def operate(self, left: ir.Expr, right: ir.Expr) -> ir.Expr:
        return left == right 
    operate = operator.eq


class Ne(Binary):

    __slots__ = ()

    operate = operator.ne


class Lt(Binary):

    __slots__ = ()

    operate = operator.lt


class Le(Binary):

    __slots__ = ()

    operate = operator.le


class Gt(Binary):

    __slots__ = ()

    @property
    def operate(self):
        return operator.gt


class Ge(Binary):

    __slots__ = ()

    @property
    def operate(self):
        return operator.ge


class Getter(Value):

    __slots__ = ()

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        try:
            parent = scope[self.parent]
        except KeyError:
            parent = expr
        return parent[self.name]

    def __call__(self, expr: ir.Expr) -> ir.Expr:
        return self.resolve(expr, {X: expr})


class Attribute(Getter):

    __slots__ = ()

    def __repr__(self) -> str:
        return '{0.parent}.{0.name}'.format(self)


class Item(Getter):

    __slots__ = ()

    def __repr__(self) -> str:
        return '{0.parent}[{0.name!r}]'.format(self)


class desc(Value):

    __slots__ = ()

    def __init__(self, expr: Value) -> None:
        super().__init__(parent=expr)

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        return ibis.desc(self.parent.resolve(expr, scope))


X = Value('X')
Y = Value('Y')


class Verb(metaclass=abc.ABCMeta):

    __slots__ = ()

    @abc.abstractmethod
    def __call__(self, other: ir.Expr) -> ir.Expr:
        pass

    def __rrshift__(self, other: ir.Expr) -> ir.Expr:
        return self(other)

    def resolve(self, other: ir.Expr, scope: Scope) -> ir.Expr:
        return self(other)


class groupby(Verb, Keyed):

    __slots__ = 'keys',

    def __init__(self, *keys: Value) -> None:
        self.keys = keys

    def __call__(self, expr: ir.TableExpr) -> ir.GroupedTableExpr:
        return expr.groupby([
            key.resolve(expr, {X: expr}) for key in self.keys
        ])


class select(Verb, Keyed):

    __slots__ = 'columns',

    def __init__(self, *columns: Union[Item, Attribute]) -> None:
        self.columns = columns

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        op = expr.op()
        if isinstance(op, ops.Join):
            scope = {X: op.left, Y: op.right}
        else:
            scope = {X: expr}
        return expr.projection([
            column.resolve(expr, scope) for column in self.columns
        ])


class sift(Verb, Keyed):

    __slots__ = 'predicates',

    def __init__(self, *predicates: Value) -> None:
        self.predicates = predicates

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        scope = {X: expr}
        return expr.filter([
            predicate.resolve(expr, scope)
            for predicate in self.predicates
        ])


class summarize(Verb, Keyed):

    __slots__ = 'metrics',

    def __init__(self, **metrics: ir.Expr) -> None:
        self.metrics = sorted(metrics.items(), key=operator.itemgetter(0))

    def __call__(self, grouped: ir.GroupedTableExpr) -> ir.Expr:
        return grouped.aggregate([
            operation(grouped.table).name(name)
            for name, operation in self.metrics
        ])


class head(Verb, Keyed):

    __slots__ = 'n',

    def __init__(self, n: Union[int, Value]=5) -> None:
        self.n = n

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        return expr.head(self.n)


class Reduction(Value):

    __slots__ = 'column', 'where', 'func',

    def __init__(self, column: Value, where: Optional[Value]=None) -> None:
        self.column = column
        self.where = where
        self.func = operator.attrgetter(type(self).__name__.lower())

    def __call__(self, expr: ir.Expr) -> ir.Expr:
        where = self.where
        scope = {X: expr}
        column = self.column.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where
        )


class mean(Reduction):

    __slots__ = ()


class sum(Reduction):

    __slots__ = ()


class count(Reduction):

    __slots__ = ()


n = count


class SpreadReduction(Reduction):

    __slots__ = 'how',

    def __init__(
        self, column: Value, where: Optional[Value]=None, how: str='sample'
    ) -> None:
        super().__init__(column, where=where)
        self.how = how

    def __call__(self, expr: ir.Expr) -> Union[ir.ColumnExpr, ir.ScalarExpr]:
        where = self.where
        scope = {X: expr}
        column = self.column.resolve(expr, scope)
        return self.func(column)(
            where=where.resolve(expr, scope) if where is not None else where,
            how=self.how
        )


class var(SpreadReduction):

    __slots__ = ()


class std(SpreadReduction):

    __slots__ = ()


class min(Reduction):

    __slots__ = ()


class max(Reduction):

    __slots__ = ()


class mutate(Verb, Keyed):

    __slots__ = 'mutations',

    def __init__(self, **mutations: ir.ColumnExpr) -> None:
        self.mutations = mutations

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        return expr.mutate(**{
            name: column.resolve(expr, {X: expr})
            for name, column in self.mutations.items()
        })


class transmute(Verb, Keyed):

    __slots__ = 'mutations',

    def __init__(self, **mutations: ir.ColumnExpr) -> None:
        self.mutations = mutations

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        columns = [
            column.name(name) for name, column in self.mutations.items()
        ]
        return expr.projection(columns)


class sort_by(Verb, Keyed):

    __slots__ = 'sort_keys',

    def __init__(self, *sort_keys: Value) -> None:
        self.sort_keys = sort_keys

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        return expr.sort_by([
            key.resolve(expr, {X: expr}) for key in self.sort_keys
        ])


JoinKey = Union[List[Value], List[str]]


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


class join(Verb, Keyed):

    __slots__ = 'right', 'on', 'how',

    def __init__(
        self, right: ir.TableExpr, on: JoinKey, how: str='inner'
    ) -> None:
        self.right = right
        self.on = On(right, on)
        self.how = how

    def __call__(self, left: ir.TableExpr) -> ir.TableExpr:
        right = self.right
        on = self.on.resolve(left, {X: left, Y: right})
        return left.join(right, on, how=self.how)


class inner_join(join):

    __slots__ = ()


class left_join(join):

    __slots__ = ()

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        super().__init__(right, on, how='left')


class right_join(join):

    __slots__ = ()

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        super().__init__(right, on, how='right')


class outer_join(join):

    __slots__ = ()

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        super().__init__(right, on, how='outer')


class semi_join(join):

    __slots__ = ()

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        super().__init__(right, on, how='semi')


class anti_join(join):

    __slots__ = ()

    def __init__(self, right: ir.TableExpr, on: JoinKey) -> None:
        super().__init__(right, on, how='anti')


class distinct(Verb):

    def __init__(
        self, **expressions: Union[ir.TableExpr, ir.ColumnExpr]
    ) -> None:
        self.expressions = expressions

    def __call__(self, expr):
        return expr.projection([
            e.distinct().name(name) for name, e in self.expressions.items()
        ])


class do(object):

    __slots__ = 'execute',

    def __init__(
        self, execute: Callable=operator.methodcaller('execute')
    ) -> None:
        self.execute = execute

    def __call__(self, expr: ir.Expr) -> Union[
        pd.DataFrame, pd.Series, str, float, int
    ]:
        return self.execute(expr)


def from_dataframe(
    df: pd.DataFrame, name: str='t'
) -> ibis.pandas.PandasClient:
    return ibis.pandas.connect({name: df}).table(name)
