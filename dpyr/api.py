import operator

from typing import Union, Callable

import pandas as pd

import ibis
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.expr.datatypes as dt

from ibis.expr.groupby import GroupedTableExpr

from dpyr.core import Scope, Result
from dpyr.core import Value, Getter, Verb, Reduction, SpreadReduction
from dpyr.core import JoinKey, On
from dpyr.core import X, Y


class desc(Value):

    __slots__ = ()

    def __init__(self, expr: Value) -> None:
        super().__init__('desc', expr=expr)

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.Expr:
        assert self.expr is not None
        return ibis.desc(self.expr.resolve(expr, scope))


class groupby(Verb):

    __slots__ = 'keys',

    def __init__(self, *keys: Value) -> None:
        self.keys = keys

    def __call__(self, expr: ir.TableExpr) -> GroupedTableExpr:
        return expr.groupby([
            key.resolve(expr, {X: expr}) for key in self.keys
        ])


class select(Verb):

    __slots__ = 'columns',

    def __init__(self, *columns: Getter) -> None:
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


class sift(Verb):

    __slots__ = 'predicates',

    def __init__(self, *predicates: Value) -> None:
        self.predicates = predicates

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        scope = {X: expr}
        return expr.filter([
            predicate.resolve(expr, scope)
            for predicate in self.predicates
        ])


class summarize(Verb):

    __slots__ = 'metrics',

    def __init__(self, **metrics: Value) -> None:
        self.metrics = sorted(metrics.items(), key=operator.itemgetter(0))

    def __call__(self, grouped: GroupedTableExpr) -> ir.Expr:
        return grouped.aggregate([
            operation(grouped.table).name(name)
            for name, operation in self.metrics
        ])


class head(Value):

    __slots__ = 'n',

    def __init__(self, n: Union[int, Value]=5) -> None:
        self.n = n

    def resolve(self, expr: ir.TableExpr, scope: Scope) -> ir.TableExpr:
        return expr.head(
            self.n if isinstance(self.n, int) else self.n.resolve(expr, scope)
        )


class mean(Reduction):

    __slots__ = ()


class sum(Reduction):

    __slots__ = ()


class count(Reduction):

    __slots__ = ()


n = count


class var(SpreadReduction):

    __slots__ = ()


class std(SpreadReduction):

    __slots__ = ()


class min(Reduction):

    __slots__ = ()


class max(Reduction):

    __slots__ = ()


class nunique(Reduction):

    __slots__ = ()

    def __init__(self, column: Value) -> None:
        super().__init__(column)

    def resolve(self, expr: ir.Expr, scope: Scope) -> ir.ValueExpr:
        return self.func(self.column.resolve(expr, scope))()


class mutate(Verb):

    __slots__ = 'mutations',

    def __init__(self, **mutations: Value) -> None:
        self.mutations = mutations

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        return expr.mutate(**{
            name: column.resolve(expr, {X: expr})
            for name, column in self.mutations.items()
        })


class transmute(Verb):

    __slots__ = 'mutations',

    def __init__(self, **mutations: Value) -> None:
        self.mutations = mutations

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        columns = [
            column.resolve(expr, {X: expr}).name(name)
            for name, column in self.mutations.items()
        ]
        return expr.projection(columns)


class sort_by(Verb):

    __slots__ = 'sort_keys',

    def __init__(self, *sort_keys: Value) -> None:
        self.sort_keys = sort_keys

    def __call__(self, expr: ir.TableExpr) -> ir.TableExpr:
        return expr.sort_by([
            key.resolve(expr, {X: expr}) for key in self.sort_keys
        ])


class join(Verb):

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


class distinct(Value):

    __slots__ = 'expression',

    def __init__(self, expression: Value) -> None:
        self.expression = expression

    def resolve(self, expr: ir.ColumnExpr, scope: Scope) -> ir.ColumnExpr:
        return self.expression.resolve(expr, scope).distinct()


class cast(Value):

    __slots__ = 'value', 'to',

    def __init__(self, value: Value, to: Union[str, dt.DataType]) -> None:
        self.value = value
        self.to = to

    def resolve(self, table: ir.TableExpr, scope: Scope) -> ir.ValueExpr:
        return self.value.resolve(table, {X: table}).cast(self.to)


class do:

    __slots__ = 'execute',

    def __init__(
        self,
        execute: Callable[[ir.Expr], Result]=operator.methodcaller('execute')
    ) -> None:
        self.execute = execute

    def __call__(self, expr: ir.Expr) -> Result:
        return self.execute(expr)

    def __rrshift__(self, other: ir.Expr) -> Result:
        return self(other)


def from_dataframe(df: pd.DataFrame, name: str='t') -> ir.TableExpr:
    """Convert a pandas DataFrame into an ibis Table"""
    client = ibis.pandas.connect({name: df})  # type: ibis.pandas.PandasClient
    return client.table(name)