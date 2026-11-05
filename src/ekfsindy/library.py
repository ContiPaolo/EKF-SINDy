"""Symbolic SINDy feature libraries compiled to JAX."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from itertools import combinations_with_replacement

import jax
import jax.numpy as jnp
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


def _as_symbols(variables: Sequence[sp.Symbol | str]) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(v) if isinstance(v, str) else v for v in variables)


class SymbolicLibrary:
    """A SINDy library ``Theta(z) = [theta_1(z), ..., theta_m(z)]`` of symbolic features.

    ``variables`` is the ordered list of library inputs ``z`` (states, then parameters, then
    external inputs). Each feature is a SymPy expression in those variables; the library is
    compiled to a jitted JAX function, so it is differentiable with ``jax.jacfwd`` and friends.
    """

    def __init__(
        self,
        variables: Sequence[sp.Symbol | str],
        expressions: Sequence[sp.Expr | float],
        names: Sequence[str] | None = None,
    ):
        self.variables = _as_symbols(variables)
        self.expressions = tuple(sp.sympify(e) for e in expressions)
        self.names = tuple(names) if names is not None else tuple(str(e) for e in self.expressions)
        if len(self.names) != len(self.expressions):
            raise ValueError("names and expressions must have the same length")
        unknown = set().union(*(e.free_symbols for e in self.expressions)) - set(self.variables)
        if unknown:
            raise ValueError(f"expressions use symbols that are not variables: {sorted(map(str, unknown))}")

    # ---------------------------------------------------------------- constructors
    @classmethod
    def polynomial(
        cls,
        variables: Sequence[sp.Symbol | str],
        degree: int,
        include_bias: bool = True,
    ) -> SymbolicLibrary:
        """Monomials up to ``degree``, in the same order as ``pysindy.PolynomialLibrary``."""
        variables = _as_symbols(variables)
        expressions: list[sp.Expr] = [sp.Integer(1)] if include_bias else []
        for d in range(1, degree + 1):
            for combo in combinations_with_replacement(variables, d):
                expressions.append(sp.Mul(*combo))
        return cls(variables, expressions)

    @classmethod
    def from_feature_names(
        cls,
        names: Sequence[str],
        variables: Sequence[sp.Symbol | str],
    ) -> SymbolicLibrary:
        """Parse pysindy feature names such as ``"1"``, ``"u1 u2^2"`` into expressions.

        Only the given variables are accepted as names, so arbitrary code cannot be evaluated.
        """
        variables = _as_symbols(variables)
        local_dict = {str(v): v for v in variables}
        # parse_expr only sees SymPy number/symbol constructors and the library variables
        global_dict = {"Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
                       "Symbol": sp.Symbol, "__builtins__": {}}
        expressions = []
        for name in names:
            source = " * ".join(name.replace("^", "**").split())
            try:
                expressions.append(parse_expr(source, local_dict=local_dict,
                                              global_dict=global_dict,
                                              transformations=standard_transformations))
            except (NameError, SyntaxError, TypeError) as err:
                raise ValueError(f"cannot parse feature name {name!r}") from err
        return cls(variables, expressions, names)

    def __add__(self, other: SymbolicLibrary) -> SymbolicLibrary:
        """Concatenate the features of two libraries defined on the same variables."""
        if self.variables != other.variables:
            raise ValueError("libraries must share the same variables to be concatenated")
        return SymbolicLibrary(
            self.variables, self.expressions + other.expressions, self.names + other.names
        )

    def subset(self, indices: Sequence[int]) -> SymbolicLibrary:
        return SymbolicLibrary(
            self.variables,
            [self.expressions[i] for i in indices],
            [self.names[i] for i in indices],
        )

    # ---------------------------------------------------------------- properties
    @property
    def n_features(self) -> int:
        return len(self.expressions)

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    def __len__(self) -> int:
        return self.n_features

    def __repr__(self) -> str:
        return (f"SymbolicLibrary(variables={[str(v) for v in self.variables]}, "
                f"features={list(self.names)})")

    # ---------------------------------------------------------------- symbolic
    def symbolic_jacobian(self) -> sp.Matrix:
        """``d theta_i / d z_j`` as a SymPy matrix of shape ``(n_features, n_variables)``."""
        return sp.Matrix(self.expressions).jacobian(self.variables)

    # ---------------------------------------------------------------- numeric
    @cached_property
    def _features(self) -> Callable[[jax.Array], jax.Array]:
        terms = sp.lambdify(self.variables, list(self.expressions), modules="jax")

        def features(z: jax.Array) -> jax.Array:
            values = terms(*(z[i] for i in range(self.n_variables)))
            # constant features (e.g. the bias) come back as Python scalars
            return jnp.stack([jnp.broadcast_to(jnp.asarray(v, dtype=z.dtype), z.shape[1:])
                              for v in values])

        return features

    @cached_property
    def evaluate(self) -> Callable[[jax.Array], jax.Array]:
        """Jitted ``z -> Theta(z)`` for a single point ``z`` of shape ``(n_variables,)``."""
        return jax.jit(self._features)

    @cached_property
    def jacobian(self) -> Callable[[jax.Array], jax.Array]:
        """Jitted ``z -> d Theta / d z`` of shape ``(n_features, n_variables)``."""
        return jax.jit(jax.jacfwd(self._features))

    def __call__(self, z) -> jax.Array:
        """Evaluate on a point ``(n_variables,)`` or a batch ``(n_samples, n_variables)``."""
        z = jnp.asarray(z)
        if z.ndim == 1:
            return self.evaluate(z)
        return jax.vmap(self.evaluate)(z)

    # ---------------------------------------------------------------- pysindy bridge
    def to_pysindy(self):
        """Equivalent ``pysindy.CustomLibrary`` (one function of all variables per feature)."""
        import pysindy as ps

        # pysindy infers each function's arity from its positional arguments; lambdify keeps
        # all n_variables arguments, so every function is applied once to the full input.
        functions = [sp.lambdify(self.variables, expr, modules="numpy") for expr in self.expressions]
        function_names = [lambda *_, name=name: name for name in self.names]
        return ps.CustomLibrary(library_functions=functions, function_names=function_names)
