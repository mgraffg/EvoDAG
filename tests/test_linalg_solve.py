"""``EvoDAG.linalg_solve.compute_weight`` against a plain Python version of
the same algorithm, bit for bit. The reference uses ``array_utils.dot`` (the
sum of the products where both factors are not zero, adding in order), so it
checks that the kernel keeps adding the same numbers in the same order."""
import math
import sys

import numpy as np
import pytest

from EvoDAG import array_utils as au
from EvoDAG.linalg_solve import compute_weight

EPS = sys.float_info.epsilon


def iszero(x):
    return not (x >= EPS or x <= -EPS)


def zero_round(x):
    return 0.0 if iszero(x) else x


def gauss_jordan(m):
    size = len(m)
    for i in range(size):
        pos, best = i, abs(m[i][i])
        for j in range(i + 1, size):
            if abs(m[j][i]) > best:
                best, pos = abs(m[j][i]), j
        if pos != i:
            m[i], m[pos] = m[pos], m[i]
        c_den = m[i][i]
        if iszero(c_den):
            return True
        for j in range(i + 1, size):
            c = m[j][i] / c_den
            for k in range(i, size + 1):
                m[j][k] = m[j][k] - m[i][k] * c
    for i in range(size - 1, -1, -1):
        c = m[i][i]
        if iszero(c):
            return True
        for j in range(i):
            m[j][size] = zero_round(m[j][size] - m[i][size] * m[j][i] / c)
        tmp = m[i][size] / c
        if not math.isfinite(tmp):
            return True
        m[i][size] = zero_round(tmp)
    return False


def reference(r, ytr, mask):
    if not isinstance(mask, np.ndarray):
        mask = np.full(ytr.shape[0], float(mask))
    index = np.flatnonzero(mask)
    support = mask[index]
    columns, var, dependent = [], [], []
    for i, ri in enumerate(r):
        tmp = au.dot(ytr, ri)
        if not math.isfinite(tmp):
            return None
        out = np.zeros(len(index))
        for k in range(len(index)):
            x, m = ri[index[k]], support[k]
            out[k] = 0.0 if x == 0 or m == 0 else x * m
        if np.any(out != 0):
            columns.append(out)
            var.append(i)
            dependent.append(tmp)
    size = len(columns)
    X = [[0.0] * (size + 1) for _ in range(size)]
    for i in range(size):
        X[i][size] = dependent[i]
        for j in range(i, size):
            tmp = au.dot(columns[i], columns[j])
            if not math.isfinite(tmp):
                return None
            X[i][j] = X[j][i] = tmp
    if gauss_jordan(X):
        return None
    res = np.zeros(len(r))
    for k, i in enumerate(var):
        res[i] = X[k][size]
    return res


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(a, b, equal_nan=True) and \
        np.signbit(a).tolist() == np.signbit(b).tolist()


def make_case(rs, n, size, density, kind):
    ytr = np.where(rs.rand(n) < 0.6, (rs.rand(n) > 0.5).astype(float), 0.0)
    if kind == "binary":
        mask = (rs.rand(n) < 0.5).astype(float)
    elif kind == "weighted":
        mask = np.where(rs.rand(n) < 0.5, rs.rand(n) * 3, 0.0)
    else:
        mask = np.ones(n)
    mask.flags.writeable = False
    r = []
    for _ in range(size):
        v = np.where(rs.rand(n) < density, rs.randn(n), 0.0)
        r.append(v)
    return r, ytr, mask


@pytest.mark.parametrize("kind", ["binary", "weighted", "ones"])
@pytest.mark.parametrize("density", [1.0, 0.5, 0.05])
def test_compute_weight_equals_the_plain_algorithm(kind, density):
    rs = np.random.RandomState(1)
    for n, size in ((50, 1), (200, 2), (300, 4), (120, 5), (37, 3), (150, 9),
                    (90, 12), (64, 13)):
        r, ytr, mask = make_case(rs, n, size, density, kind)
        assert same(compute_weight(r, ytr, mask),
                    reference(r, ytr, mask)), (kind, density, n, size)


def test_compute_weight_with_extreme_values():
    rs = np.random.RandomState(2)
    r, ytr, mask = make_case(rs, 100, 3, 1.0, "binary")
    cases = []
    # products that underflow to -0.0 and to +0.0, huge values, negative zero
    tiny = [x.copy() for x in r]
    tiny[0][:] *= 1e-170
    tiny[1][:] *= -1e-170
    cases.append((tiny, ytr * 1e-170))
    huge = [x.copy() for x in r]
    huge[0][:] *= 1e100
    cases.append((huge, ytr))
    zeros = [x.copy() for x in r]
    zeros[1][:] = -0.0
    zeros[2][::2] = -0.0
    cases.append((zeros, ytr))
    # non-finite values: the result is None, as in the reference
    for bad in (np.inf, -np.inf, np.nan):
        values = [x.copy() for x in r]
        values[1][3] = bad
        cases.append((values, ytr))
    big = [x.copy() for x in r]
    big[0][:] = 1e200
    cases.append((big, ytr))
    for values, y in cases:
        assert same(compute_weight(values, y, mask),
                    reference(values, y, mask))
    # the same inside a longer list (the arguments are taken four at a time
    # when they are all finite) and a product that overflows
    r9, y9, m9 = make_case(rs, 60, 11, 1.0, "binary")
    assert same(compute_weight(r9, y9, m9), reference(r9, y9, m9))
    for position in (0, 5, 10):
        for bad in (np.inf, np.nan):
            values = [x.copy() for x in r9]
            values[position][7] = bad
            assert same(compute_weight(values, y9, m9),
                        reference(values, y9, m9))
    values = [x.copy() for x in r9]
    values[6][:] = 1e200
    values[7][:] = 1e200
    assert same(compute_weight(values, y9, m9), reference(values, y9, m9))
    # a scalar mask and a mask without support
    r2, y2, _ = make_case(rs, 80, 2, 1.0, "ones")
    assert same(compute_weight(r2, y2, 1.0), reference(r2, y2, 1.0))
    none = np.zeros(80)
    none.flags.writeable = False
    assert same(compute_weight(r2, y2, none), reference(r2, y2, none))
    # collinear arguments: the system is singular
    same_arg = [r2[0], r2[0].copy()]
    assert same(compute_weight(same_arg, y2, 1.0),
                reference(same_arg, y2, 1.0))
