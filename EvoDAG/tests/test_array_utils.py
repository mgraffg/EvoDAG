# Copyright 2024 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numpy as np
from EvoDAG import array_utils as au

INF = float('inf')
NAN = float('nan')


def vec(*values):
    return np.array(values, dtype=float)


def sparse_data(seed=0, n=200):
    "A vector with ~70% zeros and one with zeros and non finite values"
    rs = np.random.RandomState(seed)
    a = rs.randn(n) * (rs.rand(n) > 0.7)
    b = rs.randn(n) * (rs.rand(n) > 0.7)
    return a, b


def test_finite_in_place():
    a = vec(1, INF, -INF, NAN, 0, 2)
    r = au.finite(a)
    assert r is a
    assert np.array_equal(a, vec(1, 0, 0, 0, 0, 2))
    b = vec(0, 0, 0)
    assert au.finite(b) is b


def test_isfinite():
    assert au.isfinite(vec(0, 1, -2))
    assert au.isfinite(vec())
    assert not au.isfinite(vec(0, 1, NAN))
    assert not au.isfinite(vec(INF, 0))
    assert not au.isfinite(vec(0, -INF))


def test_density():
    assert au.density(vec(0, 0, 0, 0)) == 0
    assert au.density(vec(1, 0, 2, 0)) == 0.5
    assert au.density(vec(1, -0.0, NAN, INF)) == 0.75
    a, _ = sparse_data()
    assert au.density(a) == (a != 0).sum() / float(a.shape[0])


def test_lgamma():
    x = vec(0.5, 1, 2, 3.5, -0.5, -1.5, 100, 1e-5, 0, -1, -2)
    r = au.lgamma(x)
    for v, w in zip(x, r):
        if v in (0, -1, -2):
            assert w == INF
        else:
            # python's math.lgamma is its own implementation, not libm's
            assert math.isclose(w, math.lgamma(v), rel_tol=1e-13, abs_tol=1e-15)


def test_unary_math_is_libm():
    "The C library's functions, as the sparse vectors used (numpy's can differ)"
    rs = np.random.RandomState(1)
    x = rs.randn(1000) * 3
    names = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'sinh': 'sinh',
             'cosh': 'cosh', 'tanh': 'tanh', 'atan': 'atan',
             'asinh': 'asinh', 'exp': 'exp', 'expm1': 'expm1'}
    for name, mname in names.items():
        r = getattr(au, name)(x)
        assert np.array_equal(r, [getattr(math, mname)(v) for v in x]), name
    pos = np.abs(x) + 0.01
    for name in ['log', 'log2', 'log10', 'log1p', 'sqrt']:
        r = getattr(au, name)(pos)
        assert np.array_equal(r, [getattr(math, name)(v) for v in pos]), name


def test_unary_out_of_domain():
    x = vec(2, -2, 0, 1)
    assert np.isnan(au.asin(x)[:2]).all()
    assert np.isnan(au.acos(x)[:2]).all()
    assert au.acos(x)[2] == math.pi / 2
    r = au.log(vec(0, -1))
    assert r[0] == -INF and np.isnan(r[1])
    assert np.isnan(au.sqrt(vec(-1))[0])
    assert au.exp(vec(1000))[0] == INF
    assert np.isnan(au.acosh(vec(0))[0])


def test_sign_sq_fabs_ceil_floor():
    x = vec(-2.5, 0, 3.5, NAN, INF, -INF)
    assert np.array_equal(au.sign(x), vec(-1, 0, 1, 0, 1, -1))
    assert np.array_equal(au.sq(vec(-2, 0, 3)), vec(4, 0, 9))
    assert np.array_equal(au.fabs(vec(-2, 0, 3)), vec(2, 0, 3))
    assert np.array_equal(au.ceil(vec(-1.5, 1.2)), vec(-1, 2))
    assert np.array_equal(au.floor(vec(-1.5, 1.2)), vec(-2, 1))


def test_mul_zero_wins():
    r = au.mul(vec(0, INF, NAN, 2, 0), vec(INF, 0, 0, 3, 5))
    assert np.array_equal(r, vec(0, 0, 0, 6, 0))
    r = au.mul(vec(INF, NAN), vec(2, 3))
    assert r[0] == INF and np.isnan(r[1])


def test_mul_scalar():
    a = vec(0, 1, -2, INF)
    assert np.array_equal(au.mul_scalar(a, 3.0), vec(0, 3, -6, INF))
    r = au.mul_scalar(a, INF)
    assert np.array_equal(r, vec(0, INF, -INF, INF))
    r = au.mul_scalar(a, NAN)
    assert r[0] == 0 and np.isnan(r[1:]).all()


def test_div():
    # b == 0: infinity (any sign) if a is not zero, nan if it is; a == 0
    # and b != 0: zero
    a = vec(1, -1, 0, 0, 6, INF)
    b = vec(0, 0, 0, 2, 3, 2)
    r = au.div(a, b)
    assert r[0] == INF and r[1] == INF
    assert np.isnan(r[2])
    assert r[3] == 0 and r[4] == 2 and r[5] == INF
    assert au.div(vec(0), vec(NAN))[0] == 0


def test_atan2_minus_zero_is_zero():
    r = au.atan2(vec(-0.0, 0, 1, 0, -0.0), vec(-1, -1, 0, 2, -0.0))
    assert np.array_equal(r, vec(math.pi, math.pi, math.pi / 2, 0, 0))
    assert not np.signbit(r).any()


def test_hypot():
    r = au.hypot(vec(3, 0, -4, INF), vec(4, 0, 0, NAN))
    assert np.array_equal(r, vec(5, 0, 4, INF))


def test_dot_skips_zeros_and_adds_in_order():
    a, b = sparse_data(2)
    expected = 0.0
    for x, y in zip(a, b):
        if x != 0 and y != 0:
            expected += x * y
    assert au.dot(a, b) == expected
    # 0 * inf is skipped
    assert au.dot(vec(0, 2), vec(INF, 3)) == 6


def test_sequential_sums_differ_from_pairwise():
    "np.sum / np.dot add in a different order: the last bits differ"
    rs = np.random.RandomState(0)
    x = rs.randn(5000) * 10.0 ** rs.randint(-6, 6, 5000)
    expected = 0.0
    for v in x:
        expected += v
    assert au.sum(x) == expected
    assert au.sum(x) != float(np.sum(x))
    sq = 0.0
    for v in x:
        sq += v * v
    assert au.sq_sum(x) == sq


def test_SSE_SAE():
    a, b = sparse_data(3)
    sse, sae = 0.0, 0.0
    for x, y in zip(a, b):
        sse += (x - y) * (x - y)
        sae += abs(x - y)
    assert au.SSE(a, b) == sse
    assert au.SAE(a, b) == sae
    assert au.SSE(a, a) == 0


def test_reductions():
    a, b = sparse_data(4, 50)
    c, _ = sparse_data(5, 50)
    lst = [a, b, c]
    assert np.array_equal(au.cumsum(lst), (a + b) + c)
    assert np.array_equal(au.cummul(lst), a * b * c)
    assert np.array_equal(au.cummin(lst), np.minimum(np.minimum(a, b), c))
    assert np.array_equal(au.cummax(lst), np.maximum(np.maximum(a, b), c))
    # the inputs are not modified
    assert np.array_equal(lst[0], a)
    # cummul: a zero wins over a non finite value
    r = au.cummul([vec(0, INF), vec(INF, 0), vec(2, 2)])
    assert np.array_equal(r, vec(0, 0))
    # min / max ignore nan, as C's fmin / fmax
    assert np.array_equal(au.cummin([vec(NAN, 1), vec(2, NAN)]), vec(2, 1))
    assert np.array_equal(au.cummax([vec(NAN, 1), vec(2, NAN)]), vec(2, 1))


def test_cumsum_needs_two():
    try:
        au.cumsum([vec(1, 2)])
    except IndexError:
        return
    assert False


def test_argmax_argmin():
    lst = [vec(1, 0, -1, 5), vec(1, 0, -3, 4), vec(0, 2, -1, 5)]
    # ties go to the first one; zeros count as values
    assert np.array_equal(au.argmax(lst), vec(0, 2, 0, 0))
    assert np.array_equal(au.argmin(lst), vec(2, 0, 1, 1))
    assert au.argmax(lst).dtype == np.float64
    assert np.array_equal(au.argmax([vec(3, 4)]), vec(0, 0))
