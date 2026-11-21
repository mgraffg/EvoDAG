# cython: wraparound=False
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
"""Operations on vectors, i.e. 1-D C-contiguous float64 numpy arrays.

EvoDAG used to keep its vectors as sparse objects that only stored the
entries that are not zero. The functions here reproduce their
semantics on dense arrays so that the results do not change:

* the C library's math functions are used (not numpy's, whose SIMD versions
  can differ in the last bit), through typed memoryviews;
* where one operand is zero the sparse operations did not compute anything
  (e.g. ``0 * inf`` is 0, ``x / 0`` is infinity for any non-zero ``x`` and
  ``nan`` for ``x == 0``): ``mul``, ``mul_scalar``, ``div`` and ``atan2``
  follow that;
* ``dot``, ``SSE``, ``SAE`` and ``sum`` add sequentially in index order and
  not pairwise, which changes the last bits and, with them, the search
  trajectory.
"""
cimport cython
import numpy as np
from libc cimport math
from libc.math cimport INFINITY, NAN
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM

ctypedef double (*one_argument)(double) noexcept nogil


cdef inline double sq_op(double a) noexcept nogil:
    return a * a


cdef inline double sign_op(double a) noexcept nogil:
    if a > 0:
        return 1
    elif a < 0:
        return -1
    return 0


cdef object apply1(const double[::1] a, one_argument func):
    cdef Py_ssize_t i, n = a.shape[0]
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        out[i] = func(a[i])
    return res


def acos(const double[::1] a):
    """Element-wise arc cosine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.acos)


def asin(const double[::1] a):
    """Element-wise arc sine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.asin)


def atan(const double[::1] a):
    """Element-wise arc tangent of ``a`` using the C library, returns a new array."""
    return apply1(a, math.atan)


def cos(const double[::1] a):
    """Element-wise cosine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.cos)


def sin(const double[::1] a):
    """Element-wise sine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.sin)


def tan(const double[::1] a):
    """Element-wise tangent of ``a`` using the C library, returns a new array."""
    return apply1(a, math.tan)


def cosh(const double[::1] a):
    """Element-wise hyperbolic cosine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.cosh)


def sinh(const double[::1] a):
    """Element-wise hyperbolic sine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.sinh)


def tanh(const double[::1] a):
    """Element-wise hyperbolic tangent of ``a`` using the C library, returns a new array."""
    return apply1(a, math.tanh)


def acosh(const double[::1] a):
    """Element-wise inverse hyperbolic cosine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.acosh)


def asinh(const double[::1] a):
    """Element-wise inverse hyperbolic sine of ``a`` using the C library, returns a new array."""
    return apply1(a, math.asinh)


def atanh(const double[::1] a):
    """Element-wise inverse hyperbolic tangent of ``a`` using the C library, returns a new array."""
    return apply1(a, math.atanh)


def exp(const double[::1] a):
    """Element-wise exponential of ``a`` using the C library, returns a new array."""
    return apply1(a, math.exp)


def expm1(const double[::1] a):
    """Element-wise exponential minus one of ``a`` using the C library, returns a new array."""
    return apply1(a, math.expm1)


def log(const double[::1] a):
    """Element-wise natural logarithm of ``a`` using the C library, returns a new array."""
    return apply1(a, math.log)


def log2(const double[::1] a):
    """Element-wise base 2 logarithm of ``a`` using the C library, returns a new array."""
    return apply1(a, math.log2)


def log10(const double[::1] a):
    """Element-wise base 10 logarithm of ``a`` using the C library, returns a new array."""
    return apply1(a, math.log10)


def log1p(const double[::1] a):
    """Element-wise natural logarithm of one plus the value of ``a`` using the C library, returns a new array."""
    return apply1(a, math.log1p)


def lgamma(const double[::1] a):
    "Log of the absolute value of gamma (the C library's, numpy has none)"
    return apply1(a, math.lgamma)


def sqrt(const double[::1] a):
    """Element-wise square root of ``a`` using the C library, returns a new array."""
    return apply1(a, math.sqrt)


def sq(const double[::1] a):
    """Element-wise square of ``a`` using the C library, returns a new array."""
    return apply1(a, sq_op)


def sign(const double[::1] a):
    "-1, 0 or 1; ``nan`` gives 0"
    return apply1(a, sign_op)


def fabs(const double[::1] a):
    """Element-wise absolute value of ``a`` using the C library, returns a new array."""
    return apply1(a, math.fabs)


def ceil(const double[::1] a):
    """Element-wise ceiling of ``a`` using the C library, returns a new array."""
    return apply1(a, math.ceil)


def floor(const double[::1] a):
    """Element-wise floor of ``a`` using the C library, returns a new array."""
    return apply1(a, math.floor)


def finite(a):
    "Sets to zero, in place, the entries that are not finite; returns ``a``"
    cdef double[::1] v = a
    cdef Py_ssize_t i, n = v.shape[0]
    for i in range(n):
        if not math.isfinite(v[i]):
            v[i] = 0
    return a


def isfinite(const double[::1] a):
    "Whether every entry is finite"
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        if not math.isfinite(a[i]):
            return False
    return True


def density(a):
    "Fraction of entries that are not zero (``nan`` counts as not zero)"
    return np.count_nonzero(a) / float(len(a))


cdef union double_bits:
    double value
    unsigned long long bits


def mul(const double[::1] a, const double[::1] b):
    "``a * b``; it is 0 wherever one of them is, also if the other is inf"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, y
    cdef double_bits p
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        x = a[i]
        y = b[i]
        # the product where both are not zero and +0.0 elsewhere, without a
        # branch (the zeros of a mask are unpredictable)
        p.value = x * y
        p.bits &= -(<unsigned long long> ((x != 0) & (y != 0)))
        out[i] = p.value
    return res


def mul_scalar(const double[::1] a, double w):
    "``a * w``; zeros stay zero, also if ``w`` is not finite"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x
    if math.isfinite(w):
        return np.multiply(a, w)
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        x = a[i]
        if x == 0:
            out[i] = 0
        else:
            out[i] = x * w
    return res


def div(const double[::1] a, const double[::1] b):
    """``a / b`` where ``b`` is zero: infinity if ``a`` is not zero and
    ``nan`` if it is; where ``a`` is zero and ``b`` is not: 0"""
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, y
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        x = a[i]
        y = b[i]
        if y == 0:
            if x == 0:
                out[i] = NAN
            else:
                out[i] = INFINITY
        elif x == 0:
            out[i] = 0
        else:
            out[i] = x / y
    return res


def atan2(const double[::1] a, const double[::1] b):
    """Element-wise arc tangent of ``a / b`` using the quadrant, a zero (also -0.0) in either operand is taken as +0.0."""
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, y
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        x = a[i]
        y = b[i]
        # -0.0 is a zero, i.e. +0.0 (atan2 tells them apart)
        out[i] = math.atan2(0.0 if x == 0 else x, 0.0 if y == 0 else y)
    return res


def hypot(const double[::1] a, const double[::1] b):
    """Element-wise Euclidean norm of ``a`` and ``b``, i.e. the square root of ``a * a + b * b``."""
    cdef Py_ssize_t i, n = a.shape[0]
    res = np.empty(n)
    cdef double[::1] out = res
    for i in range(n):
        out[i] = math.hypot(a[i], b[i])
    return res


def dot(const double[::1] a, const double[::1] b):
    "Sum of the products of the positions where both are not zero, in order"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, y, res = 0
    for i in range(n):
        x = a[i]
        y = b[i]
        if x != 0 and y != 0:
            res += x * y
    return res


def SSE(const double[::1] a, const double[::1] b):
    "Sum of squared errors, adding in order"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double d, res = 0
    for i in range(n):
        d = a[i] - b[i]
        res += d * d
    return res


def SAE(const double[::1] a, const double[::1] b):
    "Sum of absolute errors, adding in order"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double res = 0
    for i in range(n):
        res += math.fabs(a[i] - b[i])
    return res


def sum(const double[::1] a):
    "Sum adding in order"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double res = 0
    for i in range(n):
        res += a[i]
    return res


def sq_sum(const double[::1] a):
    "Sum of the squares, adding in order"
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double res = 0
    for i in range(n):
        res += a[i] * a[i]
    return res


def cumsum(list lst):
    """Sum of the vectors in ``lst`` (at least two) added in order, returns a new array."""
    if len(lst) < 2:
        raise IndexError('cumsum needs at least two vectors')
    a = lst[0] + lst[1]
    for r in lst[2:]:
        a += r
    return a


def cummul(list lst):
    """Product of the vectors in ``lst`` (at least two) with the semantics of ``mul``, returns a new array."""
    if len(lst) < 2:
        raise IndexError('cummul needs at least two vectors')
    a = mul(lst[0], lst[1])
    for r in lst[2:]:
        a = mul(a, r)
    return a


def cummin(list lst):
    """Element-wise minimum of the vectors in ``lst`` (at least two) ignoring ``nan``, returns a new array."""
    if len(lst) < 2:
        raise IndexError('cummin needs at least two vectors')
    a = np.fmin(lst[0], lst[1])
    for r in lst[2:]:
        np.fmin(a, r, out=a)
    return a


def cummax(list lst):
    """Element-wise maximum of the vectors in ``lst`` (at least two) ignoring ``nan``, returns a new array."""
    if len(lst) < 2:
        raise IndexError('cummax needs at least two vectors')
    a = np.fmax(lst[0], lst[1])
    for r in lst[2:]:
        np.fmax(a, r, out=a)
    return a


def argmax(list lst):
    "Position in ``lst`` of the largest value in each position (the first)"
    if len(lst) == 0:
        raise IndexError('argmax of no vectors')
    cdef Py_ssize_t i, j, n
    cdef const double[::1] first = lst[0], comp
    n = first.shape[0]
    best = np.array(first)
    index = np.zeros(n)
    cdef double[::1] best_C = best, index_C = index
    for j in range(1, PyList_GET_SIZE(lst)):
        comp = <object> PyList_GET_ITEM(lst, j)
        for i in range(n):
            if comp[i] > best_C[i]:
                best_C[i] = comp[i]
                index_C[i] = j
    return index


def argmin(list lst):
    "Position in ``lst`` of the smallest value in each position (the first)"
    if len(lst) == 0:
        raise IndexError('argmin of no vectors')
    cdef Py_ssize_t i, j, n
    cdef const double[::1] first = lst[0], comp
    n = first.shape[0]
    best = np.array(first)
    index = np.zeros(n)
    cdef double[::1] best_C = best, index_C = index
    for j in range(1, PyList_GET_SIZE(lst)):
        comp = <object> PyList_GET_ITEM(lst, j)
        for i in range(n):
            if comp[i] < best_C[i]:
                best_C[i] = comp[i]
                index_C[i] = j
    return index
