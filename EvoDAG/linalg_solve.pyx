# Copyright 2016 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
cimport numpy as np
from . import array_utils
from libc cimport math
from libc.float cimport DBL_EPSILON
from cpython cimport array
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM, PyList_SET_ITEM, PyList_Append


cdef bint iszero(double x):
    if x >= DBL_EPSILON or x <= -DBL_EPSILON:
        return False
    return True


cdef double zero_round(double value):
    if iszero(value):
        return 0.0
    return value

        
cdef void swap(list m, Py_ssize_t i):
    cdef double max, comp
    cdef Py_ssize_t pos=i, size=PyList_GET_SIZE(m)
    cdef array.array data, tmp
    # data = m[i]
    data = <array.array> PyList_GET_ITEM(m, i)
    max = math.fabs(data.data.as_doubles[i])
    for j in range(i+1, size):
        # data = m[j]
        data = <array.array> PyList_GET_ITEM(m, j)
        comp = math.fabs(data.data.as_doubles[i])
        if comp > max:
            max = comp
            pos = j
    if pos == i:
        return
    # data = m[i]
    data = <array.array> PyList_GET_ITEM(m, i)
    # m[i] = m[pos]
    tmp = <array.array> PyList_GET_ITEM(m, pos)
    PyList_SET_ITEM(m, i, tmp)
    # m[pos] = data
    PyList_SET_ITEM(m, pos, data)
    

cdef bint gauss_jordan(list m):
    cdef Py_ssize_t i, j, k, size=PyList_GET_SIZE(m)
    cdef array.array data, below
    cdef double c, tmp, c_den
    cdef double *data_values, *below_values
    for i in range(size):
        swap(m, i)
        # data = m[i]
        data = <array.array> PyList_GET_ITEM(m, i)
        data_values = data.data.as_doubles
        c_den = data_values[i]
        if iszero(c_den):
            return True
        for j in range(i+1, size):
            # below = m[j]
            below = <array.array> PyList_GET_ITEM(m, j)
            below_values = below.data.as_doubles
            c = below_values[i] / c_den
            for k in range(i, size+1):
                tmp = below_values[k] - data_values[k] * c
                below_values[k] = tmp
    for i in range(size-1, -1, -1):
        # data = m[i]
        data = <array.array> PyList_GET_ITEM(m, i)
        data_values = data.data.as_doubles
        c = data_values[i]
        if iszero(c):
            return True
        for j in range(0, i):
            # below = m[j]
            below = <array.array> PyList_GET_ITEM(m, j)
            below_values = below.data.as_doubles
            k = size
            tmp = below_values[k] - data_values[k] *\
                  below_values[i] / c
            below_values[k] = zero_round(tmp)
        tmp = data_values[size] / c
        if not math.isfinite(tmp):
            return True
        data_values[size] = zero_round(tmp)
    return False

        
cdef union double_bits:
    double value
    unsigned long long bits


cdef double dot(double[::1] a, double[::1] b):
    """Sum of the products of the positions where both are not zero, adding
    in order (see array_utils.dot)"""
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, y, res = 0
    cdef double_bits p
    for i in range(n):
        x = a[i]
        y = b[i]
        # the product where both are not zero and +0.0 elsewhere, without a
        # branch (it is unpredictable on masked vectors); adding +0.0 is the
        # same as not adding because res is never -0.0
        p.value = x * y
        p.bits &= -(<unsigned long long> ((x != 0) & (y != 0)))
        res += p.value
    return res


cdef bint all_finite(double[::1] a):
    "Whether every entry is finite"
    cdef Py_ssize_t i, n = a.shape[0]
    for i in range(n):
        if not math.isfinite(a[i]):
            return False
    return True


cdef void dot4(double[::1] a, double[::1] b0, double[::1] b1,
               double[::1] b2, double[::1] b3, double *out):
    """``out[k] = dot(a, bk)`` for four vectors at once, when every entry is
    finite. Then a product with a zero is a zero and adding it changes
    nothing (the sum is never -0.0), so the guard of ``dot`` is not needed.
    Each sum adds the same numbers in the same order as ``dot`` (the results
    are identical) but the four chains of additions do not wait for each
    other."""
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double x, r0 = 0, r1 = 0, r2 = 0, r3 = 0
    for i in range(n):
        x = a[i]
        r0 += x * b0[i]
        r1 += x * b1[i]
        r2 += x * b2[i]
        r3 += x * b3[i]
    out[0] = r0
    out[1] = r1
    out[2] = r2
    out[3] = r3


cdef dict support_cache = {}


cdef tuple support(mask):
    """Positions where the mask is not zero and the mask there. Masks made by
    the library are read-only, which makes the result safe to remember: it
    is asked for every node, with the same few masks."""
    cdef bint cacheable = not mask.flags.writeable
    if cacheable:
        hit = support_cache.get(id(mask))
        if hit is not None and hit[0] is mask:
            return hit[1], hit[2]
    index = np.flatnonzero(mask)
    values = np.ascontiguousarray(mask[index])
    if cacheable:
        if len(support_cache) > 256:
            support_cache.clear()
        support_cache[id(mask)] = (mask, index, values)
    return index, values


cdef Py_ssize_t masked(double[::1] a, Py_ssize_t[::1] index, double[::1] mask,
                       double[::1] out):
    """``out = a * mask`` restricted to the positions in ``index`` (those
    where the mask is not zero): a zero in either is a zero, as in
    ``array_utils.mul``. Everything else is zero outside ``index``, which is
    why the products between these vectors only need these positions.
    Returns how many entries are not zero."""
    cdef Py_ssize_t k, n = index.shape[0], nz = 0
    cdef double x, m
    for k in range(n):
        x = a[index[k]]
        m = mask[k]
        if x == 0 or m == 0:
            out[k] = 0
        else:
            out[k] = x * m
            if out[k] != 0:
                nz += 1
    return nz


cpdef compute_weight(list r, double[::1] ytr, mask):
    """Returns the weight (w) using OLS of r * w = gp._ytr """
    cdef Py_ssize_t i, j, size=PyList_GET_SIZE(r), k=0
    cdef double[::1] ri, rj, rc
    cdef array.array data = array.array('d'), other, dependent
    cdef list X = [], R = [], var = []
    cdef double tmp, *dependent_value
    if not isinstance(mask, np.ndarray):
        mask = np.full(ytr.shape[0], float(mask))
    index, mask_support = support(mask)
    dependent = array.clone(data, size, False)
    dependent_value = dependent.data.as_doubles
    for i in range(size):
        ri = <object> PyList_GET_ITEM(r, i)
        tmp = dot(ytr, ri)
        if not math.isfinite(tmp):
            return None
        rc_array = np.empty(len(index))
        rc = rc_array
        if masked(ri, index, mask_support, rc):
            PyList_Append(R, rc_array)
            PyList_Append(var, i)
            dependent_value[k] = tmp
            k += 1
    size = PyList_GET_SIZE(R)
    X = [array.clone(data, size+1, False) for i in range(size)]
    cdef double[::1] r1, r2, r3
    cdef double four[4]
    cdef bint finite_all = True
    for i in range(size):
        ri = <object> PyList_GET_ITEM(R, i)
        if not all_finite(ri):
            finite_all = False
            break
    for i in range(size):
        data = <array.array> PyList_GET_ITEM(X, i)
        ri = <object> PyList_GET_ITEM(R, i)
        data.data.as_doubles[size] = dependent_value[i]
        j = i
        while j < size:
            if finite_all and j + 3 < size:
                # four dot products at a time: the same sums, in the same order
                rj = <object> PyList_GET_ITEM(R, j)
                r1 = <object> PyList_GET_ITEM(R, j + 1)
                r2 = <object> PyList_GET_ITEM(R, j + 2)
                r3 = <object> PyList_GET_ITEM(R, j + 3)
                dot4(ri, rj, r1, r2, r3, four)
                for k in range(4):
                    if not math.isfinite(four[k]):
                        return None
                    data.data.as_doubles[j + k] = four[k]
                    other = <array.array> PyList_GET_ITEM(X, j + k)
                    other.data.as_doubles[i] = four[k]
                j += 4
            else:
                rj = <object> PyList_GET_ITEM(R, j)
                tmp = dot(ri, rj)
                if not math.isfinite(tmp):
                    return None
                data.data.as_doubles[j] = tmp
                other = <array.array> PyList_GET_ITEM(X, j)
                other.data.as_doubles[i] = tmp
                j += 1
    if gauss_jordan(X):
        return None
    res = np.zeros(PyList_GET_SIZE(r))
    for k, i in enumerate(var):
        data = <array.array> PyList_GET_ITEM(X, k)
        res[i] = data[size]
    return res
