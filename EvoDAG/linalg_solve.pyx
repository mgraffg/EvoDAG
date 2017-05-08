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
from SparseArray.sparse_array cimport SparseArray
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

        
cpdef compute_weight(list r, SparseArray ytr, mask):
    """Returns the weight (w) using OLS of r * w = gp._ytr """
    cdef Py_ssize_t i, j, size=PyList_GET_SIZE(r), k=0
    cdef SparseArray ri, rj
    cdef array.array data = array.array('d'), other, dependent
    cdef list X = [], R = [], var = []
    cdef double tmp, *dependent_value
    dependent = array.clone(data, size, False)
    dependent_value = dependent.data.as_doubles
    for i in range(size):
        ri = <SparseArray> PyList_GET_ITEM(r, i)
        tmp = ytr.dot(ri)
        if not math.isfinite(tmp):
            return None        
        ri = ri * mask
        if ri.non_zero:
            PyList_Append(R, ri)
            PyList_Append(var, i)
            dependent_value[k] = tmp
            k += 1
    size = PyList_GET_SIZE(R)
    X = [array.clone(data, size+1, False) for i in range(size)]
    for i in range(size):
        data = <array.array> PyList_GET_ITEM(X, i)
        ri = <SparseArray> PyList_GET_ITEM(R, i)
        data.data.as_doubles[size] = dependent_value[i]
        for j in range(i, size):
            rj = <SparseArray> PyList_GET_ITEM(R, j)
            tmp = ri.dot(rj)
            if not math.isfinite(tmp):
                return None
            data.data.as_doubles[j] = tmp
            other = <array.array> PyList_GET_ITEM(X, j)
            other.data.as_doubles[i] = tmp
    if gauss_jordan(X):
        return None
    res = np.zeros(PyList_GET_SIZE(r))
    for k, i in enumerate(var):
        data = <array.array> PyList_GET_ITEM(X, k)
        res[i] = data[size]
    return res
