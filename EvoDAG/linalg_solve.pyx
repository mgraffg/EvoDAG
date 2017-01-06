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


cdef void swap(list m, Py_ssize_t i):
    cdef double max, comp
    cdef Py_ssize_t pos=i, size=len(m)
    cdef array.array data
    data = m[i]
    max = math.fabs(data.data.as_doubles[i])
    for j in range(i+1, size):
        data = m[j]
        comp = math.fabs(data.data.as_doubles[i])
        if comp > max:
            max = comp
            pos = j
    if pos == i:
        return
    data = m[i]
    m[i] = m[pos]
    m[pos] = data
    

cdef bint gauss_jordan(list m):
    cdef Py_ssize_t i, j, k, size=len(m)
    cdef array.array data, below
    cdef double c, tmp
    for i in range(size):
        swap(m, i)
        data = m[i]
        if math.fabs(data.data.as_doubles[i]) < DBL_EPSILON:
            return True
        for j in range(i+1, size):
            below = m[j]
            c = below[i] / data.data.as_doubles[i]
            for k in range(i, size+1):
                below.data.as_doubles[k] -= data.data.as_doubles[k] * c
    for i in range(size-1, -1, -1):
        data = m[i]
        c = data.data.as_doubles[i]
        for j in range(0, i):
            below = m[j]
            for k in range(size, i-1, -1):
                below.data.as_doubles[k] -= data.data.as_doubles[k] *\
                                            below.data.as_doubles[i] / c
        data.data.as_doubles[i] /= c
        tmp = data.data.as_doubles[size] / c
        if not math.isfinite(tmp):
            return True
        data.data.as_doubles[size] = tmp
    return False

        
cpdef compute_weight(list r, SparseArray ytr, mask):
    """Returns the weight (w) using OLS of r * w = gp._ytr """
    cdef Py_ssize_t i, j, size=len(r)
    cdef SparseArray ri, rj
    cdef array.array data = array.array('d'), other
    cdef list X = [array.clone(data, size+1, False) for i in range(size)]
    cdef double tmp
    for i in range(size):
        data = X[i]
        ri = r[i]
        tmp = ytr.dot(ri)
        if not math.isfinite(tmp):
            return None
        data.data.as_doubles[size] = tmp 
        ri = ri * mask
        for j in range(i, size):
            rj = r[j]
            tmp = ri.dot(rj)
            if not math.isfinite(tmp):
                return None
            data.data.as_doubles[j] = tmp
            other = X[j]
            other.data.as_doubles[i] = tmp
    if gauss_jordan(X):
        return None
    return np.array([x[size] for x in X])
