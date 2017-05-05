# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from SparseArray.sparse_array cimport SparseArray
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM, PyList_New, PyList_Append
from cpython cimport array
from cpython cimport list
from libc cimport math
import random
cimport cython


cpdef double fitness_SSE(list _ytr, list _hy, SparseArray _mask):
    cdef SparseArray ytr, hy, mask
    cdef double res = 0
    cdef Py_ssize_t i
    for i in range(PyList_GET_SIZE(_ytr)):
        ytr = <SparseArray> PyList_GET_ITEM(_ytr, i)
        hy = <SparseArray> PyList_GET_ITEM(_hy, i)
        res += -ytr.SSE(hy.mul(_mask))
    return res / len(_ytr)


cpdef double fitness_SAE(list _ytr, list _hy, list _mask):
    cdef SparseArray ytr, hy, mask
    cdef double res = 0
    for ytr, hy, mask in zip(_ytr, _hy, _mask):
        res += -ytr.SAE(hy.mul(mask))
    return res / len(_ytr)


cpdef bint naive_bayes_isfinite(list coef, int nclass):
    cdef Py_ssize_t i, j=0
    cdef array.array c
    cdef double *c_value
    for j in range(1, 3):
        c = <array.array> PyList_GET_ITEM(coef, j)
        c_value = c.data.as_doubles
        for i in range(nclass):
            if c_value[i] == 0:
                return False
    return True


cpdef bint naive_bayes_isfinite_MN(list coef, int nclass):
    cdef Py_ssize_t i, j=0
    cdef array.array c
    cdef double *c_value
    for j in range(2):
        c = <array.array> PyList_GET_ITEM(coef, j)
        c_value = c.data.as_doubles
        for i in range(nclass):
            if not math.isfinite(c_value[i]):
                return False
    return True


cpdef list naive_bayes_mean_std2(SparseArray var, array.array klass, array.array mask, int nclass):
    cdef array.array mean_num, mean_den, std_num
    cdef unsigned int *a = var.index.data.as_uints, index
    cdef unsigned int *klass_value = klass.data.as_uints
    cdef double *a_value = var.data.data.as_doubles, tmp
    cdef double *mean_num_value, *mean_den_value, *std_num_value
    cdef unsigned int *mask_value = mask.data.as_uints
    mean_num = array.clone(var.data, nclass, zero=True)
    mean_num_value = mean_num.data.as_doubles
    std_num = array.clone(var.data, nclass, zero=True)
    std_num_value = std_num.data.as_doubles
    mean_den = array.clone(var.data, nclass, zero=True)
    mean_den_value = mean_den.data.as_doubles
    cdef Py_ssize_t i, k=0
    cdef unsigned int var_end = var.non_zero
    if var_end == 0:
        return [mean_num, std_num, mean_den]
    for i in range(len(mask)):
        index = mask_value[i]
        for k in range(k, var_end):
            if a[k] >= index:
                break
        if a[k] == index:
            mean_num_value[klass_value[index]] += a_value[k]
            mean_den_value[klass_value[index]] += 1
        else:
            mean_den_value[klass_value[index]] += 1
    for i in range(nclass):
        mean_num_value[i] = mean_num_value[i] / mean_den_value[i]
    k = 0
    for i in range(len(mask)):
        index = mask_value[i]
        for k in range(k, var_end):
            if a[k] >= index:
                break
        if a[k] == index:
            tmp = a_value[k] - mean_num[klass_value[index]]
        else:
            tmp = mean_num[klass_value[index]]
        tmp *= tmp
        std_num_value[klass_value[index]] += tmp
    for i in range(nclass):
        std_num_value[i] = std_num_value[i] / mean_den_value[i]
    tmp = len(mask)
    for i in range(nclass):
        mean_den_value[i] = mean_den_value[i] / tmp
    return [mean_num, std_num, mean_den]


@cython.cdivision(True)
cpdef list naive_bayes(list var, list coef, unsigned int nclass):
    cdef Py_ssize_t i, j
    cdef list l = <list> PyList_GET_ITEM(coef, 0)
    cdef array.array p_klass = <array.array> PyList_GET_ITEM(l, 2), s_klass, m_klass
    cdef double *p_klass_value = p_klass.data.as_doubles, a, *s_klass_value, *m_klass_value
    cdef double c
    cdef SparseArray b = <SparseArray> PyList_GET_ITEM(var, 0), v, tmp
    cdef unsigned int _len = len(b), nvar = len(var)
    cdef list res = []
    for i in range(nclass):
        a = math.log(p_klass_value[i])
        b = b.mul2(0)
        c = 0
        for j in range(nvar):
            v = <SparseArray> PyList_GET_ITEM(var, j)
            l = <list> PyList_GET_ITEM(coef, j)
            m_klass = <array.array> PyList_GET_ITEM(l, 0)
            m_klass_value = m_klass.data.as_doubles
            s_klass = <array.array> PyList_GET_ITEM(l, 1)
            s_klass_value = s_klass.data.as_doubles
            c += math.log(2 * math.pi * s_klass_value[i])
            b = b.add(((v.add2(-m_klass_value[i])).sq()).mul2(1 / s_klass_value[i]))
        b = b.mul2(-0.5)
        b = b.add2(a - (0.5 * c))
        PyList_Append(res, b)
    return res


@cython.cdivision(True)
cpdef list naive_bayes_Nc(SparseArray var, array.array klass, array.array mask, int nclass):
    cdef array.array mean_num, mean_den
    cdef unsigned int *a = var.index.data.as_uints, index
    cdef unsigned int *klass_value = klass.data.as_uints
    cdef double *a_value = var.data.data.as_doubles, tmp
    cdef double *mean_num_value, *mean_den_value, N=0
    cdef unsigned int *mask_value = mask.data.as_uints
    mean_num = array.clone(var.data, nclass, zero=True)
    mean_num_value = mean_num.data.as_doubles
    mean_den = array.clone(var.data, nclass, zero=True)
    mean_den_value = mean_den.data.as_doubles
    cdef Py_ssize_t i, k=0
    cdef unsigned int var_end = var.non_zero
    if var_end == 0:
        return [mean_num, mean_den]
    for i in range(len(mask)):
        index = mask_value[i]
        for k in range(k, var_end):
            if a[k] >= index:
                break
        if a[k] == index and a_value[k] > 0:
            mean_num_value[klass_value[index]] += a_value[k]
            N += a_value[k]
            mean_den_value[klass_value[index]] += 1
        else:
            mean_den_value[klass_value[index]] += 1
    for i in range(nclass):
        mean_num_value[i] = math.log(mean_num_value[i] / N)
    tmp = len(mask)
    for i in range(nclass):
        mean_den_value[i] = math.log(mean_den_value[i] / tmp)
    return [mean_num, mean_den]


@cython.cdivision(True)
cpdef list naive_bayes_MN(list var, list coef, unsigned int nclass):
    cdef Py_ssize_t i, j, k
    cdef list l = <list> PyList_GET_ITEM(coef, 0)
    cdef array.array p_klass = <array.array> PyList_GET_ITEM(l, 1), m_klass
    cdef double *p_klass_value = p_klass.data.as_doubles, *m_klass_value
    cdef SparseArray b = <SparseArray> PyList_GET_ITEM(var, 0), v
    cdef unsigned int nvar = len(var)
    cdef list res = []
    for i in range(nclass):
        b = b.mul2(0)
        for j in range(nvar):
            v = <SparseArray> PyList_GET_ITEM(var, j)
            l = <list> PyList_GET_ITEM(coef, j)
            m_klass = <array.array> PyList_GET_ITEM(l, 0)
            m_klass_value = m_klass.data.as_doubles
            b = b.add(v.mul2(m_klass_value[i]))
        # b = b.add2(p_klass_value[i])
        b = b.add(SparseArray.constant(p_klass_value[i], b.index, b._len))
        m_klass_value = b.data.data.as_doubles
        for k in range(b.non_zero):
            m_klass_value[k] = math.exp(m_klass_value[k])
        PyList_Append(res, b)
    return res


cdef class SelectNumbers:
    cdef array.array data
    cdef public Py_ssize_t pos
    cdef public Py_ssize_t size
    def __cinit__(self, list lst):
        random.shuffle(lst)
        self.data = array.array('I', lst)
        self.size = PyList_GET_SIZE(lst)
        self.pos = 0

    cpdef list get(self, Py_ssize_t k):
        cdef Py_ssize_t end = self.pos + k, i, pos = self.pos
        cdef list res = PyList_New(0)
        cdef array.array data = self.data
        if end > self.size:
            end = self.size
        self.pos = end
        for i in range(pos, end):
            PyList_Append(res, data[i])
        return res

    cpdef int get_one(self):
        cdef pos = self.pos
        self.pos += 1
        return self.data[pos]
    
    cpdef bint empty(self):
        return self.pos == self.size
