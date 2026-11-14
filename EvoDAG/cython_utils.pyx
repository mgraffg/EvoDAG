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


import numpy as np
from . import array_utils
from cpython.list cimport PyList_GET_SIZE, PyList_GET_ITEM, PyList_New, PyList_Append, PyList_SetItem
from cpython cimport array
from cpython cimport list
from libc cimport math
import random
cimport cython


cdef bint all_zero(double[::1] var):
    "Whether every entry is zero (``nan`` is not zero)"
    cdef Py_ssize_t i
    for i in range(var.shape[0]):
        if var[i] != 0:
            return False
    return True


cdef object times_zero(double[::1] a):
    """``a * 0``: zeros are not stored so they stay zero, but a value that is
    not finite gives ``nan``. The naive bayes kernels start every class from
    ``previous * 0`` and not from a fresh array, so it matters when the
    previous class has non finite values."""
    cdef Py_ssize_t i, n = a.shape[0]
    res = np.zeros(n)
    cdef double[::1] out = res
    for i in range(n):
        if a[i] != 0 and not math.isfinite(a[i]):
            out[i] = math.NAN
    return res


cpdef double fitness_SSE(list _ytr, list _hy, _mask):
    cdef double[::1] ytr, hy
    cdef double res = 0
    cdef Py_ssize_t i
    cdef double x, m, d
    for i in range(PyList_GET_SIZE(_ytr)):
        ytr = <object> PyList_GET_ITEM(_ytr, i)
        hy = <object> PyList_GET_ITEM(_hy, i)
        res += -array_utils.SSE(ytr, array_utils.mul(hy, _mask))
    return res / len(_ytr)


cpdef double fitness_SAE(list _ytr, list _hy, list _mask):
    cdef double res = 0
    for ytr, hy, mask in zip(_ytr, _hy, _mask):
        res += -array_utils.SAE(ytr, array_utils.mul(hy, mask))
    return res / len(_ytr)


cpdef bint naive_bayes_isfinite(list coef, int nclass):
    cdef Py_ssize_t i, j=0
    cdef double[::1] c
    for j in range(1, 3):
        c = <object> PyList_GET_ITEM(coef, j)
        for i in range(nclass):
            if c[i] == 0:
                return False
    return True


cpdef bint naive_bayes_isfinite_MN(list coef, int nclass):
    cdef Py_ssize_t i, j=0
    cdef double[::1] c
    for j in range(2):
        c = <object> PyList_GET_ITEM(coef, j)
        for i in range(nclass):
            if not math.isfinite(c[i]):
                return False
    return True


cpdef list naive_bayes_mean_std2(double[::1] var, unsigned int[::1] klass, unsigned int[::1] mask, int nclass):
    cdef unsigned int index
    cdef double tmp
    mean_num = np.zeros(nclass)
    std_num = np.zeros(nclass)
    mean_den = np.zeros(nclass)
    cdef double[::1] mean_num_value = mean_num, std_num_value = std_num, mean_den_value = mean_den
    cdef Py_ssize_t i, size = mask.shape[0]
    if all_zero(var):
        return [mean_num, std_num, mean_den]
    for i in range(size):
        index = mask[i]
        mean_num_value[klass[index]] += var[index]
        mean_den_value[klass[index]] += 1
    for i in range(nclass):
        mean_num_value[i] = mean_num_value[i] / mean_den_value[i]
    for i in range(size):
        index = mask[i]
        tmp = var[index] - mean_num_value[klass[index]]
        tmp *= tmp
        std_num_value[klass[index]] += tmp
    for i in range(nclass):
        std_num_value[i] = std_num_value[i] / mean_den_value[i]
    tmp = size
    for i in range(nclass):
        mean_den_value[i] = mean_den_value[i] / tmp
    return [mean_num, std_num, mean_den]


@cython.cdivision(True)
cpdef list naive_bayes(list var, list coef, unsigned int nclass):
    cdef Py_ssize_t i, j, k
    cdef list l = <list> PyList_GET_ITEM(coef, 0)
    cdef double[::1] p_klass = <object> PyList_GET_ITEM(l, 2), s_klass, m_klass
    cdef double a, c, m, inv, t, v, s
    cdef double[::1] b, vj
    cdef double[::1] first = <object> PyList_GET_ITEM(var, 0)
    cdef Py_ssize_t size = first.shape[0]
    cdef unsigned int nvar = len(var)
    cdef list res = []
    for i in range(nclass):
        a = math.log(p_klass[i])
        res_i = times_zero(first)
        b = res_i
        c = 0
        for j in range(nvar):
            vj = <object> PyList_GET_ITEM(var, j)
            l = <list> PyList_GET_ITEM(coef, j)
            m_klass = <object> PyList_GET_ITEM(l, 0)
            s_klass = <object> PyList_GET_ITEM(l, 1)
            m = m_klass[i]
            s = s_klass[i]
            c += math.log(2 * math.pi * s)
            inv = 1 / s
            for k in range(size):
                t = vj[k] - m
                t = t * t
                # a zero is not stored, so it is not multiplied
                if t != 0:
                    t = t * inv
                b[k] = b[k] + t
        c = a - (0.5 * c)
        for k in range(size):
            if b[k] != 0:
                b[k] = b[k] * -0.5
            b[k] = b[k] + c
        PyList_Append(res, res_i)
        first = res_i
    return res


@cython.cdivision(True)
cpdef list naive_bayes_Nc(double[::1] var, unsigned int[::1] klass, unsigned int[::1] mask, int nclass):
    cdef unsigned int index
    cdef double tmp, N=0
    mean_num = np.zeros(nclass)
    mean_den = np.zeros(nclass)
    cdef double[::1] mean_num_value = mean_num, mean_den_value = mean_den
    cdef Py_ssize_t i, size = mask.shape[0]
    if all_zero(var):
        return [mean_num, mean_den]
    for i in range(size):
        index = mask[i]
        if var[index] > 0:
            mean_num_value[klass[index]] += var[index]
            N += var[index]
        mean_den_value[klass[index]] += 1
    for i in range(nclass):
        mean_num_value[i] = math.log(mean_num_value[i] / N)
    tmp = size
    for i in range(nclass):
        mean_den_value[i] = math.log(mean_den_value[i] / tmp)
    return [mean_num, mean_den]


@cython.cdivision(True)
cdef list _naive_bayes_MN(list var, list coef, unsigned int nclass, list counts):
    cdef Py_ssize_t i, j, k, stored
    cdef list l = <list> PyList_GET_ITEM(coef, 0)
    cdef double[::1] p_klass = <object> PyList_GET_ITEM(l, 1), m_klass
    cdef double m, x, p
    cdef double[::1] b, vj
    cdef double[::1] first = <object> PyList_GET_ITEM(var, 0)
    cdef Py_ssize_t size = first.shape[0]
    cdef unsigned int nvar = len(var)
    cdef list res = []
    for i in range(nclass):
        res_i = times_zero(first)
        b = res_i
        p = p_klass[i]
        for j in range(nvar):
            vj = <object> PyList_GET_ITEM(var, j)
            l = <list> PyList_GET_ITEM(coef, j)
            m_klass = <object> PyList_GET_ITEM(l, 0)
            m = m_klass[i]
            for k in range(size):
                # a zero is not stored, so it is not multiplied
                if vj[k] != 0:
                    b[k] = b[k] + vj[k] * m
        stored = 0
        for k in range(size):
            # the constant is only added where there is a stored value, and
            # only the stored values go through exp
            if b[k] != 0:
                x = b[k] + p
                if x != 0:
                    stored += 1
                    x = math.exp(x)
                b[k] = x
        PyList_Append(res, res_i)
        if counts is not None:
            PyList_Append(counts, stored)
        first = res_i
    return res


cpdef list naive_bayes_MN(list var, list coef, unsigned int nclass):
    return _naive_bayes_MN(var, coef, nclass, None)


cpdef tuple _naive_bayes_MN_stored(list var, list coef, unsigned int nclass):
    """Like ``naive_bayes_MN`` and also the number of entries each output
    stores. The sparse vectors EvoDAG used before stored the entries that went
    through ``exp`` even if it underflowed to zero, which counted in their
    density; the population's density (it drives the choice of functions)
    still needs that count."""
    cdef list counts = []
    cdef list res = _naive_bayes_MN(var, coef, nclass, counts)
    return res, counts


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


cdef class Score:
    cdef public array.array precision
    cdef public array.array precision_den
    cdef public array.array recall
    cdef public array.array recall_den
    cdef public array.array precision2
    cdef public array.array precision2_den
    cdef public array.array recall2
    cdef public array.array recall2_den
    cdef public double _accuracy
    cdef public double _accuracy2
    cdef unsigned int nclasses

    def __cinit__(self, unsigned int nclasses):
        cdef  array.array mask = array.array('d', [])
        self.precision = array.clone(mask, nclasses, zero=False)
        self.recall = array.clone(mask, nclasses, zero=False)
        self.precision2 = array.clone(mask, nclasses, zero=False)
        self.recall2 = array.clone(mask, nclasses, zero=False)
        mask = array.array('I', [])
        self.precision_den = array.clone(mask, nclasses, zero=False)
        self.recall_den = array.clone(mask, nclasses, zero=False)
        self.precision2_den = array.clone(mask, nclasses, zero=False)
        self.recall2_den = array.clone(mask, nclasses, zero=False)
        self.nclasses = nclasses

    cdef _cleanI(self, array.array d):
        cdef Py_ssize_t i = 0
        cdef unsigned int *a = d.data.as_uints
        for i in range(self.nclasses):
            a[i] = 0

    cdef _cleand(self, array.array d):
        cdef Py_ssize_t i = 0
        cdef double *a = d.data.as_doubles
        for i in range(self.nclasses):
            a[i] = 0
            
    cdef clean(self):
        self._cleand(self.precision)
        self._cleanI(self.precision_den)
        self._cleand(self.recall)
        self._cleanI(self.recall_den)
        self._cleand(self.precision2)
        self._cleanI(self.precision2_den)
        self._cleand(self.recall2)
        self._cleanI(self.recall2_den)
        self._accuracy = 0
        self._accuracy2 = 0

    def g_recall_precision(self, Py_ssize_t i, double[::1] y,
                           double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0
        f1 = precision[i] * recall[i]
        f12 = precision2[i] * recall2[i]
        return f1, f12
    
    @cython.cdivision(True)    
    def F1(self, Py_ssize_t i, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        den = precision[i] + recall[i]
        if den > 0:
            f1 = (2 * precision[i] * recall[i]) / den
        den = precision2[i] + recall2[i]
        if den > 0:
            f12 = (2 * precision2[i] * recall2[i]) / den
        return f1, f12

    @cython.cdivision(True)    
    def macroRecallF1(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 += recall[i]
            den = precision2[i] + recall2[i]
            if den > 0:
                f12 += (2 * precision2[i] * recall2[i]) / den
        f1 = f1 / self.nclasses
        f12 = f12 / self.nclasses
        return f1, f12

    @cython.cdivision(True)    
    def a_recall(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *recall = self.recall.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 += recall[i]
            f12 += recall2[i]
        f1 = f1 / self.nclasses
        f12 = f12 / self.nclasses
        return f1, f12

    @cython.cdivision(True)    
    def a_precision(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *recall = self.precision.data.as_doubles
        cdef double *recall2 = self.precision2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 += recall[i]
            f12 += recall2[i]
        f1 = f1 / self.nclasses
        f12 = f12 / self.nclasses
        return f1, f12

    def g_precision(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *recall = self.precision.data.as_doubles
        cdef double *recall2 = self.precision2.data.as_doubles
        cdef double f1 = 1, f12 = 1
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 *= recall[i]
            f12 *= recall2[i]
        return f1, f12    

    def g_recall(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *recall = self.recall.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 1, f12 = 1
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 *= recall[i]
            f12 *= recall2[i]
        return f1, f12
    
    @cython.cdivision(True)
    cdef recall2accuracy(self):
        cdef double *recall = self.recall.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef unsigned int *recall_den = self.recall_den.data.as_uints
        cdef unsigned int *recall2_den = self.recall2_den.data.as_uints
        cdef unsigned int den = 0, den2 = 0
        cdef Py_ssize_t i
        for i in range(self.nclasses):
            self._accuracy += recall[i]
            self._accuracy2 += recall2[i]
            den += recall_den[i]
            den2 += recall2_den[i]
        self._accuracy = self._accuracy / den
        self._accuracy2 = self._accuracy2 / den2

    def accuracy(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.recall2accuracy()
        return self._accuracy, self._accuracy2
        
    @cython.cdivision(True)    
    def accDotMacroF1(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        cdef Py_ssize_t i = 0
        self.count(y, hy, index)
        self.recall2accuracy()
        self.precision_recall()
        for i in range(self.nclasses):
            den = precision[i] + recall[i]
            if den > 0:
                f1 += (2 * precision[i] * recall[i]) / den
            den = precision2[i] + recall2[i]
            if den > 0:
                f12 += (2 * precision2[i] * recall2[i]) / den
        f1 = f1 / self.nclasses
        f12 = f12 / self.nclasses
        return f1 * self._accuracy, f12 * self._accuracy2

    def g_g_recall_precision(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 1, f12 = 1
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            f1 *= precision[i] * recall[i]
            f12 *= precision2[i] * recall2[i]
        return f1, f12
    
    def g_F1(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 1, f12 = 1, den
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            den = precision[i] + recall[i]
            if den > 0:
                f1 *= (2 * precision[i] * recall[i]) / den
            else:
                f1 = 0
            den = precision2[i] + recall2[i]
            if den > 0:
                f12 *= (2 * precision2[i] * recall2[i]) / den
            else:
                f12 = 0
        return f1, f12
    
    @cython.cdivision(True)    
    def a_F1(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.count(y, hy, index)
        self.precision_recall()
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef double f1 = 0, f12 = 0, den
        cdef Py_ssize_t i = 0
        for i in range(self.nclasses):
            den = precision[i] + recall[i]
            if den > 0:
                f1 += (2 * precision[i] * recall[i]) / den
            den = precision2[i] + recall2[i]
            if den > 0:
                f12 += (2 * precision2[i] * recall2[i]) / den
        f1 = f1 / self.nclasses
        f12 = f12 / self.nclasses
        return f1, f12

    @cython.cdivision(True)
    cdef precision_recall(self):
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef unsigned int *precision_den = self.precision_den.data.as_uints
        cdef unsigned int *recall_den = self.recall_den.data.as_uints
        cdef unsigned int *precision2_den = self.precision2_den.data.as_uints
        cdef unsigned int *recall2_den = self.recall2_den.data.as_uints
        cdef Py_ssize_t i
        for i in range(self.nclasses):
            if precision_den[i] > 0:
                precision[i] = precision[i] / precision_den[i]
            else:
                precision[i] = 0
            if recall_den[i] > 0:
                recall[i] = recall[i] / recall_den[i]
            else:
                recall[i] = 0
            if precision2_den[i] > 0:
                precision2[i] = precision2[i] / precision2_den[i]
            else:
                precision2[i] = 0
            if recall2_den[i] > 0:
                recall2[i] = recall2[i] / recall2_den[i]
            else:
                recall2[i] = 0
        
    @cython.cdivision(True)
    cdef count(self, double[::1] y, double[::1] hy, unsigned int[::1] index):
        self.clean()
        cdef bint flag
        cdef unsigned int i = 0, size = y.shape[0]
        cdef unsigned int k = 0, end = index.shape[0]
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef unsigned int *precision_den = self.precision_den.data.as_uints
        cdef unsigned int *recall_den = self.recall_den.data.as_uints
        cdef unsigned int *precision2_den = self.precision2_den.data.as_uints
        cdef unsigned int *recall2_den = self.recall2_den.data.as_uints
        cdef int _y, _hy
        for i in range(size):
            _y = <int> y[i]
            _hy = <int> hy[i]
            flag = False
            if k < end:
                if index[k] == i:
                    flag = True
                    k += 1
            if flag:
                precision_den[_hy] += 1
                recall_den[_y] += 1
                if _y == _hy:
                    precision[_y] += 1
                    recall[_y] += 1
            else:
                precision2_den[_hy] += 1
                recall2_den[_y] += 1
                if _y == _hy:
                    precision2[_y] += 1
                    recall2[_y] += 1
@cython.cdivision(True)
cpdef list centroid_coef(list vars, unsigned int[::1] klass, unsigned int[::1] mask, int nclass):
    cdef unsigned int index
    cdef Py_ssize_t i, j, k, nvars = PyList_GET_SIZE(vars), nmask = mask.shape[0]
    cdef list res = []
    cdef double[::1] var, coef
    mean_num = np.zeros(nclass)
    mean_den = np.zeros(nclass)
    cdef double[::1] mean_num_value = mean_num, mean_den_value = mean_den
    for k in range(nclass):
        PyList_Append(res, np.zeros(nvars))
    for i in range(nvars):
        for k in range(nclass):
            mean_num_value[k] = 0
            mean_den_value[k] = 0
        var = <object> PyList_GET_ITEM(vars, i)
        if all_zero(var):
            continue
        for j in range(nmask):
            index = mask[j]
            mean_num_value[klass[index]] += var[index]
            mean_den_value[klass[index]] += 1
        for k in range(nclass):
            coef = <object> PyList_GET_ITEM(res, k)
            coef[i] = - mean_num_value[k] / mean_den_value[k]
    return res
