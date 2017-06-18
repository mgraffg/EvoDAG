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


cdef class Score:
    cdef public array.array precision
    cdef public array.array precision_den
    cdef public array.array recall
    cdef public array.array recall_den
    cdef public array.array precision2
    cdef public array.array precision2_den
    cdef public array.array recall2
    cdef public array.array recall2_den
    cdef public double accuracy
    cdef public double accuracy2
    cdef unsigned int y_pos
    cdef unsigned int hy_pos
    cdef unsigned int y_size
    cdef unsigned int hy_size
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
        self.y_pos = 0
        self.hy_pos = 0
        self.accuracy = 0
        self.accuracy2 = 0

    cdef double get_klass(self, double *data, unsigned int *index, unsigned int i):
        cdef double res = 0
        if self.y_pos >= self.y_size:
            return res
        if index[self.y_pos] == i:
            res = data[self.y_pos]
            self.y_pos += 1
        return res

    cdef double get_predicted(self, double *data, unsigned int *index, unsigned int i):
        cdef double res = 0
        if self.hy_pos >= self.hy_size:
            return res
        if index[self.hy_pos] == i:
            res = data[self.hy_pos]
            self.hy_pos += 1
        return res

    def RecallDotPrecision(self, Py_ssize_t i, SparseArray y,
                           SparseArray hy, array.array index):
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
    def F1(self, Py_ssize_t i, SparseArray y, SparseArray hy, array.array index):
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
    def macroRecallF1(self, SparseArray y, SparseArray hy, array.array index):
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
    def macroRecall(self, SparseArray y, SparseArray hy, array.array index):
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
    def macroPrecision(self, SparseArray y, SparseArray hy, array.array index):
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

    def DotPrecision(self, SparseArray y, SparseArray hy, array.array index):
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

    def DotRecall(self, SparseArray y, SparseArray hy, array.array index):
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
            self.accuracy += recall[i]
            self.accuracy2 += recall2[i]
            den += recall_den[i]
            den2 += recall2_den[i]
        self.accuracy = self.accuracy / den
        self.accuracy2 = self.accuracy2 / den2

    def errorRate(self, SparseArray y, SparseArray hy, array.array index):
        self.count(y, hy, index)
        self.recall2accuracy()
        return 1 - self.accuracy, 1 - self.accuracy2
        
    @cython.cdivision(True)    
    def accDotMacroF1(self, SparseArray y, SparseArray hy, array.array index):
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
        return f1 * self.accuracy, f12 * self.accuracy2

    def DotRecallDotPrecision(self, SparseArray y, SparseArray hy, array.array index):
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
    
    def DotF1(self, SparseArray y, SparseArray hy, array.array index):
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
    def macroF1(self, SparseArray y, SparseArray hy, array.array index):
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
    cdef count(self, SparseArray y, SparseArray hy, array.array index):
        self.clean()
        cdef unsigned int *index_value = index.data.as_uints
        cdef bint flag
        cdef unsigned int i = 0, size = y._len
        cdef unsigned int k = 0, end = len(index)
        cdef double *y_data = y.data.data.as_doubles
        cdef unsigned int *y_index = y.index.data.as_uints
        cdef double *hy_data = hy.data.data.as_doubles
        cdef unsigned int *hy_index = hy.index.data.as_uints
        cdef double *precision = self.precision.data.as_doubles
        cdef double *recall = self.recall.data.as_doubles
        cdef double *precision2 = self.precision2.data.as_doubles
        cdef double *recall2 = self.recall2.data.as_doubles
        cdef unsigned int *precision_den = self.precision_den.data.as_uints
        cdef unsigned int *recall_den = self.recall_den.data.as_uints
        cdef unsigned int *precision2_den = self.precision2_den.data.as_uints
        cdef unsigned int *recall2_den = self.recall2_den.data.as_uints
        cdef int _y, _hy
        self.y_size = y.non_zero
        self.hy_size = hy.non_zero
        for i in range(size):
            _y = <int> self.get_klass(y_data, y_index, i)
            _hy = <int> self.get_predicted(hy_data, hy_index, i)
            flag = False
            if k < end:
                if index_value[k] == i:
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
