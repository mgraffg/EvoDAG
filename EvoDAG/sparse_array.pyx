# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#cython: profile=True
#cython: nonecheck=False


from array import array as p_array
# cimport numpy as npc
cimport cython
import numpy as np
import types
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cdef extern from "numpy/npy_math.h":
    bint npy_isinf(double)
    bint npy_isnan(double)
    long double INFINITY "NPY_INFINITY"
    long double PI "NPY_PI"


cpdef rebuild(data, index, size):
    cdef SparseArray r = SparseArray()
    r.init(len(data))
    r.set_size(size)
    r.set_data_index(data, index)
    return r


@cython.freelist(512)
cdef class SparseArray:
    def __cinit__(self):
        self._size = 0
        self._nele = 0
        self._usemem = 0

    cpdef int size(self):
        return self._size

    cpdef set_size(self, int s):
        self._size = s

    cpdef int nele(self):
        return self._nele

    cpdef set_nele(self, int s):
        self._nele = s

    cpdef init_ptr(self, array.array[int] index,
                   array.array[double] data):
        self._indexC = index.data.as_ints
        self._dataC = data.data.as_doubles

    cpdef init(self, int nele):
        if nele == 0:
            self._usemem = 1
            return
        if self._usemem == 0:
            # print "Poniendo memoria", self
            self._usemem = 1
            self._dataC = <double*> PyMem_Malloc(nele * sizeof(double))
            if not self._dataC:
                raise MemoryError("dataC")
            self._indexC = <int *> PyMem_Malloc(nele * sizeof(int))
            if not self._indexC:
                raise MemoryError("indexC")
            self.set_nele(nele)
        else:
            print "Error init", self

    def __dealloc__(self):
        if self._usemem and self.nele() == 0:
            return
        if self._usemem:
            # print "Borrando", self
            self._usemem = 0
            PyMem_Free(self._dataC)
            PyMem_Free(self._indexC)
            # self._dataC = null
            # self._indexC = null
        else:
            print "Error dealloc", self
            
    cpdef int nunion(self, SparseArray other):
        cdef int a=0, b=0, c=0
        cdef int anele = self.nele(), bnele=other.nele()
        if self.size() == anele or self.size() == bnele:
            return self.size()
        while (a < anele) and (b < bnele):
            if self._indexC[a] == other._indexC[b]:
                a += 1
                b += 1
            elif self._indexC[a] < other._indexC[b]:
                a += 1
            else:
                b += 1
            c += 1
        if a == anele:
            return c + bnele - b
        else:
            return c + anele - a

    cpdef int nintersection(self, SparseArray other):
        cdef int a=0, b=0, c=0, *indexC = self._indexC, *oindexC = other._indexC
        cdef int anele = self.nele(), bnele=other.nele()
        if self.size() == anele or self.size() == bnele:
            if anele < bnele:
                return anele
            return bnele
        while (a < anele) and (b < bnele):
            if indexC[a] == oindexC[b]:
                a += 1
                b += 1
                c += 1
            elif indexC[a] < oindexC[b]:
                a += 1
            else:
                b += 1                
        return c

    def __add__(self, other):
        if isinstance(other, SparseArray):
            return self.add(other)
        return self.add2(other)

    cpdef SparseArray add(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] + other._dataC[c]
                res._indexC[c] = c
            return res
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] + other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return res

    cpdef SparseArray add2(self, double other):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i
        for i in range(self.size()):
            res._indexC[i] = i
            res._dataC[i] = other
        for i in range(self.nele()):
            res._dataC[self._indexC[i]] = self._dataC[i] + other
        return res

    def __sub__(self, other):
        if isinstance(other, SparseArray):
            return self.sub(other)
        return self.sub2(other)
        
    cpdef SparseArray sub(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] - other._dataC[c]
                res._indexC[c] = c
            return res        
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = - other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] - other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = - other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return res

    cpdef SparseArray sub2(self, double other):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i
        for i in range(self.size()):
            res._indexC[i] = i
            res._dataC[i] = - other
        for i in range(self.nele()):
            res._dataC[self._indexC[i]] = self._dataC[i] - other
        return res
        
    def __mul__(self, other):
        if isinstance(other, SparseArray):
            return self.mul(other)
        if other == 1:
            return self
        return self.mul2(other)

    cpdef SparseArray mul(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nintersection(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] * other._dataC[c]
                res._indexC[c] = c
            return res        
        while c < rnele:
            if a >= anele:
                b += 1
            elif b >= bnele:
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    res._dataC[c] = self._dataC[a] * other._dataC[b]
                    res._indexC[c] = index
                    a += 1; b += 1; c += 1
                elif index < other._indexC[b]:
                    a += 1
                else:
                    b += 1
        return res

    cpdef SparseArray mul2(self, double other):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i] * other
        return res
                    
    def __div__(self, other):
        if isinstance(other, SparseArray):
            return self.div(other)
        return self.div2(other)

    def __truediv__(self, other):
        if isinstance(other, SparseArray):
            return self.div(other)
        return self.div2(other)

    @cython.cdivision(True)
    cpdef SparseArray div(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nintersection(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                res._dataC[c] = self._dataC[c] / other._dataC[c]
                res._indexC[c] = c
            return res        
        while c < rnele:
            if a >= anele:
                b += 1
            elif b >= bnele:
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    res._dataC[c] = self._dataC[a] / other._dataC[b]
                    res._indexC[c] = index
                    a += 1; b += 1; c += 1
                elif index < other._indexC[b]:
                    a += 1
                else:
                    b += 1
        return res

    @cython.cdivision(True)    
    cpdef SparseArray div2(self, double other):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i] / other
        return res
        
    cpdef double sum(self):
        cdef double res=0, *data = self._dataC
        cdef int i
        for i in xrange(self._nele):
            res += data[i]
        return res

    cpdef double mean(self):
        cdef double res=0
        res = self.sum()    
        return res / self.size()

    @cython.boundscheck(False)
    @cython.nonecheck(False)    
    def var_per_cl(self, list X, list mu,
                   array.array[double] kfreq):
        cdef int i, k=0, ncl = len(kfreq), j, xnele, ynele=self.nele()
        cdef SparseArray x
        cdef list m=[]
        cdef double epsilon = 1e-9
        cdef array.array[double] var, mu_x
        for x, mu_x in zip(X, mu):
            var = array.array('d', [0] * ncl)
            k = 0
            i = 0
            xnele = x.nele()
            while i < xnele:
                if x._indexC[i] == self._indexC[k]:
                    j = int(self._dataC[k])
                    var[j] += math.pow(x._dataC[i] - mu_x[j], 2)
                    k += 1
                    i += 1
                elif self._indexC[k] < x._indexC[i]:
                    if k < ynele:
                        k += 1
                    else:
                        var[0] += math.pow(x._dataC[i] - mu_x[0], 2)
                        i += 1
                else:
                    var[0] += math.pow(x._dataC[i] - mu_x[0], 2)
                    i += 1
            for i in range(ncl):
                var[i] = var[i] / kfreq[i] + epsilon
            m.append(var)
        return m

    @cython.boundscheck(False)
    @cython.nonecheck(False)    
    def mean_per_cl(self, list X, array.array[double] kfreq):
        cdef int i, k=0, ncl = len(kfreq), j, xnele, ynele=self.nele()
        cdef SparseArray x
        cdef list m=[]
        cdef array.array[double] mu
        for x in X:
            mu = array.array('d', [0] * ncl)
            k = 0
            i = 0
            xnele = x.nele()
            while i < xnele:
                if x._indexC[i] == self._indexC[k]:
                    j = int(self._dataC[k])
                    mu[j] += x._dataC[i]
                    k += 1
                    i += 1
                elif self._indexC[k] < x._indexC[i]:
                    if k < ynele:
                        k += 1
                    else:
                        mu[0] += x._dataC[i]
                        i += 1
                else:
                    mu[0] += x._dataC[i]
                    i += 1
            for i in range(ncl):
                mu[i] = mu[i] / kfreq[i]
            m.append(mu)
        return m

    @cython.boundscheck(False)
    @cython.nonecheck(False)        
    def class_freq(self, int ncl):
        cdef int i
        cdef list f
        cdef array.array[double] mu = array.array('d', [0] * ncl)
        for i in range(self.nele()):
            mu[int(self._dataC[i])] += 1.0
        mu[0] = self.size() - sum(mu[1:])
        return mu    
        
    cpdef double std(self):
        cdef SparseArray res = self.sub2(self.mean())
        res = res.sq()
        return math.sqrt(res.sum() / res.size())

    cpdef SparseArray fabs(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in xrange(self.nele()):
            res._dataC[i] = math.fabs(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray exp(self):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i, j=0, nele=self.nele()
        for i in xrange(self.size()):
            if j < nele and self._indexC[j] == i:
                res._dataC[i] = math.exp(self._dataC[j])
                j += 1
            else:
                res._dataC[i] = 1
            res._indexC[i] = i
        return res
        
    cpdef SparseArray sin(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in xrange(self.nele()):
            res._dataC[i] = math.sin(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray cos(self):
        cdef SparseArray res = self.empty(self.size(), self.size())
        cdef int i, j=0, nele=self.nele()
        for i in xrange(self.size()):
            if j < nele and self._indexC[j] == i:
                res._dataC[i] = math.cos(self._dataC[j])
                j += 1
            else:
                res._dataC[i] = 1
            res._indexC[i] = i
        return res
        
    cpdef SparseArray ln(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            res._dataC[i] = math.log(math.fabs(r))
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray sq(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            res._dataC[i] = r * r
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray sqrt(self):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            res._dataC[i] = math.sqrt(self._dataC[i])
            res._indexC[i] = self._indexC[i]
        return res

    @cython.cdivision(True)
    cpdef SparseArray sigmoid(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            res._dataC[i] = 1 / (1 + math.exp((-self._dataC[i] + 1) * 30))
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray if_func(self, SparseArray y, SparseArray z):
        cdef SparseArray s = self.sigmoid()
        cdef SparseArray sy, sz
        cdef SparseArray r
        cdef SparseArray res
        sy = s.mul(y)
        sz = s.mul(z)
        r = sy.sub(sz)
        res = r.add(z)
        return res

    cpdef SparseArray sign(self):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            res._dataC[i] = 0
            if r > 0:
                res._dataC[i] = 1
            elif r < 0:
                res._dataC[i] = -1
            res._indexC[i] = self._indexC[i]
        return res

    cpdef SparseArray min(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                if self._dataC[c] < other._dataC[c]:
                    res._dataC[c] = self._dataC[c]
                else:
                    res._dataC[c] = other._dataC[c]
                res._indexC[c] = c
            return res
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = other._dataC[b]
                if r > 0:
                    r = 0
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                if r > 0:
                    r = 0                
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    if self._dataC[a] < other._dataC[b]:
                        r = self._dataC[a]
                    else:
                        r = other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    if r > 0:
                        r = 0
                    a += 1
                else:
                    r = other._dataC[b]
                    if r > 0:
                        r = 0
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return res

    cpdef SparseArray max(self, SparseArray other):
        cdef SparseArray res = self.empty(self.nunion(other), self.size())
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self.nele(), bnele=other.nele(), rnele=res.nele()
        cdef double r
        if anele == rnele and bnele == rnele and res.size() == rnele:
            for c in range(rnele):
                if self._dataC[c] > other._dataC[c]:
                    res._dataC[c] = self._dataC[c]
                else:
                    res._dataC[c] = other._dataC[c]
                res._indexC[c] = c
            return res
        for c in range(rnele):
            if a >= anele:
                index = other._indexC[b]
                r = other._dataC[b]
                if r < 0:
                    r = 0
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                if r < 0:
                    r = 0                
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    if self._dataC[a] > other._dataC[b]:
                        r = self._dataC[a]
                    else:
                        r = other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    if r < 0:
                        r = 0
                    a += 1
                else:
                    r = other._dataC[b]
                    if r < 0:
                        r = 0
                    index = other._indexC[b]
                    b += 1
            res._dataC[c] = r
            res._indexC[c] = index
        return res

    cpdef SparseArray boundaries(self, float lower=-1, float upper=1):
        cdef SparseArray res = self.empty(self.nele(), self.size())
        cdef int i
        cdef double r
        for i in xrange(self.nele()):
            r = self._dataC[i]
            if r > upper:
                res._dataC[i] = upper
            elif r < lower:
                res._dataC[i] = lower
            else:
                res._dataC[i] = r                
            res._indexC[i] = self._indexC[i]
        return res
    
    cpdef double SAE(self, SparseArray other):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self._nele, bnele=other._nele
        cdef double r, res=0
        cdef SparseArray last
        while True:
            if a >= anele and b >= bnele:
                break
            elif a >= anele:
                index = other._indexC[b]
                r = - other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] - other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = - other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res +=  math.fabs(r)
        if npy_isnan(res):
            return INFINITY
        return res
            
    cpdef double SSE(self, SparseArray other):
        cdef int a=0, b=0, index=0, c=0
        cdef int anele = self._nele, bnele=other._nele
        cdef double r, res=0
        cdef SparseArray last
        while True:
            if a >= anele and b >= bnele:
                break
            elif a >= anele:
                index = other._indexC[b]
                r = - other._dataC[b]
                b += 1
            elif b >= bnele:
                index = self._indexC[a]
                r = self._dataC[a]
                a += 1
            else:
                index = self._indexC[a]
                if index == other._indexC[b]:
                    r = self._dataC[a] - other._dataC[b]
                    a += 1; b += 1
                elif index < other._indexC[b]:
                    r = self._dataC[a]
                    a += 1
                else:
                    r = - other._dataC[b]
                    index = other._indexC[b]
                    b += 1
            res +=  r * r
        if npy_isnan(res):
            return INFINITY
        return res

    cpdef double pearsonr(self, SparseArray other):
        cdef double mx, my, up
        mx = self.sum() / self.size()
        my = other.sum() / other.size()
        up = ((self - mx) * (other - my)).sum()
        mx = math.sqrt((self - mx).sq().sum())
        my = math.sqrt((other - my).sq().sum())
        if mx == 0 or my == 0:
            return 1
        return up / (mx * my)
            
    cpdef bint isfinite(self):
        cdef int i
        cdef double r    
        for i in range(self.nele()):
            r = self._dataC[i]
            if npy_isnan(r) or npy_isinf(r):
                return 0
        return 1

    cdef SparseArray select(self, npc.ndarray[long, ndim=1] index):
        cdef long *indexC = <long *>index.data
        cdef list data = [], index2 = []
        cdef int anele=self.nele(), bnele=index.shape[0]
        cdef int a=0, b=0, c=0, i=0
        cdef SparseArray res
        for i in range(index.shape[0]-1):
            if index[i] > index[i+1]:
                raise NotImplementedError("The index must be in order")
        while (a < anele) and (b < bnele):
            if self._indexC[a] == indexC[b]:
                data.append(self._dataC[a])
                index2.append(c)
                b += 1
                c += 1
                if (b >= bnele) or self._indexC[a] != indexC[b]:
                    a += 1
            elif self._indexC[a] < indexC[b]:
                a += 1
            else:
                b += 1
                c += 1
        res = self.empty(len(data), index.shape[0])
        res.set_data_index(data, index2)        
        return res
                
    def __getitem__(self, value):
        cdef int i, init=-1, cnt=0
        cdef SparseArray res
        if isinstance(value, np.ndarray):
            return self.select(value)
        if not isinstance(value, slice):
            raise NotImplementedError("Not implemented yet %s" %(type(value)))
        start = value.start if value.start is not None else 0
        stop = value.stop if value.stop is not None else self.size()
        if stop > self.size():
            stop = self.size()
        for i in range(self.nele()):
            if self._indexC[i] >= start and init == -1:
                init = i
            if init > -1 and self._indexC[i] < stop:
                cnt += 1
            if self._indexC[i] >= stop:
                break
        res = self.empty(cnt, stop - start)
        for i in range(init, init+cnt):
            res._indexC[i - init] = self._indexC[i] - start
            res._dataC[i - init] = self._dataC[i]
        return res

    def concatenate(self, SparseArray dos):
        cdef SparseArray res = self.empty(self.nele() + dos.nele(),
                                          self.size() + dos.size())
        cdef int i, j=0, size=self.size()
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i]
        for i in range(self.nele(), res.nele()):
            res._indexC[i] = dos._indexC[j] + size
            res._dataC[i] = dos._dataC[j] 
            j += 1
        return res

    def tonparray(self):
        import numpy as np
        cdef npc.ndarray[double, ndim=1] res = np.zeros(self.size())
        cdef double * resC = <double *>res.data
        cdef int i, ele
        for i in range(self.nele()):
            ele = self._indexC[i]
            resC[ele] = self._dataC[i]
        return res

    def get_data(self):
        cdef list d=[]
        cdef int i
        for i in range(self.nele()):
            d.append(self._dataC[i])
        return d

    def get_index(self):
        cdef list d=[]
        cdef int i
        for i in range(self.nele()):
            d.append(self._indexC[i])
        return d    
        
    def tolist(self):
        cdef int i, ele
        lst = [0] * self.size()
        for i in range(self.nele()):
            ele = self._indexC[i]
            lst[ele] = self._dataC[i]
        return lst

    def set_data_index(self, data, index):
        cdef int c=0
        for d, i in zip(data, index):
            self._dataC[c] = d
            self._indexC[c] = i
            c += 1

    def print_data(self):
        cdef int i
        for i in range(self.nele()):
            print self._dataC[i]

    @classmethod
    def init_index_data(cls, list index, list data, int _size):
        self = cls()
        self.init(len(data))
        self.set_data_index(data, index)    
        self.set_size(_size)
        return self
        
    @classmethod
    def fromlist(cls, iter, bint force_finite=False):
        self = cls()
        data = []
        index = []
        k = -1
        for k, v in enumerate(iter):
            if force_finite and (npy_isnan(v) or npy_isinf(v)):
                continue
            if v == 0:
                continue
            data.append(v)
            index.append(k)
        self.init(len(data))
        self.set_data_index(data, index)    
        self.set_size(k + 1)
        return self

    cpdef SparseArray empty(self, int nele, int size=-1):
        cdef SparseArray res = SparseArray()
        res.init(nele)
        if size == -1:
            res.set_size(nele)
        else:
            res.set_size(size)
        return res

    cpdef SparseArray copy(self):
        cdef SparseArray res = self.empty(self.nele(), self._size)
        cdef int i
        for i in range(self.nele()):
            res._indexC[i] = self._indexC[i]
            res._dataC[i] = self._dataC[i]
        return res
            
    cpdef SparseArray constant(self, double v, int size=-1):
        cdef int i
        cdef SparseArray res = SparseArray()
        if size == -1:
            size = self.size()
        res.set_size(size)        
        if v == 0:
            res.init(0)
        else:
            res.init(size)
        for i in range(res.nele()):
            res._dataC[i] = v
            res._indexC[i] = i
        return res

    @cython.boundscheck(False)
    @cython.nonecheck(False)        
    def BER(self, SparseArray yh, array.array[double] class_freq):
        cdef array.array[double] err = array.array('d', [0] * len(class_freq))
        cdef int i=0, j=0, k=0, ynele=self.nele(), yhnele=yh.nele()
        cdef int c1=0, c2=0
        cdef double res=0
        for k in range(self.size()):
            if self._indexC[i] == k:
                c1 = int(self._dataC[i])
                if i < ynele - 1:
                    i += 1
            else:
                c1 = 0
            if j < yhnele and yh._indexC[j] == k:
                c2 = int(yh._dataC[j])
                j += 1
            else:
                c2 = 0
            if c1 != c2:
                err[c1] += 1
        for i in range(len(class_freq)):
            res += err[i] / class_freq[i]
        return res / len(class_freq) * 100.
        
    @staticmethod
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    def distance(list X, list ev, npc.ndarray[double, ndim=2] output):
        cdef SparseArray x
        cdef SparseArray y
        cdef double *data = <double *> output.data
        cdef int c = 0, i=0, j=0, len_X = len(X)
        for i in range(len(ev)):
            x = ev[i]
            for j in range(len_X):
                y = X[j]
                data[c] = x.SAE(y)
                c += 1

    def __reduce__(self):
        cdef list data = []
        cdef list index = []
        cdef int i
        for i in range(self.nele()):
            data.append(self._dataC[i])
            index.append(self._indexC[i])
        return (rebuild, (data, index, self.size()))
