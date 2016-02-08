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
#cython: nonecheck=True

from cpython cimport array
cimport numpy as npc
cimport cython
cimport libc.math as math


cdef class SparseArray:
    cdef int _nele
    cdef int _size
    cdef int * _indexC
    cdef double * _dataC
    cdef bint _usemem
    cpdef int size(self)
    cpdef set_size(self, int s)
    cpdef int nele(self)
    cpdef set_nele(self, int s)
    cpdef init_ptr(self, array.array[int] index, array.array[double] data)
    cpdef int nunion(self, SparseArray b)
    cpdef int nintersection(self, SparseArray other)
    cpdef SparseArray add(self, SparseArray other)
    cpdef SparseArray add2(self, double other)    
    cpdef SparseArray sub(self, SparseArray other)
    cpdef SparseArray sub2(self, double other)
    cpdef SparseArray mul(self, SparseArray other)
    cpdef SparseArray mul2(self, double other)
    cpdef SparseArray div(self, SparseArray other)
    cpdef SparseArray div2(self, double other)    
    cpdef double sum(self)
    cpdef double mean(self)
    cpdef double std(self)
    cpdef SparseArray fabs(self)
    cpdef SparseArray exp(self)
    cpdef SparseArray sqrt(self)
    cpdef SparseArray sin(self)
    cpdef SparseArray cos(self)
    cpdef SparseArray ln(self)
    cpdef SparseArray sq(self)
    cpdef SparseArray sigmoid(self)
    cpdef SparseArray if_func(self, SparseArray y, SparseArray z)
    cpdef SparseArray sign(self)
    cpdef SparseArray min(self, SparseArray other)
    cpdef SparseArray max(self, SparseArray other)
    cpdef SparseArray boundaries(self, float lower=?, float upper=?)
    cpdef double SAE(self, SparseArray other)
    cpdef double SSE(self, SparseArray other)
    cpdef double pearsonr(self, SparseArray other)
    cpdef bint isfinite(self)
    cdef SparseArray select(self, npc.ndarray[long, ndim=1] index)
    cpdef init(self, int nele)
    cpdef SparseArray empty(self, int nele, int size=?)
    cpdef SparseArray copy(self)    
    cpdef SparseArray constant(self, double v, int size=?)
