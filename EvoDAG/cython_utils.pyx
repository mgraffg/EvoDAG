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


cpdef double fitness_SSE(list _ytr, list _hy, list _mask):
    cdef SparseArray ytr, hy, mask
    cdef double res = 0
    for ytr, hy, mask in zip(_ytr, _hy, _mask):
        res += -ytr.SSE(hy.mul(mask))
    return res / len(_ytr)


cpdef double fitness_SAE(list _ytr, list _hy, list _mask):
    cdef SparseArray ytr, hy, mask
    cdef double res = 0
    for ytr, hy, mask in zip(_ytr, _hy, _mask):
        res += -ytr.SAE(hy.mul(mask))
    return res / len(_ytr)
