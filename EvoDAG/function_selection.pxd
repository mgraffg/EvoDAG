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


from cpython cimport array
from cpython cimport set
cimport cython


cdef class FunctionSelection:
    cdef public array.array fitness
    cdef public array.array times
    cdef public array.array nargs
    cdef public array.array density_safe
    cdef public unsigned int nfunctions
    cdef public unsigned int density_safe_size
    cdef public unsigned int tournament_size
    cdef public float density
    cdef public float min_density
    cdef public set unfeasible_functions
    cpdef int random_function(self) except -1
    cpdef double avg_fitness(self, Py_ssize_t k)
    cpdef int tournament(self) except -1
    cpdef bint comparison(self, int best, int comp)
