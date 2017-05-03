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
import random
from cpython cimport array
from libc.math cimport INFINITY
from cpython.set cimport PySet_Contains, PySet_New, PySet_GET_SIZE


cdef class FunctionSelection:
    def __cinit__(self, nfunctions=0, seed=0, tournament_size=2,
                  nargs=None, density_safe=None):
        cdef unsigned int k
        self.fitness = array.clone(array.array('d'), nfunctions, zero=True)
        self.times = array.clone(array.array('I'), nfunctions, zero=True)
        self.nargs = array.clone(array.array('I'), nfunctions, zero=True)
        self.unfeasible_functions = PySet_New([])
        if nargs is not None:
            for i, k in enumerate(nargs):
                self.nargs[i] = k
        if density_safe is not None:
            self.density_safe_size = len(density_safe)
            self.density_safe = array.clone(array.array('I'), self.density_safe_size, zero=False)
            for k, d in enumerate(density_safe):
                self.density_safe[k] = d
        self.nfunctions = nfunctions
        self.tournament_size = tournament_size
        self.min_density = 0.0
        self.density = 1.0
        random.seed(seed)

    def __setitem__(self, k, v):
        self.fitness[k] += v
        self.times[k] += 1

    cpdef int random_function(self) except -1:
        cdef int value, r
        cdef Py_ssize_t _
        if self.density < self.min_density and self.density_safe_size > 0:
            for _ in range(5):
                value = random.randrange(0, self.density_safe_size)
                r = self.density_safe[value]
                if not PySet_Contains(self.unfeasible_functions, r):
                    return r
            return r
        for _ in range(5):
            r = random.randrange(0, self.nfunctions)
            if not PySet_Contains(self.unfeasible_functions, r):
                return r
        return r    

    cpdef double avg_fitness(self, Py_ssize_t x):
        cdef unsigned int _times = self.times[x]
        if _times == 0:
            return 0.0
        return self.fitness[x] / _times

    cpdef bint comparison(self, int best, int comp):
        if best == comp:
            if self.density < self.min_density:
                if self.density_safe_size == 1:
                    return False
                if PySet_GET_SIZE(self.unfeasible_functions) >= (self.density_safe_size - 1):
                    return False
            if PySet_Contains(self.unfeasible_functions, comp):
                return False
            if PySet_GET_SIZE(self.unfeasible_functions) >= (self.nfunctions - 1):
                return False
            return True
        return False

    cpdef int tournament(self) except -1:
        cdef int best, comp
        cdef double best_fit, comp_fit
        cdef unsigned int *nargs = self.nargs.data.as_uints
        if self.nfunctions == 1:
            return 0
        best = self.random_function()
        best_fit = self.avg_fitness(best)
        for i in range(1, self.tournament_size):
            comp = self.random_function()
            while self.comparison(best, comp):
                comp = self.random_function()
            comp_fit = self.avg_fitness(comp)
            if comp_fit > best_fit:
                best_fit = comp_fit
                best = comp
            elif comp_fit == best_fit and nargs[comp] > nargs[best]:
                best_fit = comp_fit
                best = comp                
        return best
