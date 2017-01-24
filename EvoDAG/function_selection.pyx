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


cdef class FunctionSelection:
    def __cinit__(self, nfunctions=0, seed=0, tournament_size=2):
        self.fitness = array.clone(array.array('d'), nfunctions, zero=True)
        self.times = array.clone(array.array('I'), nfunctions, zero=True)
        self.nfunctions = nfunctions
        self.tournament_size = tournament_size
        random.seed(seed)

    def __setitem__(self, k, v):
        self.fitness[k] += v
        self.times[k] += 1

    cpdef int random_function(self) except -1:
        return random.randrange(0, self.nfunctions)

    cpdef double avg_fitness(self, Py_ssize_t x):
        cdef unsigned int _times = self.times[x]
        if _times == 0:
            return 0.0
        return self.fitness[x] / _times

    cpdef int tournament(self) except -1:
        cdef int best, comp
        cdef double best_fit, comp_fit
        if self.nfunctions == 1:
            return 0
        best = self.random_function()
        best_fit = self.avg_fitness(best)
        for i in range(1, self.tournament_size):
            comp = self.random_function()
            while comp == best:
                comp = self.random_function()
            comp_fit = self.avg_fitness(comp)
            if comp_fit > best_fit:
                best_fit = comp_fit
                best = comp
        return best
