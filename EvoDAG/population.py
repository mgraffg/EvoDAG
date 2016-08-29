# Copyright 2015 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
from .node import Function
from .model import Model
import gc


class SteadyState(object):
    def __init__(self, tournament_size=2,
                 classifier=True,
                 labels=None,
                 popsize=10000,
                 random_generations=0,
                 es_extra_test=lambda x: True):
        self._p = []
        self._hist = []
        self._bsf = None
        self._estopping = None
        self._tournament_size = tournament_size
        self._index = None
        self._classifier = classifier
        self._es_extra_test = es_extra_test
        self._labels = labels
        self._logger = logging.getLogger('EvoDAG.Population')
        self._previous_estopping = False
        self._popsize = popsize
        self._inds_replace = 0
        self.generation = 1
        self._random_generations = random_generations

    @property
    def previous_estopping(self):
        """
        Returns whether the last individual set in the population was
        an early stopping individual
        """
        return self._previous_estopping

    @previous_estopping.setter
    def previous_estopping(self, v):
        self._previous_estopping = v

    @property
    def hist(self):
        "List containing all the individuals generated"
        return self._hist

    @property
    def bsf(self):
        "Best so far"
        return self._bsf

    @property
    def estopping(self):
        "Early stopping individual"
        return self._estopping

    @estopping.setter
    def estopping(self, v):
        self.previous_estopping = False
        if v.fitness_vs is None:
            return None
        flag = False
        if self.estopping is None:
            if not self._es_extra_test(v):
                return None
            self._estopping = v
            flag = True
        elif v.fitness_vs > self.estopping.fitness_vs:
            if not self._es_extra_test(v):
                return None
            self._estopping = v
            flag = True
        if flag:
            self.previous_estopping = flag
            vfvs = v.fitness_vs
            self._logger.info('(%i) ES: %0.4f %0.4f' % (v.position,
                                                        v.fitness,
                                                        vfvs))

    @bsf.setter
    def bsf(self, v):
        flag = False
        if self.bsf is None:
            self._bsf = v
            flag = True
        elif v.fitness > self.bsf.fitness:
            self._bsf = v
            flag = True
        if flag:
            if v.fitness_vs is None:
                fvs = ""
            else:
                fvs = "%0.4f" % v.fitness_vs
            fts = "%0.4f" % v.fitness
            self._logger.log(logging.INFO-1,
                             '(%(position)s) BSF: %(fts)s %(fvs)s',
                             {'fts': fts, 'fvs': fvs, 'position': v.position})

    def model(self, v=None):
        "Returns the model of node v"
        if v is None:
            v = self.estopping
        hist = self.hist
        trace = self.trace(v)
        m = Model(trace, hist, classifier=self._classifier,
                  labels=self._labels)
        return m

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        trace_map = {}
        self._trace(n, trace_map)
        s = list(trace_map.keys())
        s.sort()
        return s

    def _trace(self, n, trace_map):
        if n.position in trace_map:
            return
        else:
            trace_map[n.position] = 1
        if isinstance(n, Function):
            if isinstance(n.variable, list):
                for x in n.variable:
                    self._trace(self.hist[x], trace_map)
            else:
                self._trace(self.hist[n.variable], trace_map)

    def add(self, v):
        "Add an individual to the population"
        self.population.append(v)
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v

    def clean(self, v):
        if self.estopping is not None and v == self.estopping:
            return
        v.y = None
        v.hy = None

    def replace(self, v):
        """Replace an individual selected by negative tournament selection with
        individual v"""
        if self.popsize < self._popsize:
            return self.add(v)
        k = self.tournament(negative=True)
        self.clean(self.population[k])
        self.population[k] = v
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._inds_replace += 1
        if self._inds_replace == self._popsize:
            self._inds_replace = 0
            self.generation += 1
            gc.collect()

    @property
    def popsize(self):
        return len(self.population)

    @property
    def population(self):
        "List containing the population"
        return self._p

    def random_selection(self, negative=False):
        return np.random.randint(self.popsize)

    def tournament(self, negative=False):
        """Tournament selection and when negative is True it performs negative
        tournament selection"""
        if self.generation <= self._random_generations:
            return self.random_selection(negative=negative)
        if self._index is None or self._index.shape[0] != self.popsize:
            self._index = np.arange(self.popsize)
        np.random.shuffle(self._index)
        vars = self._index[:self._tournament_size]
        fit = [self.population[x].fitness for x in vars]
        if negative:
            index = np.argsort(fit)[0]
        else:
            index = np.argsort(fit)[-1]
        return vars[index]


class Generational(SteadyState):
    "Generational GP using a steady-state as base"
    def __init__(self, *args, **kwargs):
        self._inner = []
        super(Generational, self).__init__(*args, **kwargs)

    def replace(self, v):
        if self.popsize < self._popsize:
            return self.add(v)
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v
        self._inner.append(v)
        if len(self._inner) == self._popsize:
            [self.clean(x) for x in self.population]
            self._p = self._inner
            self._inner = []
            self.generation += 1
            gc.collect()
