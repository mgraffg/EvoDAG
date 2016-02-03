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


class Population(object):
    def __init__(self, tournament_size=2):
        self._p = []
        self._hist = []
        self._bsf = None
        self._estopping = None
        self._tournament_size = tournament_size
        self._index = None
        self._logger = logging.getLogger('RGP.Population')

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
        if v.fitness_vs is None:
            return None
        flag = False
        if self.estopping is None:
            self._estopping = v
            flag = True
        elif v.fitness_vs > self.estopping.fitness_vs:
            self._estopping = v
            flag = True
        if flag:
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

    def add(self, v):
        "Add an individual to the population"
        self._p.append(v)
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v

    def replace(self, v):
        """Replace an individual selected by negative tournament selection with
        individual v"""
        k = self.tournament(negative=True)
        self._p[k] = v
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v

    @property
    def popsize(self):
        return len(self._p)

    @property
    def population(self):
        "List containing the population"
        return self._p

    def tournament(self, negative=False):
        """Tournament selection and when negative is True it performs negative
        tournament selection"""
        if self._index is None or self._index.shape[0] != self.popsize:
            self._index = np.arange(self.popsize)
        np.random.shuffle(self._index)
        vars = self._index[:self._tournament_size]
        fit = map(lambda x: self.population[x].fitness, vars)
        if negative:
            index = np.argsort(fit)[0]
        else:
            index = np.argsort(fit)[-1]
        return vars[index]
