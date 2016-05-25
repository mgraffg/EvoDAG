# Copyright 2016 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .node import Function, Variable, Add
from .model import Model
import numpy as np
import logging


class Individual(object):
    """Object to store in individual on prefix notation"""

    def __init__(self, ind, classifier=True, labels=None):
        self._ind = ind
        self._pos = 0
        self._classifier = classifier
        self._labels = labels
        self._X = None

    @property
    def individual(self):
        "Individual"
        return self._ind

    def decision_function(self, X):
        "Decision function i.e. the raw data of the prediction"
        self._X = Model.convert_features(X)
        self._eval()
        return self._ind[0].hy

    def _eval(self):
        "Evaluates a individual using recursion and self._pos as pointer"
        pos = self._pos
        self._pos += 1
        node = self._ind[pos]
        if isinstance(node, Function):
            args = [self._eval() for x in range(node.nargs)]
            node.eval(args)
            for x in args:
                x.hy = None
                x.hy_test = None
        else:
            node.eval(self._X)
        return node


class Population(object):
    "Population of a tree-based GP system"
    def __init__(self, function_set=None, nterminals=None, seed=0):
        assert function_set is not None
        assert nterminals is not None
        self._function_set = function_set
        self._nterminals = nterminals
        self._logger = logging.getLogger('EvoDAG.gp')
        np.random.seed(seed)

    def random_function(self):
        func = np.random.randint(len(self._function_set))
        func = self._function_set[func]
        if issubclass(func, Add) and func.nargs > 1:
            return func(range(func.nargs), weight=np.ones(func.nargs))
        elif func.nargs == 1:
            return func(0, weight=1)
        return func(range(func.nargs), weight=1)

    def random_terminal(self):
        terminal = np.random.randint(self._nterminals)
        return Variable(terminal, 1)

    def create_random_ind_full(self, depth=0):
        "Random individual using full method"
        lst = []
        self._create_random_ind_full(depth=depth, output=lst)
        return lst

    def _create_random_ind_full(self, depth=0, output=None):
        if depth == 0:
            output.append(self.random_terminal())
        else:
            func = self.random_function()
            output.append(func)
            depth -= 1
            [self._create_random_ind_full(depth=depth, output=output)
             for x in range(func.nargs)]

    def grow_use_function(self, depth=0):
        "Select either function or terminal in grow method"
        if depth == 0:
            return False
        if depth == self._depth:
            return True
        return np.random.random() < 0.5

    def create_random_ind_grow(self, depth=0):
        "Random individual using grow method"
        lst = []
        self._depth = depth
        self._create_random_ind_grow(depth=depth, output=lst)
        return lst

    def _create_random_ind_grow(self, depth=0, output=None):
        if self.grow_use_function(depth=depth):
            func = self.random_function()
            output.append(func)
            depth -= 1
            [self._create_random_ind_grow(depth=depth, output=output)
             for x in range(func.nargs)]
        else:
            output.append(self.random_terminal())

    def create_population(self, popsize=1000, min_depth=2,
                          max_depth=4,
                          X=None):
        "Creates random population using ramped half-and-half method"
        import itertools
        args = [x for x in itertools.product(range(min_depth,
                                                   max_depth+1),
                                             [True, False])]
        index = 0
        output = []
        while len(output) < popsize:
            depth, full = args[index]
            index += 1
            if index >= len(args):
                index = 0
            if full:
                ind = self.create_random_ind_full(depth=depth)
            else:
                ind = self.create_random_ind_grow(depth=depth)
            flag = True
            if X is not None:
                x = Individual(ind)
                x.decision_function(X)
                flag = x.individual[0].isfinite()
            l_vars = (flag, len(output), full, depth, len(ind))
            l_str = " flag: %s len(output): %s full: %s depth: %s len(ind): %s"
            self._logger.debug(l_str % l_vars)
            if flag:
                output.append(ind)
        return output
