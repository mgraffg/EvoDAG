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


from test_root import cl, X
from EvoDAG.model import Model
from EvoDAG.node import Variable, Add, Mul, Sin
from EvoDAG.gp import Individual
from nose.tools import assert_almost_equals
import numpy as np


def test_indindividual_decision_function():
    Add.nargs = 2
    Mul.nargs = 2
    vars = Model.convert_features(X)
    for x in vars:
        x._eval_ts = x._eval_tr.copy()
    vars = [Variable(k, weight=np.ones(1)) for k in range(len(vars))]
    for i in range(len(vars)):
        ind = Individual([vars[i]])
        ind.decision_function(X)
        hy = ind._ind[0].hy.tonparray()
        [assert_almost_equals(a, b) for a, b in zip(X[:, i], hy)]
    ind = Individual([Sin(0, weight=np.ones(1)),
                      Add(range(2), np.ones(2)), vars[0], vars[-1]])
    ind.decision_function(X)
    print(ind._ind[0].hy, ind._ind[1].hy)
    hy = ind._ind[0].hy.tonparray()
    y = np.sin(X[:, 0] + X[:, -1])
    [assert_almost_equals(a, b) for a, b in zip(y, hy)]
    y = np.sin((X[:, 0] + X[:, 1]) * X[:, 0] + X[:, 2])
    ind = Individual([Sin(0, weight=1), Add(range(2), weight=np.ones(2)),
                      Mul(range(2), weight=1),
                      Add(range(2), weight=np.ones(2)),
                      vars[0], vars[1], vars[0], vars[2]])
    ind.decision_function(X)
    # assert v.hy.SSE(v.hy_test) == 0
    hy = ind._ind[0].hy.tonparray()
    [assert_almost_equals(a, b) for a, b in zip(hy, y)]
    

def test_gp_population_full():
    Add.nargs = 2
    Mul.nargs = 2
    from EvoDAG.gp import Population
    from EvoDAG import EvoDAG
    fs = EvoDAG()._function_set

    class Population2(Population):
        def __init__(self, *args, **kwargs):
            super(Population2, self).__init__(*args, **kwargs)
            self._funcs = [Add, Sin]
            self._terms = [2, 0]

        def random_function(self):
            func = self._funcs.pop()
            if func.nargs == 1:
                return func(0, weight=1)
            return func(range(func.nargs), weight=np.ones(func.nargs))

        def random_terminal(self):
            return Variable(self._terms.pop(), 1)

    pop = Population2(fs, nterminals=3)
    ind = pop.create_random_ind_full(depth=2)
    assert len(pop._funcs) == 0 and len(pop._terms) == 0
    assert isinstance(ind[0], Sin) and isinstance(ind[1], Add)
    assert ind[2].variable == 0 and ind[3].variable == 2
    ind = Individual(ind)
    print(X.shape, ind.individual)
    hy = ind.decision_function(X)
    assert hy.isfinite()


def test_gp_population_grow():
    Add.nargs = 2
    Mul.nargs = 2
    from EvoDAG.gp import Population
    from EvoDAG import EvoDAG
    fs = EvoDAG()._function_set

    class Population2(Population):
        def __init__(self, *args, **kwargs):
            super(Population2, self).__init__(*args, **kwargs)
            self._funcs = [Sin, Add]
            self._terms = [2, 0]
            self._g_funcs = [0, 1, 1]

        def random_function(self):
            func = self._funcs.pop()
            if func.nargs == 1:
                return func(0, weight=1)
            return func(range(func.nargs), weight=np.ones(func.nargs))

        def random_terminal(self):
            return Variable(self._terms.pop(), 1)

        def grow_use_function(self, depth=0):
            if depth == 0:
                return False
            return self._g_funcs.pop()

    pop = Population2(fs, nterminals=3)
    ind = pop.create_random_ind_grow(depth=2)
    print(ind)
    assert len(pop._funcs) == 0 and len(pop._terms) == 0
    assert len(pop._g_funcs) == 0
    assert isinstance(ind[0], Add) and isinstance(ind[1], Sin)
    assert ind[2].variable == 0 and ind[3].variable == 2
    ind = Individual(ind)
    print(X.shape, ind.individual)
    hy = ind.decision_function(X)
    assert hy.isfinite()


def test_gp_create_population():
    Add.nargs = 2
    Mul.nargs = 2
    from EvoDAG.gp import Population
    from EvoDAG import EvoDAG
    fs = EvoDAG()._function_set
    pop = Population(fs, nterminals=X.shape[1])
    inds = pop.create_population(X=X)
    for i in inds:
        assert i[0].hy.isfinite()
