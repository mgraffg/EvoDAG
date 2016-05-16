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


import numpy as np
from nose.tools import assert_almost_equals
try:
    from mock import MagicMock
except ImportError:
    from unittest.mock import MagicMock
from EvoDAG.node import Variable
from EvoDAG.sparse_array import SparseArray


cl = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2])

X = np.array([[5.1, 3.5, 1.4, 0.2],
              [4.9,  3. ,  1.4,  0.2],
              [4.7,  3.2,  1.3,  0.2],
              [4.6,  3.1,  1.5,  0.2],
              [5. ,  3.6,  1.4,  0.2],
              [5.4,  3.9,  1.7,  0.4],
              [4.6,  3.4,  1.4,  0.3],
              [5. ,  3.4,  1.5,  0.2],
              [4.4,  2.9,  1.4,  0.2],
              [4.9,  3.1,  1.5,  0.1],
              [5.4,  3.7,  1.5,  0.2],
              [4.8,  3.4,  1.6,  0.2],
              [4.8,  3. ,  1.4,  0.1],
              [4.3,  3. ,  1.1,  0.1],
              [5.8,  4. ,  1.2,  0.2],
              [5.7,  4.4,  1.5,  0.4],
              [5.4,  3.9,  1.3,  0.4],
              [5.1,  3.5,  1.4,  0.3],
              [5.7,  3.8,  1.7,  0.3],
              [5.1,  3.8,  1.5,  0.3],
              [5.4,  3.4,  1.7,  0.2],
              [5.1,  3.7,  1.5,  0.4],
              [4.6,  3.6,  1. ,  0.2],
              [5.1,  3.3,  1.7,  0.5],
              [4.8,  3.4,  1.9,  0.2],
              [5. ,  3. ,  1.6,  0.2],
              [5. ,  3.4,  1.6,  0.4],
              [5.2,  3.5,  1.5,  0.2],
              [5.2,  3.4,  1.4,  0.2],
              [4.7,  3.2,  1.6,  0.2],
              [4.8,  3.1,  1.6,  0.2],
              [5.4,  3.4,  1.5,  0.4],
              [5.2,  4.1,  1.5,  0.1],
              [5.5,  4.2,  1.4,  0.2],
              [4.9,  3.1,  1.5,  0.1],
              [5. ,  3.2,  1.2,  0.2],
              [5.5,  3.5,  1.3,  0.2],
              [4.9,  3.1,  1.5,  0.1],
              [4.4,  3. ,  1.3,  0.2],
              [5.1,  3.4,  1.5,  0.2],
              [5. ,  3.5,  1.3,  0.3],
              [4.5,  2.3,  1.3,  0.3],
              [4.4,  3.2,  1.3,  0.2],
              [5. ,  3.5,  1.6,  0.6],
              [5.1,  3.8,  1.9,  0.4],
              [4.8,  3. ,  1.4,  0.3],
              [5.1,  3.8,  1.6,  0.2],
              [4.6,  3.2,  1.4,  0.2],
              [5.3,  3.7,  1.5,  0.2],
              [5. ,  3.3,  1.4,  0.2],
              [7. ,  3.2,  4.7,  1.4],
              [6.4,  3.2,  4.5,  1.5],
              [6.9,  3.1,  4.9,  1.5],
              [5.5,  2.3,  4. ,  1.3],
              [6.5,  2.8,  4.6,  1.5],
              [5.7,  2.8,  4.5,  1.3],
              [6.3,  3.3,  4.7,  1.6],
              [4.9,  2.4,  3.3,  1. ],
              [6.6,  2.9,  4.6,  1.3],
              [5.2,  2.7,  3.9,  1.4],
              [5. ,  2. ,  3.5,  1. ],
              [5.9,  3. ,  4.2,  1.5],
              [6. ,  2.2,  4. ,  1. ],
              [6.1,  2.9,  4.7,  1.4],
              [5.6,  2.9,  3.6,  1.3],
              [6.7,  3.1,  4.4,  1.4],
              [5.6,  3. ,  4.5,  1.5],
              [5.8,  2.7,  4.1,  1. ],
              [6.2,  2.2,  4.5,  1.5],
              [5.6,  2.5,  3.9,  1.1],
              [5.9,  3.2,  4.8,  1.8],
              [6.1,  2.8,  4. ,  1.3],
              [6.3,  2.5,  4.9,  1.5],
              [6.1,  2.8,  4.7,  1.2],
              [6.4,  2.9,  4.3,  1.3],
              [6.6,  3. ,  4.4,  1.4],
              [6.8,  2.8,  4.8,  1.4],
              [6.7,  3. ,  5. ,  1.7],
              [6. ,  2.9,  4.5,  1.5],
              [5.7,  2.6,  3.5,  1. ],
              [5.5,  2.4,  3.8,  1.1],
              [5.5,  2.4,  3.7,  1. ],
              [5.8,  2.7,  3.9,  1.2],
              [6. ,  2.7,  5.1,  1.6],
              [5.4,  3. ,  4.5,  1.5],
              [6. ,  3.4,  4.5,  1.6],
              [6.7,  3.1,  4.7,  1.5],
              [6.3,  2.3,  4.4,  1.3],
              [5.6,  3. ,  4.1,  1.3],
              [5.5,  2.5,  4. ,  1.3],
              [5.5,  2.6,  4.4,  1.2],
              [6.1,  3. ,  4.6,  1.4],
              [5.8,  2.6,  4. ,  1.2],
              [5. ,  2.3,  3.3,  1. ],
              [5.6,  2.7,  4.2,  1.3],
              [5.7,  3. ,  4.2,  1.2],
              [5.7,  2.9,  4.2,  1.3],
              [6.2,  2.9,  4.3,  1.3],
              [5.1,  2.5,  3. ,  1.1],
              [5.7,  2.8,  4.1,  1.3],
              [6.3,  3.3,  6. ,  2.5],
              [5.8,  2.7,  5.1,  1.9],
              [7.1,  3. ,  5.9,  2.1],
              [6.3,  2.9,  5.6,  1.8],
              [6.5,  3. ,  5.8,  2.2],
              [7.6,  3. ,  6.6,  2.1],
              [4.9,  2.5,  4.5,  1.7],
              [7.3,  2.9,  6.3,  1.8],
              [6.7,  2.5,  5.8,  1.8],
              [7.2,  3.6,  6.1,  2.5],
              [6.5,  3.2,  5.1,  2. ],
              [6.4,  2.7,  5.3,  1.9],
              [6.8,  3. ,  5.5,  2.1],
              [5.7,  2.5,  5. ,  2. ],
              [5.8,  2.8,  5.1,  2.4],
              [6.4,  3.2,  5.3,  2.3],
              [6.5,  3. ,  5.5,  1.8],
              [7.7,  3.8,  6.7,  2.2],
              [7.7,  2.6,  6.9,  2.3],
              [6. ,  2.2,  5. ,  1.5],
              [6.9,  3.2,  5.7,  2.3],
              [5.6,  2.8,  4.9,  2. ],
              [7.7,  2.8,  6.7,  2. ],
              [6.3,  2.7,  4.9,  1.8],
              [6.7,  3.3,  5.7,  2.1],
              [7.2,  3.2,  6. ,  1.8],
              [6.2,  2.8,  4.8,  1.8],
              [6.1,  3. ,  4.9,  1.8],
              [6.4,  2.8,  5.6,  2.1],
              [7.2,  3. ,  5.8,  1.6],
              [7.4,  2.8,  6.1,  1.9],
              [7.9,  3.8,  6.4,  2. ],
              [6.4,  2.8,  5.6,  2.2],
              [6.3,  2.8,  5.1,  1.5],
              [6.1,  2.6,  5.6,  1.4],
              [7.7,  3. ,  6.1,  2.3],
              [6.3,  3.4,  5.6,  2.4],
              [6.4,  3.1,  5.5,  1.8],
              [6. ,  3. ,  4.8,  1.8],
              [6.9,  3.1,  5.4,  2.1],
              [6.7,  3.1,  5.6,  2.4],
              [6.9,  3.1,  5.1,  2.3],
              [5.8,  2.7,  5.1,  1.9],
              [6.8,  3.2,  5.9,  2.3],
              [6.7,  3.3,  5.7,  2.5],
              [6.7,  3. ,  5.2,  2.3],
              [6.3,  2.5,  5. ,  1.9],
              [6.5,  3. ,  5.2,  2. ],
              [6.2,  3.4,  5.4,  2.3],
              [5.9,  3. ,  5.1,  1.8]])


def test_features():
    from EvoDAG.sparse_array import SparseArray
    from EvoDAG import RootGP
    gp = RootGP(generations=1)
    gp.X = X
    assert gp.nvar == 4
    print(gp.X)
    assert isinstance(gp.X[0], Variable)
    assert isinstance(gp.X[0].hy, SparseArray)
    gp.Xtest = X
    assert gp.X[0].hy.SSE(gp.X[0].hy_test) == 0


def test_create_population():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    assert_almost_equals(gp.population.popsize, gp.popsize)
    # a = map(lambda (x, y): x == y, zip(gp.population.population,
    #                                    gp.population._hist))
    a = [x == y1 for x, y1 in zip(gp.population.population,
                                  gp.population._hist)]
    assert np.all(a)
    l = [x.variable for x in gp.population.population]
    l.sort()
    # assert np.all(map(lambda (x, y): x == y, enumerate(l)))
    assert np.all([x == y1 for x, y1 in enumerate(l)])


def test_create_population2():
    from EvoDAG import RootGP
    from EvoDAG.node import Function
    gp = RootGP(generations=1, popsize=10)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    for i in gp.population.population[4:]:
        assert isinstance(i, Function)


def test_best_so_far():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(gp.popsize))
    np.random.randint = mock
    gp.create_population()
    p = gp.population.population
    index = np.argsort([x.fitness for x in p])[-1]
    print(p[index].fitness, gp.population.bsf.fitness)
    assert gp.population.bsf == p[index]
    np.random.randint = randint


def test_early_stopping():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(gp.popsize))
    np.random.randint = mock
    gp.create_population()
    p = gp.population.population
    fit = np.array([x.fitness_vs for x in p])
    best = fit.max()
    index = np.where(best == fit)[0][0]
    assert gp.population.estopping == p[index]
    np.random.randint = randint


def test_variable():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    Xtest = [x for x in X]
    Xtest[0] = Xtest[0] + np.inf
    gp.Xtest = Xtest
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(gp.popsize))
    np.random.randint = mock
    var = gp.random_leaf()
    assert var.isfinite()
    assert var.hy.isfinite()
    np.random.randint = randint


def test_random_leaf():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4, tr_fraction=1)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock(return_value=0)
    np.random.randint = mock
    mask = gp._mask.tonparray().astype(np.bool)
    weight = np.linalg.lstsq(X[mask, 0][:, np.newaxis], y[mask])[0][0]
    var = gp.random_leaf()
    assert isinstance(var, Variable)
    print(weight, var.weight)
    assert_almost_equals(weight, var.weight[0])
    np.random.randint = randint


def test_random_leaf_inf():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, classifier=False, popsize=4, tr_fraction=1)
    Xc = [x for x in X]
    Xc[0] = Xc[0] + np.inf
    gp.X = Xc
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(2))
    np.random.randint = mock
    gp.random_leaf()
    try:
        assert mock()
        assert False
    except Exception:
        pass
    np.random.randint = randint


def test_classification_y():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    assert gp._ytr.SSE(gp.y) > 0
    assert gp._ytr.sum() == 0
    assert gp.y.sum() < 0


def test_regression_y():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4, classifier=False)
    assert not gp._classifier
    gp.X = X
    gp.y = cl
    assert gp._ytr.SSE(gp.y) > 0
    gp = RootGP(generations=1, popsize=4, classifier=False, tr_fraction=1.0)
    gp.X = X
    gp.y = cl
    assert gp._ytr.SSE(gp.y) == 0


def test_fitness():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    l = gp.random_leaf()
    assert l.fitness < 0


def test_mask_vs():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    m = ~ gp._mask.tonparray().astype(np.bool)
    f = np.zeros(gp._mask.size())
    f[y == -1] = 0.5 / (y[m] == -1).sum()
    f[y == 1] = 0.5 / (y[m] == 1).sum()
    f[~m] = 0
    assert gp._mask_vs.SSE(SparseArray.fromlist(f)) == 0


def test_BER():
    from EvoDAG.node import Add
    from EvoDAG import RootGP
    from EvoDAG.utils import BER
    gp = RootGP(generations=1, popsize=4)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    m = ~ gp._mask.tonparray().astype(np.bool)
    v = gp.random_leaf()
    v1 = gp.random_leaf()
    v1 = gp.random_leaf()
    a = Add([0, 1], ytr=gp._ytr, mask=gp._mask)
    a.eval([v, v1])
    hy = a.hy.sign()
    b = BER(y[m], hy.tonparray()[m])
    gp.fitness_vs(a)
    print(b, a.fitness_vs * 100)
    assert_almost_equals(b, -a.fitness_vs * 100)
    # assert False


def test_tournament():
    from EvoDAG import RootGP
    gp = RootGP(generations=1,
                tournament_size=4,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(gp.popsize))
    np.random.randint = mock
    gp.create_population()
    j = gp.population.tournament()
    index = np.argsort([x.fitness for x in gp.population.population])[-1]
    assert j == index
    np.random.randint = randint


def test_tournament_negative():
    from EvoDAG import RootGP
    gp = RootGP(generations=1,
                tournament_size=4,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock()
    mock.side_effect = list(range(gp.popsize))
    np.random.randint = mock
    gp.create_population()
    j = gp.population.tournament(negative=True)
    index = np.argsort([x.fitness for x in gp.population.population])[0]
    assert j == index
    np.random.randint = randint


def test_random_offspring():
    from EvoDAG import RootGP
    from EvoDAG.node import Add
    gp = RootGP(generations=1,
                seed=1,
                tournament_size=2,
                popsize=10)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    randint = np.random.randint
    mock = MagicMock(return_value=0)
    np.random.randint = mock
    a = gp.random_offspring()
    np.random.randint.assert_called_with(len(gp.function_set))
    assert isinstance(a, Add)
    assert np.isfinite(a.fitness)
    np.random.randint = randint


def test_replace_individual():
    from EvoDAG import RootGP
    gp = RootGP(generations=1,
                tournament_size=2,
                popsize=10)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    a = gp.random_offspring()
    assert a.position == 0
    gp.population.replace(a)
    assert np.any([x == a for x in gp.population.population])
    assert a.position == len(gp.population.population)


def test_X_sparse():
    from EvoDAG import RootGP
    from EvoDAG.sparse_array import SparseArray
    gp = RootGP(generations=1,
                tournament_size=2,
                popsize=10)
    X1 = list(map(SparseArray.fromlist, X.T))
    gp.X = X1


def test_fit_stopping_criteria_gens():
    from EvoDAG import RootGP
    from EvoDAG.node import Add
    Add.nargs = 2
    gp = RootGP(generations=2,
                early_stopping_rounds=None,
                tournament_size=2,
                seed=1,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    for i in range(gp.popsize):
        assert not gp.stopping_criteria()
        a = gp.random_offspring()
        gp.population.replace(a)
    assert gp.stopping_criteria()


def test_fit_stopping_criteria_estopping():
    from EvoDAG import RootGP
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=4,
                seed=0,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    print(len(gp.population.hist) - gp.population.estopping.position)
    while not gp.stopping_criteria():
        a = gp.random_offspring()
        gp.population.replace(a)
    assert (len(gp.population.hist) - gp.population.estopping.position) > 4


def test_fit():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=4).fit(X, y, test_set=X)
    assert np.isfinite(gp.population.estopping.fitness)
    assert np.isfinite(gp.population.estopping.fitness_vs)
    assert gp.population.estopping.hy.isfinite()
    assert len(gp.population.hist) > 4


def test_logging():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    RootGP(generations=np.inf,
           tournament_size=2,
           early_stopping_rounds=-1,
           seed=0,
           popsize=10).fit(X, y, test_set=X)


def test_infite_evolution():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    try:
        RootGP(generations=np.inf,
               tournament_size=2,
               tr_fraction=1,
               early_stopping_rounds=-1,
               seed=0,
               popsize=10).fit(X, y, test_set=X)
        assert False
    except RuntimeError:
        pass


def test_predict():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    es = gp.population.estopping
    assert gp.decision_function().SSE(es.hy_test.boundaries()) == 0
    hy_test = es.hy_test.boundaries()
    assert gp.decision_function(X=X[-10:]).SSE(hy_test) == 0
    hy = gp.decision_function(X=X[-10:])
    _ = gp.predict(X=X[-10:])
    assert SparseArray.fromlist(_).SSE(hy.sign()) == 0
    assert len(gp.Xtest)


def test_trace():
    from EvoDAG import RootGP
    from EvoDAG.node import Add
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    Add.nargs = 3
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                function_set=[Add],
                early_stopping_rounds=-1,
                seed=0,
                popsize=4)
    gp.X = X[:-10]
    gp.Xtest = X[-10:]
    gp.y = y[:-10]
    gp.create_population()
    a = gp.random_offspring()
    gp.population.replace(a)
    print(a.position, a.variable, a._weight, gp.population.hist[0].variable)
    s = gp.trace(a)
    assert len(s) == 4


def test_class_values():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 0
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    assert np.all(gp._labels == np.array([-1, 0]))


def test_multiclass():
    from EvoDAG import RootGP
    y = cl.copy()
    ncl = np.unique(y).shape[0]
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100).fit(X, y)
    assert len(gp._multiclass_instances) == ncl
    assert gp._multiclass


def test_multiclass_decision_function():
    from EvoDAG import RootGP
    y = cl.copy()
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100).fit(X, y, test_set=X)
    d = gp.decision_function()
    assert len(d) == np.unique(y).shape[0]


def test_multiclass_predict():
    from EvoDAG import RootGP
    y = cl.copy()
    y[y == 0] = 3
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100).fit(X, y, test_set=X)
    d = gp.predict()
    assert np.unique(d).shape[0] == np.unique(y).shape[0]
    assert np.all(np.unique(d) == np.unique(y))


def test_get_params():
    from EvoDAG import RootGP
    # y = cl.copy()
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100)
    p = gp.get_params()
    assert p['generations'] == np.inf
    assert p['popsize'] == 100
    assert p['seed'] == 0


def test_get_clone():
    from EvoDAG import RootGP
    # y = cl.copy()
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100)
    gp1 = gp.clone()
    print(gp.popsize, gp1.popsize)
    assert gp.popsize == gp1.popsize
    assert gp._generations == gp1._generations
    assert gp._seed == gp1._seed


def test_labels():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = 2
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    m = gp.model()
    hy = m.predict(X=X[:-10])
    print(np.unique(hy))
    print(np.array([1, 2]))
    assert np.all(np.unique(hy) == np.array([1, 2]))


def test_height():
    from EvoDAG import RootGP
    from EvoDAG.node import Mul
    gp = RootGP(generations=1,
                seed=1,
                tournament_size=2,
                popsize=5)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    assert np.all([x.height == 0 for x in gp.population.population[:4]])
    n = gp.population.population[-1]
    assert n.height == 1
    args = [3, 4]
    f = gp._random_offspring(Mul, args)
    assert f.height == 2


def test_regression():
    from EvoDAG import RootGP
    from EvoDAG.sparse_array import SparseArray
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    gp = RootGP(classifier=False,
                popsize=10,
                generations=2).fit([SparseArray.fromlist(x)], y,
                                   test_set=[SparseArray.fromlist(x)])
    model = gp.model()
    yh = gp.predict()
    assert not model._classifier
    yh1 = model.predict(X=[SparseArray.fromlist(x)])
    spf = SparseArray.fromlist
    assert spf(yh).SSE(spf(yh1)) == 0


def test_unique():
    from EvoDAG import RootGP
    mock = MagicMock(side_effect=RuntimeError('Mock'))
    ui = RootGP.unique_individual
    RootGP.unique_individual = mock
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                unique_individuals=True,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100)
    try:
        gp.fit(X, cl)
    except RuntimeError:
        pass
    RootGP.unique_individual = ui


def test_RSE():
    from EvoDAG import RootGP
    from EvoDAG.sparse_array import SparseArray
    from EvoDAG.utils import RSE as rse
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    y[10:12] = 0
    gp = RootGP(classifier=False,
                popsize=10,
                generations=2).fit([SparseArray.fromlist(x)], y,
                                   test_set=[SparseArray.fromlist(x)])
    model = gp.model()
    yh = gp.predict()
    assert not model._classifier
    model.predict(X=[SparseArray.fromlist(x)])
    gp._mask = SparseArray.fromlist([2] * yh.shape[0])
    gp.fitness_vs(model._hist[-1])
    print(rse(y, yh), model._hist[-1].fitness_vs)
    assert_almost_equals(rse(y, yh),
                         -model._hist[-1].fitness_vs)


def test_RSE_avg_zero():
    from EvoDAG import EvoDAG
    from EvoDAG.sparse_array import SparseArray

    class EvoDAG2(EvoDAG):
        def __init__(self, **kw):
            super(EvoDAG2, self).__init__(**kw)
            self._times = 0

        def set_regression_mask(self, v):
            mask = np.ones(v.size())
            if self._times == 0:
                mask[10:12] = 0
            else:
                mask[10:13] = 0
            self._mask = SparseArray.fromlist(mask)
            self._times += 1

    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    y[10:12] = 0
    gp = EvoDAG2(classifier=False,
                 popsize=10,
                 generations=2)
    gp.X = [SparseArray.fromlist(x)]
    gp.y = y
    print(gp._times)
    assert gp._times == 2
    gp.create_population()
    while not gp.stopping_criteria():
        a = gp.random_offspring()
        gp.replace(a)


def test_population_as_parameter():
    from EvoDAG import RootGP
    mock = MagicMock(side_effect=RuntimeError('Mock'))
    gp = RootGP(generations=np.inf,
                population_class=mock,
                tournament_size=2,
                unique_individuals=True,
                early_stopping_rounds=-1,
                seed=0,
                popsize=100)
    try:
        gp.fit(X, cl)
        assert False
    except RuntimeError:
        pass


def test_es_extra_test():
    from EvoDAG import RootGP
    from EvoDAG.sparse_array import SparseArray
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    es_extra_test = RootGP.es_extra_test
    RootGP.es_extra_test = MagicMock(side_effect=RuntimeError('Mock'))
    try:
        RootGP(classifier=False,
               popsize=10,
               generations=2).fit([SparseArray.fromlist(x)], y,
                                  test_set=[SparseArray.fromlist(x)])
        assert False
    except RuntimeError:
        RootGP.es_extra_test = es_extra_test


def test_fname():
    from EvoDAG.node import Add
    from EvoDAG import RootGP
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    Add.nargs = 10
    gp = RootGP(classifier=False,
                popsize=10,
                generations=2).fit([SparseArray.fromlist(x)], y,
                                   test_set=[SparseArray.fromlist(x)])
    assert gp.signature.count('Ad10') == 1


def test_unfeasible_counter():
    from EvoDAG import RGP
    gp = RGP(generations=np.inf,
             tournament_size=2,
             early_stopping_rounds=-1,
             seed=0,
             popsize=100)
    assert gp._unfeasible_counter == 0
    assert gp.unfeasible_offspring() is None
    assert gp._unfeasible_counter == 1


def test_replace_population_previous_estopping():
    from EvoDAG import RGP
    gp = RGP(generations=np.inf,
             tournament_size=2,
             early_stopping_rounds=-1,
             seed=0,
             popsize=3)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    gp.unfeasible_offspring()
    es = gp.population.estopping
    for i in range(10):
        n = gp.random_offspring()
        if n.fitness_vs > es.fitness_vs:
            break
    print(es.fitness_vs, n.fitness_vs, gp._unfeasible_counter)
    assert gp._unfeasible_counter >= 1
    gp.replace(n)
    assert gp.population.previous_estopping


def test_add():
    from EvoDAG import RGP
    gp = RGP(generations=np.inf,
             tournament_size=2,
             early_stopping_rounds=-1,
             seed=0,
             popsize=3)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    gp.unfeasible_offspring()
    es = gp.population.estopping
    for i in range(10):
        n = gp.random_offspring()
        if n.fitness_vs > es.fitness_vs:
            break
    gp.add(n)
    assert gp._unfeasible_counter == 0


def test_unfeasible_counter_fit():
    from EvoDAG import RGP

    class RGP2(RGP):
        def replace(self, a):
            self.population.replace(a)

        def add(self, a):
            self.population.add(a)

    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RGP2(generations=10,
              tournament_size=2,
              early_stopping_rounds=-1,
              seed=0,
              popsize=3)
    [gp.unfeasible_offspring() for _ in range(100)]
    gp.fit(X, y)
    # print(len(gp.population.hist))
    assert len(gp.population.hist) <= 3


def test_one_instance():
    from EvoDAG import RGP
    y = cl.copy()
    y[:-1] = -1
    y[-1:] = 1
    gp = RGP(generations=np.inf,
             tournament_size=2,
             early_stopping_rounds=-1,
             seed=0,
             popsize=10).fit(X, y, test_set=X)
    assert gp
