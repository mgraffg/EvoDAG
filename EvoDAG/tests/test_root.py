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
from SparseArray import SparseArray


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


def tonparray(a):
    return np.array(a.full_array())


def test_features():
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
    from EvoDAG import EvoDAG
    gp = EvoDAG(generations=1, classifier=False, pr_variable=1,
                popsize=4)
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
    from EvoDAG import EvoDAG
    from EvoDAG.node import Function
    gp = EvoDAG(generations=1, classifier=False, pr_variable=1,
                popsize=10)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    for i in gp.population.population[4:]:
        assert isinstance(i, Function)


def test_create_population_cl():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Function, Variable, NaiveBayes, NaiveBayesMN
    gp = EvoDAG(generations=1, popsize=50, multiple_outputs=True)
    gp.X = X
    gp.nclasses(cl)
    gp.y = cl.copy()
    gp.create_population()
    flag = False
    for i in gp.population.population:
        assert isinstance(i, Function) or isinstance(i, Variable)
        if isinstance(i, Function):
            if not (isinstance(i, NaiveBayes) or isinstance(i, NaiveBayesMN)):
                flag = True
    assert flag


def test_best_so_far():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, classifier=False, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    # randint = np.random.randint
    # mock = MagicMock()
    # mock.side_effect = list(range(gp.popsize))
    # np.random.randint = mock
    gp.create_population()
    p = gp.population.population
    index = np.argsort([x.fitness for x in p])[-1]
    print(p[index].fitness, gp.population.bsf.fitness)
    assert gp.population.bsf == p[index]
    # np.random.randint = randint


def test_early_stopping():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4, classifier=False)
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
    gp = RootGP(generations=1, classifier=False,
                popsize=4)
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
    gp = RootGP(generations=1, classifier=False, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    randint = np.random.randint
    mock = MagicMock(return_value=0)
    np.random.randint = mock
    mask = tonparray(gp._mask).astype(np.bool)
    weight = np.linalg.lstsq(X[mask, 0][:, np.newaxis], y[mask])[0][0]
    var = gp.random_leaf()
    assert isinstance(var, Variable)
    print(weight, var.weight)
    assert_almost_equals(weight, var.weight)
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
    gp = RootGP(generations=1, multiple_outputs=True, popsize=4)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    gp.nclasses(y)
    gp.y = y
    print(gp._ytr, gp.y)
    for a, b in zip(gp._ytr, gp.y):
        assert a.SSE(b) > 0
        assert a.sum() == 0
        assert b.sum() < 0


def test_regression_y():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4, classifier=False)
    assert not gp._classifier
    gp.X = X
    gp.y = cl.copy()
    assert gp._ytr.SSE(gp.y) > 0
    gp = RootGP(generations=1, popsize=4, classifier=False, tr_fraction=1.0)
    gp.X = X
    gp.y = cl.copy()
    assert gp._ytr.SSE(gp.y) == 0


def test_fitness():
    from EvoDAG import RootGP
    gp = RootGP(generations=1, classifier=False,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    l = gp.random_leaf()
    assert l.fitness < 0


def test_mask_vs():
    from EvoDAG.node import Add
    from EvoDAG import RootGP
    from EvoDAG.utils import BER
    gp = RootGP(generations=1, popsize=4, multiple_outputs=True)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    gp.nclasses(y)
    gp.y = y
    m = np.sign(tonparray(gp._mask_vs)).astype(np.bool)
    v = gp.random_leaf()
    v1 = gp.random_leaf()
    v1 = gp.random_leaf()
    a = Add([0, 1], ytr=gp._ytr, mask=gp._mask)
    a.eval([v, v1])
    hy = SparseArray.argmax(a.hy)
    b = BER(y[m], tonparray(hy)[m])
    gp._bagging_fitness.set_fitness(a)
    print(b, a.fitness_vs * 100)
    assert_almost_equals(b, -a.fitness_vs * 100)


def test_BER():
    from EvoDAG.node import Add
    from EvoDAG import RootGP
    from EvoDAG.utils import BER
    gp = RootGP(generations=1, popsize=4, multiple_outputs=True)
    assert gp._classifier
    gp.X = X
    y = cl.copy()
    gp.nclasses(y)
    gp.y = y
    m = np.sign(tonparray(gp._mask_ts)).astype(np.bool)
    v = gp.random_leaf()
    v1 = gp.random_leaf()
    v1 = gp.random_leaf()
    a = Add([0, 1], ytr=gp._ytr, mask=gp._mask)
    a.eval([v, v1])
    hy = SparseArray.argmax(a.hy)
    b = BER(y[m], tonparray(hy)[m])
    gp._bagging_fitness.fitness(a)
    print(b, a.fitness * 100)
    assert_almost_equals(b, -a.fitness * 100)


def test_tournament():
    from EvoDAG import RootGP
    gp = RootGP(generations=1,
                tournament_size=4,
                classifier=False,
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
                classifier=False,
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
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Sin
    gp = EvoDAG(generations=1,
                function_set=[Add, Sin],
                multiple_outputs=True,
                seed=1,
                tournament_size=2,
                popsize=10)
    gp.X = X
    gp.nclasses(cl)
    gp.y = cl.copy()
    gp.create_population()
    a = gp.random_offspring()
    assert isinstance(a, Add) or isinstance(a, Sin)
    assert np.isfinite(a.fitness)


def test_replace_individual():
    from EvoDAG import RootGP
    gp = RootGP(generations=1,
                tournament_size=2,
                classifier=False,
                popsize=5)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    print(gp.population.popsize)
    a = gp.random_offspring()
    assert a.position == 0
    gp.population.replace(a)
    assert np.any([x == a for x in gp.population.population])
    print(a.position, len(gp.population.population))
    assert a.position == len(gp.population.population)


def test_X_sparse():
    from EvoDAG import RootGP
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
                classifier=False,
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
                classifier=False,
                seed=0,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    while not gp.stopping_criteria():
        a = gp.random_offspring()
        gp.population.replace(a)
    print(len(gp.population.hist) - gp.population.estopping.position)
    assert (len(gp.population.hist) - gp.population.estopping.position) <= 9


def test_fit():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                classifier=False,
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
           classifier=False,
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
                classifier=False,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    es = gp.population.estopping
    assert gp.decision_function().SSE(es.hy_test) == 0
    hy_test = es.hy_test
    assert gp.decision_function(X=X[-10:]).SSE(hy_test) == 0
    hy = gp.decision_function(X=X[-10:])
    _ = gp.predict(X=X[-10:])
    assert SparseArray.fromlist(_).SSE(hy) == 0
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
                classifier=False,
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
    print(len(s), s)
    assert a._weight.shape[0] + 1 == len(s)


def test_class_values():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 0
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                classifier=True,
                multiple_outputs=True,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    assert np.all(gp._labels == np.array([-1, 0]))


# def test_multiclass():
#     from EvoDAG import EvoDAG
#     y = cl.copy()
#     ncl = np.unique(y).shape[0]
#     gp = EvoDAG(generations=np.inf,
#                 tournament_size=2,
#                 early_stopping_rounds=-1,
#                 seed=0,
#                 popsize=100).fit(X, y)
#     assert len(gp._multiclass_instances) == ncl
#     assert gp._multiclass


# def test_multiclass_decision_function():
#     from EvoDAG import EvoDAG
#     y = cl.copy()
#     gp = EvoDAG(generations=np.inf,
#                 tournament_size=2,
#                 early_stopping_rounds=-1,
#                 seed=0,
#                 popsize=100).fit(X, y, test_set=X)
#     d = gp.decision_function()
#     assert len(d) == np.unique(y).shape[0]


# def test_multiclass_predict():
#     from EvoDAG import RootGP
#     y = cl.copy()
#     y[y == 0] = 3
#     gp = RootGP(generations=np.inf,
#                 tournament_size=2,
#                 early_stopping_rounds=-1,
#                 seed=0,
#                 popsize=100).fit(X, y, test_set=X)
#     d = gp.predict()
#     assert np.unique(d).shape[0] == np.unique(y).shape[0]
#     assert np.all(np.unique(d) == np.unique(y))


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
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                multiple_outputs=True,
                seed=0,
                popsize=100).fit(X, y, test_set=X)
    m = gp.model()
    hy = m.predict(X=X)
    print(np.unique(hy), np.unique(y))
    print(np.array([1, 2]))
    for k in np.unique(hy):
        assert k in [1, 2]
    # assert np.all(np.unique(hy) == np.array([1, 2]))


def test_height():
    from EvoDAG.node import Mul, NaiveBayesMN, NaiveBayes
    from EvoDAG import EvoDAG
    gp = EvoDAG(generations=1,
                seed=1,
                multiple_outputs=True,
                tournament_size=2,
                popsize=5)
    gp.X = X
    gp.nclasses(cl)
    gp.y = cl.copy()
    gp.create_population()
    print(NaiveBayes.nargs, NaiveBayesMN.nargs)
    print([(x, x.height) for x in gp.population.population])
    assert np.all([x.height == 0 for x in gp.population.population])
    args = [3, 4]
    f = gp._random_offspring(Mul, args)
    assert f.height == 1


def test_regression():
    from EvoDAG import RootGP
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
                multiple_outputs=True,
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
    gp._bagging_fitness.fitness_vs(model._hist[-1])
    print(rse(y, yh), model._hist[-1].fitness_vs)
    assert_almost_equals(rse(y, yh),
                         -model._hist[-1].fitness_vs)


def test_RSE_avg_zero():
    from EvoDAG.bagging_fitness import BaggingFitness
    from EvoDAG import EvoDAG

    class B(BaggingFitness):
        def __init__(self, **kw):
            super(B, self).__init__(**kw)
            self._base._times = 0

        def set_regression_mask(self, v):
            base = self._base
            mask = np.ones(v.size())
            if base._times == 0:
                mask[10:12] = 0
            else:
                mask[10:13] = 0
            base._mask = SparseArray.fromlist(mask)
            base._times += 1

    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    y[10:12] = 0
    gp = EvoDAG(classifier=False,
                popsize=10,
                generations=2)
    gp._bagging_fitness = B(base=gp)
    gp.X = [SparseArray.fromlist(x)]
    gp.y = y
    print(gp._times)
    assert gp._times == 2
    gp.create_population()
    while not gp.stopping_criteria():
        a = gp.random_offspring()
        gp.replace(a)


def test_population_as_parameter():
    class Mock(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('Mock')
    from EvoDAG import RootGP
    mock = Mock
    gp = RootGP(generations=np.inf,
                population_class=mock,
                tournament_size=2,
                multiple_outputs=True,
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
             classifier=False,
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
             classifier=False,
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
    for i in range(20):
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
             classifier=False,
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
    for i in range(20):
        n = gp.random_offspring()
        if n.fitness_vs > es.fitness_vs:
            break
    gp.population.add(n)
    print(gp._unfeasible_counter)
    assert gp._unfeasible_counter == 0


def test_unfeasible_counter_fit():
    from EvoDAG import RGP

    class RGP2(RGP):
        def replace(self, a):
            self.population.replace(a)
            self._unfeasible_counter = 100

        def add(self, a):
            self.population.add(a)
            self._unfeasible_counter = 100

    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RGP2(generations=10,
              tournament_size=2,
              early_stopping_rounds=1,
              classifier=False,
              seed=0,
              popsize=3)
    [gp.unfeasible_offspring() for _ in range(100)]
    assert gp._unfeasible_counter == 100
    gp.fit(X, y)
    print('Hist', len(gp.population.hist), gp.population.hist)
    assert len(gp.population.hist) <= 4


def test_two_instances():
    from EvoDAG import EvoDAG
    y = cl.copy()
    y[:-2] = -1
    y[-2:] = 1
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                classifier=False,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X, y, test_set=X)
    assert gp


def test_time_limit():
    from EvoDAG import RGP
    import time
    y = cl.copy()
    t = time.time()
    gp = RGP(generations=np.inf,
             tournament_size=2,
             early_stopping_rounds=100,
             multiple_outputs=True,
             time_limit=0.9,
             seed=0,
             popsize=10000).fit(X, y, test_set=X)
    t2 = time.time()
    print(t2 - t)
    assert t2 - t < 1
    assert gp._time_limit == 0.9


def test_transform_to_mo():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=10000)
    gp.nclasses(y)
    k = np.unique(y)
    y = gp._bagging_fitness.transform_to_mo(y)
    assert k.shape[0] == y.shape[1]


def test_multiple_outputs():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=10000)
    gp.X = X
    gp.nclasses(y)
    gp.y = y
    gp.create_population()
    assert len(gp.y) == 3


def test_multiple_outputs2():
    from EvoDAG import EvoDAG
    from EvoDAG.model import Model
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=10000).fit(X, y, test_set=X)
    m = gp.model()
    assert isinstance(m, Model)
    assert len(gp.y) == 3


def test_add_repeated_args():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    y = cl.copy()
    for ff in [Add, Min, Max]:
        ff.nargs = 10
        gp = EvoDAG(generations=np.inf,
                    tournament_size=2,
                    early_stopping_rounds=100,
                    time_limit=0.9,
                    # multiple_outputs=True,
                    classifier=False,
                    all_inputs=True,
                    function_set=[ff],
                    pr_variable=1,
                    seed=0,
                    popsize=10000)
        gp.X = X
        # gp.nclasses(y)
        gp.y = y
        gp.create_population()
        print(gp.population.population)
        node = gp.random_offspring()
        print(node, node._variable, X.shape)
        assert len(node._variable) <= X.shape[1]
        ff.nargs = 2


def test_classification_mo():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=10,
                time_limit=0.9,
                multiple_outputs=True,
                all_inputs=True,
                seed=0,
                popsize=10000)
    gp.X = X
    gp.nclasses(y)
    y = gp._bagging_fitness.transform_to_mo(y)
    gp.y = [SparseArray.fromlist(x) for x in y.T]
    assert isinstance(gp._mask, list)
    gp.create_population()


def test_classification_mo2():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf, tournament_size=2,
                early_stopping_rounds=10, time_limit=0.9,
                multiple_outputs=True, all_inputs=True,
                seed=0, popsize=10000)
    gp.X = X
    gp.nclasses(y)
    y = gp._bagging_fitness.transform_to_mo(y)
    y = [SparseArray.fromlist(x) for x in y.T]
    gp = EvoDAG(generations=np.inf, tournament_size=2,
                early_stopping_rounds=10, time_limit=0.9,
                multiple_outputs=True, all_inputs=True, seed=0,
                popsize=10000).fit(X, y)
    m = gp.model()
    print([(x, x._variable, x.height) for x in m._hist])
    # assert False
    assert len(m.decision_function(gp.X)) == 3


def test_function_selection():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=100).fit(X, y, test_set=X)
    assert gp._function_selection_ins
    for i in range(gp._function_selection_ins.nfunctions):
        assert gp._function_selection_ins.avg_fitness(i) != 0


def test_multiple_outputs_mask():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                time_limit=0.9,
                tr_fraction=0.8,
                multiple_outputs=True,
                seed=0,
                popsize=100)
    gp.X = X[:-1]
    gp.nclasses(y[:-1])
    gp.y = y[:-1]
    assert gp._mask_vs.sum() == 27


def test_multiple_outputs_BER_vs():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    from EvoDAG.utils import BER
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=100)
    gp.X = X[:-1]
    gp.nclasses(y[:-1])
    gp.y = y[:-1]
    gp.create_population()
    a = gp.random_offspring()
    hy = np.array(SparseArray.argmax(a.hy).full_array())
    mask = np.array(gp._mask_vs.full_array()).astype(np.bool)
    assert_almost_equals(-a.fitness_vs * 100, BER(y[:-1][mask], hy[mask]))


def test_multiple_outputs_BER_ts():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    from EvoDAG.utils import BER
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=100)
    gp.X = X[:-1]
    gp.nclasses(y[:-1])
    gp.y = y[:-1]
    gp.create_population()
    a = gp.random_offspring()
    hys = SparseArray.argmax(a.hy)
    hy = np.array(hys.full_array())
    print(((hys - gp._y_klass).sign().fabs() * gp._mask_ts).sum())
    mask = np.array(gp._mask_ts.full_array()).astype(np.bool)
    assert_almost_equals(-a.fitness * 100, BER(y[:-1][mask], hy[mask]))


def test_multiple_outputs_error_rate_ts():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                fitness_function='ER',
                seed=0,
                popsize=100)
    gp.X = X[:-1]
    gp.nclasses(y[:-1])
    gp.y = y[:-1]
    gp.create_population()
    a = gp.random_offspring()
    hys = SparseArray.argmax(a.hy)
    hy = np.array(hys.full_array())
    # print(((hys - gp._y_klass).sign().fabs() * gp._mask_ts).sum())
    mask = np.array(gp._mask_ts.full_array()).astype(np.bool)
    # print((y[:-1][mask] != hy[mask]).mean())
    print(-a.fitness, (y[:-1][mask] != hy[mask]).mean())
    assert_almost_equals(-a.fitness, (y[:-1][mask] != hy[mask]).mean())


def test_multiple_outputs_ER_vs():
    from EvoDAG import EvoDAG
    from EvoDAG.node import Add, Min, Max
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                function_set=[Add, Min, Max],
                early_stopping_rounds=100,
                fitness_function='ER',
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=100)
    gp.X = X[:-1]
    gp.nclasses(y[:-1])
    gp.y = y[:-1]
    gp.create_population()
    a = gp.random_offspring()
    hy = np.array(SparseArray.argmax(a.hy).full_array())
    mask = np.array(gp._mask_vs.full_array()).astype(np.bool)
    assert_almost_equals(-a.fitness_vs, (y[:-1][mask] != hy[mask]).mean())


