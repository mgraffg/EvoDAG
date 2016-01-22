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
from pymock import use_pymock, override, returns, replay
from RGP.node import Variable
from RGP.sparse_array import SparseArray


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
    from RGP.sparse_array import SparseArray
    from RGP import RootGP
    gp = RootGP(generations=1)
    gp.X = X
    assert gp.nvar == 4
    print gp.X
    assert isinstance(gp.X[0], Variable)
    assert isinstance(gp.X[0].hy, SparseArray)
    gp.Xtest = X
    assert gp.X[0].hy.SSE(gp.X[0].hy_test) == 0


@use_pymock
def test_create_population():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(gp.popsize):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.create_population()
    assert_almost_equals(gp.population.popsize, gp.popsize)
    a = map(lambda (x, y): x == y, zip(gp.population.population,
                                       gp.population._hist))
    assert np.all(a)
    l = map(lambda x: x.variable, gp.population.population)
    assert np.all(map(lambda (x, y): x == y, enumerate(l)))


@use_pymock
def test_best_so_far():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(gp.popsize):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.create_population()
    p = gp.population.population
    index = np.argsort(map(lambda x: x.fitness, p))[-1]
    print p[index].fitness, gp.population.bsf.fitness
    assert gp.population.bsf == p[index]


@use_pymock
def test_early_stopping():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(gp.popsize):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.create_population()
    p = gp.population.population
    fit = np.array(map(lambda x: x.fitness_vs, p))
    best = fit.max()
    index = np.where(best == fit)[0][0]
    assert gp.population.estopping == p[index]


@use_pymock
def test_variable():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    Xtest = map(lambda x: x, X)
    Xtest[0] = Xtest[0] + np.inf
    gp.Xtest = Xtest
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(2):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    var = gp.random_leaf()
    assert var.isfinite()
    assert var.hy.isfinite()


@use_pymock
def test_random_leaf():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4, tr_fraction=1)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    np.random.randint(gp.nvar)
    returns(0)
    replay()
    mask = gp._mask.tonparray().astype(np.bool)
    weight = np.linalg.lstsq(X[mask, 0][:, np.newaxis], y[mask])[0][0]
    var = gp.random_leaf()
    assert isinstance(var, Variable)
    print weight, var.weight
    assert_almost_equals(weight, var.weight)


@use_pymock
def test_random_leaf_inf():
    from RGP import RootGP
    gp = RootGP(generations=1, classifier=False, popsize=4, tr_fraction=1)
    Xc = map(lambda x: x, X)
    Xc[0] = Xc[0] + np.inf
    gp.X = Xc
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(2):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.random_leaf()


def test_classification_y():
    from RGP import RootGP
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
    from RGP import RootGP
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
    from RGP import RootGP
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
    from RGP import RootGP
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
    from RGP.node import Add
    from RGP import RootGP
    from SimpleGP import Classification
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
    b = Classification.BER(y[m], hy.tonparray()[m])
    gp.fitness_vs(a)
    print b, a.fitness_vs * 100
    assert_almost_equals(b, -a.fitness_vs * 100)
    # assert False


@use_pymock
def test_tournament():
    from RGP import RootGP
    gp = RootGP(generations=1,
                tournament_size=4,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(gp.popsize):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.create_population()
    j = gp.population.tournament()
    index = np.argsort(map(lambda x: x.fitness,
                           gp.population.population))[-1]
    assert j == index


@use_pymock
def test_tournament_negative():
    from RGP import RootGP
    gp = RootGP(generations=1,
                tournament_size=4,
                popsize=4)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    override(np.random, 'randint')
    for i in range(gp.popsize):
        np.random.randint(gp.nvar)
        returns(i)
    replay()
    gp.create_population()
    j = gp.population.tournament(negative=True)
    index = np.argsort(map(lambda x: x.fitness,
                           gp.population.population))[0]
    assert j == index


@use_pymock
def test_random_offspring():
    from RGP import RootGP
    from RGP.node import Add
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
    override(np.random, 'randint')
    np.random.randint(len(gp.function_set))
    returns(0)
    replay()
    a = gp.random_offspring()
    assert isinstance(a, Add)
    assert np.isfinite(a.fitness)


def test_replace_individual():
    from RGP import RootGP
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
    assert np.any(map(lambda x: x == a, gp.population.population))
    assert a.position == 10


def test_X_sparse():
    from RGP import RootGP
    from RGP.sparse_array import SparseArray
    gp = RootGP(generations=1,
                tournament_size=2,
                popsize=10)
    X1 = map(SparseArray.fromlist, X.T)
    gp.X = X1


def test_fit_stopping_criteria_gens():
    from RGP import RootGP
    gp = RootGP(generations=2,
                early_stopping_rounds=None,
                tournament_size=2,
                seed=1,
                popsize=10)
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
    from RGP import RootGP
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=10,
                seed=0,
                popsize=10)
    gp.X = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    gp.create_population()
    print len(gp.population.hist) - gp.population.estopping.position
    while not gp.stopping_criteria():
        a = gp.random_offspring()
        gp.population.replace(a)
    assert (len(gp.population.hist) - gp.population.estopping.position) > 10


def test_fit():
    from RGP import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X, y, test_set=X)
    assert np.isfinite(gp.population.estopping.fitness)
    assert np.isfinite(gp.population.estopping.fitness_vs)
    assert gp.population.estopping.hy.isfinite()
    assert len(gp.population.hist) > 10


def test_logging():
    from RGP import RootGP
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
    from RGP import RootGP
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
    from RGP import RootGP
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
    assert gp.decision_function().SSE(es.hy_test) == 0
    hy_test = es.hy_test
    assert gp.decision_function(X=X[-10:]).SSE(hy_test) == 0
    hy = gp.decision_function(X=X[-10:])
    assert gp.predict(X=X[-10:]).SSE(hy)


def test_trace():
    from RGP import RootGP
    from RGP.node import Add
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    Add.nargs = 4
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                function_set=[Add],
                early_stopping_rounds=-1,
                seed=0,
                popsize=10)
    gp.X = X[:-10]
    gp.Xtest = X[-10:]
    gp.y = y[:-10]
    gp.create_population()
    a = gp.random_offspring()
    gp.population.replace(a)
    print a.position, a.variable, a._weight, gp.population.hist[0].variable
    s = gp.trace(a)
    assert len(s) == 5


def test_class_values():
    from RGP import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 0
    y[~mask] = -1
    try:
        RootGP(generations=np.inf,
               tournament_size=2,
               early_stopping_rounds=-1,
               seed=0,
               popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
        assert False
    except RuntimeError:
        pass
