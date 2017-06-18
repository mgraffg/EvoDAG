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


from test_root import cl
from test_root import X
from SparseArray import SparseArray
import numpy as np


def test_pickle_model():
    from EvoDAG import RootGP
    import pickle
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                multiple_outputs=True,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    m = gp.model()
    hy = gp.decision_function(X=X[-10:])
    m1 = pickle.loads(pickle.dumps(m))
    hy1 = m1.decision_function(X=X[-10:])
    for a, b in zip(hy, hy1):
        assert a.SSE(b) == 0


def test_model_hist():
    from EvoDAG import EvoDAG
    from EvoDAG.base import Model
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                multiple_outputs=True,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=1,
                popsize=30).fit(X[:-10], y[:-10], test_set=X[-10:])
    hist = gp.population.hist
    trace = gp.trace(gp.population.estopping)
    a = hist[trace[-1]].variable
    if not isinstance(a, list):
        a = [a]
    m = Model(trace, hist)
    b = m._hist[-1].variable
    if not isinstance(b, list):
        b = [b]
    print([(x, x.height) for x in m._hist])
    print((m._map, a, b))
    for v1, v2 in zip(a, b):
        if v1 not in m._map:
            assert v1 == v2
        else:
            assert m._map[v1] == v2


def test_ensemble():
    from EvoDAG import RootGP
    from EvoDAG.model import Ensemble
    from EvoDAG.node import Add
    y = cl.copy()
    gps = [RootGP(generations=np.inf,
                  tournament_size=2,
                  early_stopping_rounds=-1,
                  seed=seed,
                  multiple_outputs=True,
                  popsize=10).fit(X[:-10],
                                  y[:-10],
                                  test_set=X)
           for seed in range(2, 5)]
    ens = Ensemble([gp.model() for gp in gps])
    res = [gp.decision_function() for gp in gps]
    res = [Add.cumsum([x[j] for x in res]) for j in range(3)]
    res = [x * (1 / 3.) for x in res]
    r2 = ens.decision_function(None)
    for a, b in zip(res, r2):
        print(a.SSE(b), a.data, b.data, b.full_array())
        assert a.SSE(b) == 0


def test_ensemble_model():
    from EvoDAG import RootGP
    from EvoDAG.model import Ensemble
    from EvoDAG.node import Add
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gps = [RootGP(generations=np.inf,
                  tournament_size=2,
                  early_stopping_rounds=-1,
                  classifier=False,
                  seed=seed,
                  popsize=10).fit(X[:-10],
                                  y[:-10],
                                  test_set=X)
           for seed in range(3)]
    ens = Ensemble([gp.model() for gp in gps])
    res = [gp.decision_function() for gp in gps]
    res = np.median([x.full_array() for x in res], axis=0)
    res = SparseArray.fromlist(res)
    print(res)
    # res = Add.cumsum(res) * (1 / 3.)
    r2 = ens.decision_function(None)
    print(res.full_array()[:10], r2.full_array()[:10])
    print(res.SSE(r2))
    assert res.SSE(r2) == 0


def test_regression():
    from EvoDAG import RootGP
    from EvoDAG.model import Ensemble
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    gps = [RootGP(classifier=False,
                  seed=seed,
                  popsize=10,
                  generations=2).fit([SparseArray.fromlist(x)], y,
                                     test_set=[SparseArray.fromlist(x)])
           for seed in range(3)]
    ens = Ensemble([gp.model() for gp in gps])
    hy = np.median([gp.predict() for gp in gps], axis=0)
    hy1 = ens.predict(X=[SparseArray.fromlist(x)])
    print(hy, hy1)
    assert np.all(hy == hy1)


def test_model_graphviz():
    from EvoDAG import RootGP
    from EvoDAG.node import Function
    import tempfile
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=3,
                tournament_size=2,
                early_stopping_rounds=-1,
                classifier=False,
                pr_variable=1,
                seed=0,
                popsize=10).fit(X, y)
    m = gp.model()
    print(m._hist)
    print(m._hist[-1].position, m._hist[-1]._variable)
    with tempfile.TemporaryFile('w+') as io:
        m.graphviz(io)
        io.seek(0)
        l = io.readlines()
    cnt = len(m._hist)
    for k in m._hist:
        if isinstance(k, Function):
            v = k._variable
            if isinstance(v, list):
                cnt += len(v)
            else:
                cnt += 1
    print("".join(l))
    print(cnt, len(l))
    assert 2 + cnt == len(l)


def test_random_selection():
    from EvoDAG import RootGP
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    RootGP(generations=np.inf,
           tournament_size=1,
           early_stopping_rounds=-1,
           classifier=False,
           seed=0,
           popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])


def test_multiple_outputs_decision_function():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                multiple_outputs=True,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    m = gp.model()
    assert m.multiple_outputs
    hy = m.decision_function(X)
    assert len(hy) == 3
    for i in hy:
        assert i.isfinite()


def test_multiple_outputs_predict():
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                multiple_outputs=True,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    m = gp.model()
    assert m.multiple_outputs
    hy = m.predict(X)
    u = np.unique(y)
    for i in np.unique(hy):
        assert i in u


def test_multiple_outputs_ensemble():
    from EvoDAG import EvoDAG
    from EvoDAG.model import Ensemble
    y = cl.copy()
    gp = [EvoDAG(generations=np.inf,
                 tournament_size=2,
                 multiple_outputs=True,
                 early_stopping_rounds=-1,
                 seed=x,
                 popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
          for x in range(2)]
    ens = Ensemble([x.model() for x in gp])
    assert ens.multiple_outputs
    hy = ens.predict(X)
    u = np.unique(y)
    for i in np.unique(hy):
        assert i in u
    
