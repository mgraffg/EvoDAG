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
from test_command_line import default_nargs
from nose.tools import assert_almost_equals


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
    gp = EvoDAG.init(generations=np.inf,
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
    res = [gp.model().decision_function(X) for gp in gps]
    res = np.array([np.median([x[j].finite().full_array() for x in res], axis=0) for j in range(3)]).T
    r2 = ens.decision_function(X)
    print(res, r2)
    assert np.fabs(res - r2).sum() == 0


def test_ensemble_model():
    from EvoDAG.model import EvoDAG as evodag
    from EvoDAG.model import Ensemble
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gps = [evodag(generations=np.inf,
                  tournament_size=2,
                  early_stopping_rounds=-1,
                  classifier=False,
                  seed=seed,
                  popsize=10).fit(X[:-10],
                                  y[:-10],
                                  test_set=X)
           for seed in range(3)]
    ens = Ensemble([gp.model for gp in gps])
    res = [gp.decision_function(X) for gp in gps]
    res = np.median([x.finite().full_array() for x in res], axis=0)
    r2 = ens.decision_function(X)
    print(res, r2)
    assert np.fabs(res - r2).mean() == 0


def test_regression():
    from EvoDAG import EvoDAG as evodag
    from EvoDAG.model import Ensemble
    x = np.linspace(-1, 1, 100)
    y = 4.3*x**2 + 3.2 * x - 3.2
    gps = [evodag.init(classifier=False,
                       seed=seed,
                       popsize=10,
                       generations=2).fit([SparseArray.fromlist(x)], y,
                                          test_set=[SparseArray.fromlist(x)])
           for seed in range(3)]
    model = [_.model() for _ in gps]
    ens = Ensemble(model)
    X = [SparseArray.fromlist(x)]
    hy = np.median([m.decision_function(X).finite().full_array() for m in model], axis=0)
    hy1 = ens.predict(X)
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


def test_init():
    from EvoDAG.model import Ensemble
    m = Ensemble.init(n_estimators=4, n_jobs=4, seed=10, early_stopping_rounds=100).fit(X, cl)
    hy = m.predict(X)
    assert (cl == hy).mean() > 0.9
    default_nargs()


def test_init_regression():
    from EvoDAG.model import Ensemble
    m = Ensemble.init(n_estimators=4, n_jobs=4, seed=10,
                      classifier=False, early_stopping_rounds=100).fit(X, cl)
    hy = m.predict(X)
    assert np.unique(hy).shape[0] > 3
    default_nargs()


def test_init_e():
    from EvoDAG.model import EvoDAGE
    m = EvoDAGE(n_estimators=4, n_jobs=4, seed=10, early_stopping_rounds=100).fit(X, cl)
    hy = m.predict(X)
    assert (cl == hy).mean() > 0.9
    default_nargs()


def test_init2():
    from EvoDAG.model import Ensemble
    m = Ensemble.init(n_estimators=4, n_jobs=1, seed=10, early_stopping_rounds=100).fit(X, cl)
    hy = m.predict(X)
    print((cl == hy).mean(), cl, hy)
    assert (cl == hy).mean() > 0.9
    default_nargs()


def test_init_evodag():
    from EvoDAG.model import EvoDAG
    m = EvoDAG().fit(X, cl)
    hy = m.predict(X)
    print((cl == hy).mean(), cl, hy)
    assert (cl == hy).mean() > 0.9
    default_nargs()


def test_init_time_limit():
    from EvoDAG.model import EvoDAGE
    import time
    local = time.time()
    EvoDAGE(n_estimators=30, n_jobs=2, time_limit=4).fit(X, cl)
    default_nargs()
    t = time.time() - local
    print(t)
    assert t <= 6


def test_predict_proba():
    from EvoDAG.model import EvoDAGE
    m = EvoDAGE(n_estimators=3, n_jobs=2, time_limit=4).fit(X, cl)
    pr = m.predict_proba(X)
    print(pr)
    default_nargs()
    print(pr.min(), pr.max())
    assert pr.min() >= 0 and pr.max() <= 1


def test_raw_decision_function():
    from EvoDAG.model import EvoDAGE
    m = EvoDAGE(n_estimators=3, n_jobs=2, time_limit=4)
    m.fit(X, cl)
    pr = m.raw_decision_function(X)
    default_nargs()
    print(pr.shape)
    assert pr.shape[1] == np.unique(cl).shape[0] * len(m._m.models)


def test_normalize_naive():
    from EvoDAG import EvoDAG as evodag
    m = evodag.init(time_limit=4)
    m.fit(X, cl)
    hy = [x for x in m.population.hist[0].hy]
    naive = m.model(v=m.population.hist[0])
    df = np.array([x.full_array() for x in naive.decision_function(X)]).T
    hy = np.array([x.full_array() for x in hy]).T
    hy = np.exp(hy - np.atleast_2d(np.log(np.sum(np.exp(hy), axis=1))).T) * 2 - 1
    print(hy - df)
    for a, b in zip(hy, df):
        [assert_almost_equals(v, w) for v, w in zip(a, b)]
    default_nargs()


def test_normalize_naiveMN():
    from EvoDAG import EvoDAG as evodag
    m = evodag.init(time_limit=4)
    m.fit(X, cl)
    hy = [x for x in m.population.hist[1].hy]
    naive = m.model(v=m.population.hist[1])
    df = np.array([x.full_array() for x in naive.decision_function(X)]).T
    hy = np.array([x.full_array() for x in hy]).T
    hy = hy / np.atleast_2d(hy.sum(axis=1)).T * 2 - 1
    for a, b in zip(hy, df):
        [assert_almost_equals(v, w) for v, w in zip(a, b)]
    default_nargs()


def test_normalize_Centroid():
    from EvoDAG import EvoDAG as evodag
    m = evodag.init(time_limit=4)
    m.fit(X, cl)
    if not isinstance(m.population.hist[2].hy, list):
        return
    hy = [x for x in m.population.hist[2].hy]
    naive = m.model(v=m.population.hist[2])
    df = np.array([x.full_array() for x in naive.decision_function(X)]).T
    hy = np.array([x.full_array() for x in hy]).T
    hy = np.exp(hy) * 2 - 1
    print(df, hy)
    for a, b in zip(hy, df):
        [assert_almost_equals(v, w) for v, w in zip(a, b)]
    default_nargs()


def test_probability_calibration():

    class C(object):
        def __init__(self):
            pass

        def fit(self, X, y):
            self._X = X
            return self

        def predict_proba(self, X):
            return np.array([[0, 1], [0.5, 0.5], [1, 0]])

    from EvoDAG import EvoDAG as evodag
    m = evodag.init(time_limit=4, early_stopping_rounds=10,
                    probability_calibration=C).fit(X, cl)
    model = m.model()
    hy = model.predict_proba(X)
    pr = np.array([[0, 1], [0.5, 0.5], [1, 0]])
    assert np.fabs(hy - pr).sum() == 0
    default_nargs()


def test_probability_calibration_ensemble():

    class C(object):
        def __init__(self):
            pass

        def fit(self, X, y):
            self._X = X
            return self

        def predict_proba(self, X):
            return np.array([[0, 1], [0.5, 0.5], [1, 0]])

    from EvoDAG.model import EvoDAGE as evodag
    model = evodag(time_limit=4, early_stopping_rounds=10,
                   probability_calibration=C).fit(X, cl)
    for m in model.models:
        assert isinstance(m._probability_calibration, C)
    assert model.probability_calibration
    hy = model.predict_proba(X)
    pr = np.array([[0, 1], [0.5, 0.5], [1, 0]])
    assert np.fabs(hy - pr).sum() == 0
    default_nargs()


def test_regression_multiple_outputs():
    from EvoDAG.model import EvoDAGE
    from SparseArray import SparseArray
    y = [SparseArray.fromlist(cl), SparseArray.fromlist(cl*-1), SparseArray.fromlist(cl*-1 + 0.5)]
    m = EvoDAGE(time_limit=4, multiple_outputs=True, classifier=False).fit(X, y)
    assert m
