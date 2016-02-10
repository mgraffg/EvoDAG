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
import numpy as np


def test_pickle_model():
    from RGP import RootGP
    import pickle
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=0,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    m = gp.model()
    hy = gp.decision_function(X=X[-10:])
    m1 = pickle.loads(pickle.dumps(m))
    hy1 = m1.decision_function(X=X[-10:])
    assert hy.SSE(hy1) == 0


def test_model_hist():
    from RGP import RootGP
    from RGP.base import Model
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp = RootGP(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=-1,
                seed=1,
                popsize=10).fit(X[:-10], y[:-10], test_set=X[-10:])
    hist = gp.population.hist
    trace = gp.trace(gp.population.estopping)
    a = hist[trace[-1]].variable
    m = Model(trace, hist)
    print m._map, a, m._hist[-1].variable
    for v1, v2 in zip(a, m._hist[-1].variable):
        assert m._map[v1] == v2


def test_ensemble():
    from RGP import RootGP
    from RGP.model import Ensemble
    from RGP.node import Add
    y = cl.copy()
    gps = map(lambda seed: RootGP(generations=np.inf,
                                  tournament_size=2,
                                  early_stopping_rounds=-1,
                                  seed=seed,
                                  popsize=10).fit(X[:-10],
                                                  y[:-10],
                                                  test_set=X),
              range(3))
    ens = Ensemble(map(lambda gp: gp.model(), gps))
    res = map(lambda gp: gp.decision_function(), gps)
    res = map(lambda j: Add.cumsum(map(lambda x: x[j], res)), range(3))
    res = map(lambda x: x / 3., res)
    r2 = ens.decision_function(None)
    for a, b in zip(res, r2):
        assert a.SSE(b) == 0
    r2 = ens.predict(None).tonparray()
    assert np.all(np.unique(r2) == np.unique(y))


def test_ensemble_model():
    from RGP import RootGP
    from RGP.model import Ensemble
    from RGP.node import Add
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gps = map(lambda seed: RootGP(generations=np.inf,
                                  tournament_size=2,
                                  early_stopping_rounds=-1,
                                  seed=seed,
                                  popsize=10).fit(X[:-10],
                                                  y[:-10],
                                                  test_set=X),
              range(3))
    ens = Ensemble(map(lambda gp: gp.model(), gps))
    res = map(lambda gp: gp.decision_function(), gps)
    res = Add.cumsum(res) / 3
    r2 = ens.decision_function(None)
    assert res.SSE(r2) == 0
    a = ens.predict(None)
    assert r2.sign().SSE(a) == 0

