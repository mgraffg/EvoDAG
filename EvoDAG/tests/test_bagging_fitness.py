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

from test_root import X, cl
import numpy as np
from SparseArray import SparseArray
from nose.tools import assert_almost_equals


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


def test_macro_f1():
    from EvoDAG.cython_utils import F1Score
    from EvoDAG import EvoDAG
    y = cl.copy()
    gp = EvoDAG(generations=np.inf,
                tournament_size=2,
                early_stopping_rounds=100,
                time_limit=0.9,
                multiple_outputs=True,
                seed=0,
                popsize=100)
    gp.y = y
    gp.X = X
    gp.create_population()
    off = gp.random_offspring()
    hy = SparseArray.argmax(off.hy)
    # print(gp._mask_ts.index, hy.full_array(), gp._y_klass.full_array(), gp._bagging_fitness.nclasses)
    index = np.array(gp._mask_ts.index)
    y = np.array(gp._y_klass.full_array())[index]
    hy = np.array(hy.full_array())[index]
    nclasses = gp._bagging_fitness.nclasses
    precision = np.array([(y[hy == k] == k).mean() for k in range(nclasses)])
    recall = np.array([(hy[y == k] == k).mean() for k in range(nclasses)])
    print(precision, recall)
    f1 = F1Score(nclasses)
    mf1, mf1_v = f1.macroF1(gp._y_klass, SparseArray.argmax(off.hy), gp._mask_ts.index)
    for x, y in zip(precision, f1.precision):
        if not np.isfinite(x):
            continue
        assert_almost_equals(x, y)
    for x, y in zip(recall, f1.recall):
        if not np.isfinite(x):
            continue
        assert_almost_equals(x, y)
    _ = (2 * precision * recall) / (precision + recall)
    assert_almost_equals(np.mean(_), mf1)
    print(f1.precision, f1.recall, mf1, mf1_v)
    gp._fitness_function = 'F1'
    gp._bagging_fitness.set_fitness(off)
    assert_almost_equals(off.fitness, mf1 - 1)
    assert_almost_equals(off.fitness_vs, mf1_v - 1)
    assert mf1_v != mf1
