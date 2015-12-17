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
from RGP.node import Add


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


def test_create_population():
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    gp.create_population()


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


def create_problem_node(nargs=4):
    from RGP import RootGP
    gp = RootGP(generations=1, popsize=4)
    gp.X = X
    gp.Xtest = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    return gp, map(lambda x: gp.X[x], range(nargs))


def test_node_add():
    gp, args = create_problem_node()
    coef = gp.compute_weight(map(lambda x: x.hy, args))
    n = Add(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    a = map(lambda (a, b): a.hy * b, zip(args, coef))
    r = n.cumsum(a)
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_mul():
    from RGP.node import Mul
    gp, args = create_problem_node()
    r = Mul.cumprod(map(lambda x: x.hy, args))
    coef = gp.compute_weight([r])[0]
    n = Mul(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_div():
    from RGP.node import Div
    gp, args = create_problem_node(nargs=2)
    a, b = args
    r = a.hy / b.hy
    coef = gp.compute_weight([r])[0]
    n = Div(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_fabs():
    from RGP.node import Fabs
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.fabs()
    coef = gp.compute_weight([r])[0]
    n = Fabs(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_exp():
    from RGP.node import Exp
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.exp()
    coef = gp.compute_weight([r])[0]
    n = Exp(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sqrt():
    from RGP.node import Sqrt
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sqrt()
    coef = gp.compute_weight([r])[0]
    n = Sqrt(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sin():
    from RGP.node import Sin
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sin()
    coef = gp.compute_weight([r])[0]
    n = Sin(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_cos():
    from RGP.node import Cos
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.cos()
    coef = gp.compute_weight([r])[0]
    n = Cos(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_ln():
    from RGP.node import Ln
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.ln()
    coef = gp.compute_weight([r])[0]
    n = Ln(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sq():
    from RGP.node import Sq
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sq()
    coef = gp.compute_weight([r])[0]
    n = Sq(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sigmoid():
    from RGP.node import Sigmoid
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sigmoid()
    coef = gp.compute_weight([r])[0]
    n = Sigmoid(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_if():
    from RGP.node import If
    gp, args = create_problem_node(nargs=3)
    r = args[0].hy.if_func(args[1].hy, args[2].hy)
    coef = gp.compute_weight([r])[0]
    n = If(range(len(args)), ytr=gp._ytr, mask=gp._mask)
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0



