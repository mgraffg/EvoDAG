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

from test_root import X, cl
from RGP.node import Add


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


def test_nargs_function():
    from RGP.node import Mul
    Add.nargs = 4
    assert Add.nargs == 4
    assert Mul.nargs == 2


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
