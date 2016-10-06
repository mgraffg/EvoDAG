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
from EvoDAG.node import Add
import numpy as np


def create_problem_node(nargs=4, seed=0):
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=4, seed=seed)
    gp.X = X
    gp.Xtest = X
    y = cl.copy()
    mask = y == 0
    y[mask] = 1
    y[~mask] = -1
    gp.y = y
    return gp, [gp.X[x] for x in range(nargs)]


def test_nargs_function():
    from EvoDAG.node import Mul
    Add.nargs = 4
    assert Add.nargs == 4
    assert Mul.nargs == 2


def test_node_pickle():
    import pickle
    import tempfile
    gp, args = create_problem_node()
    n = Add(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    n.position = 10
    assert n.eval(args)
    with tempfile.TemporaryFile('w+b') as io:
        pickle.dump(n, io)
        io.seek(0)
        n1 = pickle.load(io)
        assert n1._mask.SSE(n._mask) == 0


def test_node_tostore():
    gp, args = create_problem_node(nargs=4)
    Add.nargs = 4
    n = Add(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    n.position = 10
    assert n.eval(args)
    n1 = n.tostore()
    assert n1.nargs == n.nargs
    assert n1.position == n.position
    assert np.all(n1.weight == n.weight)
    assert n1.hy is None


def test_node_add():
    gp, args = create_problem_node()
    n = Add(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    D = np.array([i.hy.tonparray() for i in args]).T
    coef = n.compute_weight([x.hy for x in args])
    assert n.eval(args)
    # a = map(lambda (a, b): a.hy * b, zip(args, coef))
    a = [_a.hy * b for _a, b in zip(args, coef)]
    r = n.cumsum(a)
    print((D * coef).sum(axis=1)[:4], "*")
    print(n.hy.tonparray()[:4])
    print(r.tonparray()[:4])
    print(gp._mask.tonparray()[:4])
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_mul():
    from EvoDAG.node import Mul
    gp, args = create_problem_node()
    r = Mul.cumprod([x.hy for x in args])
    n = Mul(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_div():
    from EvoDAG.node import Div
    gp, args = create_problem_node(nargs=2)
    a, b = args
    r = a.hy / b.hy
    n = Div(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_fabs():
    from EvoDAG.node import Fabs
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.fabs()
    n = Fabs(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_exp():
    from EvoDAG.node import Exp
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.exp()
    n = Exp(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sqrt():
    from EvoDAG.node import Sqrt
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sqrt()
    n = Sqrt(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sin():
    from EvoDAG.node import Sin
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sin()
    n = Sin(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_cos():
    from EvoDAG.node import Cos
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.cos()
    n = Cos(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_ln():
    from EvoDAG.node import Ln
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.ln()
    n = Ln(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sq():
    from EvoDAG.node import Sq
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sq()
    n = Sq(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_sigmoid():
    from EvoDAG.node import Sigmoid
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.sigmoid()
    n = Sigmoid(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_if():
    from EvoDAG.node import If
    gp, args = create_problem_node(nargs=3)
    r = args[0].hy.if_func(args[1].hy, args[2].hy)
    n = If(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])[0]
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0


def test_node_min():
    from EvoDAG.node import Min
    gp, args = create_problem_node(nargs=3)
    r = args[0].hy.min(args[1].hy).min(args[2].hy)
    n = Min(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0
    assert n.weight == coef


def test_node_max():
    from EvoDAG.node import Max
    gp, args = create_problem_node(nargs=3)
    r = args[0].hy.max(args[1].hy).max(args[2].hy)
    n = Max(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    coef = n.compute_weight([r])
    assert n.eval(args)
    r = r * coef
    assert n.hy.SSE(r) == 0
    assert n.hy_test.SSE(r) == 0
    assert n.weight == coef
    

def test_node_symbol():
    from EvoDAG.node import Add, Mul, Div, Fabs,\
        Exp, Sqrt, Sin, Cos, Ln,\
        Sq, Sigmoid, If, Min, Max
    for f, s in zip([Add, Mul, Div, Fabs,
                     Exp, Sqrt, Sin, Cos, Ln,
                     Sq, Sigmoid, If, Min, Max],
                    ['+', '*', '/', 'fabs', 'exp',
                     'sqrt', 'sin', 'cos', 'ln',
                     'sq', 's', 'if', 'min', 'max']):
        assert f.symbol == s


def test_node_hash():
    from EvoDAG.node import Add, Mul
    add = Add([21, 3])
    sets = set()
    sets.add(add.signature())
    add2 = Add([21, 3])
    assert add2.signature() in sets
    assert Add([3, 21]).signature() in sets
    assert Mul([2, 3]).signature() == Mul([3, 2]).signature()


def test_variable_multiple_output():
    from EvoDAG.node import Variable
    gp, args = create_problem_node(nargs=4, seed=0)
    gp2, _ = create_problem_node(nargs=4, seed=1)
    n1 = Variable(0, ytr=gp._ytr, mask=gp._mask)
    n1.eval(args)
    print(n1.weight)
    n = Variable(0, ytr=[gp._ytr, gp._ytr],
                 mask=[gp._mask, gp2._mask])
    n.eval(args)
    assert n.weight[0] != n.weight[1]


def test_variable_multiple_output_isfinite():
    from EvoDAG.node import Variable
    gp, args = create_problem_node(nargs=4, seed=0)
    gp2, _ = create_problem_node(nargs=4, seed=1)
    n1 = Variable(0, ytr=gp._ytr, mask=gp._mask)
    n1.eval(args)
    print(n1.weight)
    n = Variable(0, ytr=[gp._ytr, gp._ytr],
                 mask=[gp._mask, gp2._mask])
    n.eval(args)
    n.isfinite()


def test_Add_multiple_output():
    from EvoDAG.node import Variable
    gp, args = create_problem_node(nargs=4, seed=0)
    gp2, _ = create_problem_node(nargs=4, seed=1)
    ytr = [gp._ytr, gp._ytr]
    mask = [gp._mask, gp2._mask]
    vars = [Variable(k, ytr=ytr, mask=mask) for k in range(len(args))]
    [x.eval(args) for x in vars]
    add = Add(range(len(vars)), ytr=ytr, mask=mask)
    assert add.eval(vars)
    gp, args = create_problem_node(nargs=4, seed=0)
    vars = [Variable(k, ytr=gp._ytr, mask=gp._mask) for k in range(len(args))]
    [x.eval(args) for x in vars]
    add2 = Add(range(len(vars)), ytr=gp._ytr, mask=gp._mask)
    assert add2.eval(vars)
    assert add2.hy.SSE(add.hy[0]) == 0
    assert add2.hy_test.SSE(add.hy_test[0]) == 0


def test_Add_multiple_output2():
    from EvoDAG.node import Variable
    gp, args = create_problem_node(nargs=4, seed=0)
    for i in args:
        i.hy_test = None
    gp2, _ = create_problem_node(nargs=4, seed=1)
    ytr = [gp._ytr, gp._ytr]
    mask = [gp._mask, gp2._mask]
    vars = [Variable(k, ytr=ytr, mask=mask) for k in range(len(args))]
    [x.eval(args) for x in vars]
    add = Add(range(len(vars)), ytr=ytr, mask=mask)
    assert add.eval(vars)
    vars = [Variable(k, ytr=gp._ytr, mask=gp._mask) for k in range(len(args))]
    [x.eval(args) for x in vars]
    add2 = Add(range(len(vars)), ytr=gp._ytr, mask=gp._mask)
    assert add2.eval(vars)
    assert add2.hy.SSE(add.hy[0]) == 0
    assert isinstance(add.weight, list) and len(add.weight) == 2
    assert isinstance(add2.weight, np.ndarray)


def test_one_multiple_output():
    from EvoDAG.node import Variable
    from EvoDAG.node import Fabs, Exp, Sqrt, Sin, Cos, Ln, Sq, Sigmoid
    for flag in [False, True]:
        gp, args = create_problem_node(nargs=4, seed=0)
        if flag:
            for i in args:
                i.hy_test = None
        gp2, _ = create_problem_node(nargs=4, seed=1)
        ytr = [gp._ytr, gp._ytr]
        mask = [gp._mask, gp2._mask]
        vars = [Variable(k, ytr=ytr, mask=mask) for k in range(len(args))]
        [x.eval(args) for x in vars]
        vars2 = [Variable(k, ytr=gp._ytr, mask=gp._mask) for k in range(len(args))]
        [x.eval(args) for x in vars2]
        for FF in [Fabs, Exp, Sqrt, Sin, Cos, Ln, Sq, Sigmoid]:
            ff = FF(0, ytr=ytr, mask=mask)
            ff.eval(vars)
            ff2 = FF(0, ytr=gp._ytr, mask=gp._mask)
            ff2.eval(vars2)
            print(ff.hy)
            if ff.hy is None:
                continue
            assert isinstance(ff.weight, list) and len(ff.weight) == 2
            assert isinstance(ff2.weight, float)
            hy = ff.hy[0]
            if hy.isfinite():
                print('*', FF)
                assert hy.SSE(ff2.hy) == 0


def test_functions_w_multiple_output():
    from EvoDAG.node import Variable, Mul, Div, If, Min, Max
    for ff in [Mul, Div, If, Min, Max]:
        for flag in [False, True]:
            gp, args = create_problem_node(nargs=4, seed=0)
            if flag:
                for i in args:
                    i.hy_test = None
            gp2, _ = create_problem_node(nargs=4, seed=1)
            ytr = [gp._ytr, gp._ytr]
            mask = [gp._mask, gp2._mask]
            vars = [Variable(k, ytr=ytr, mask=mask) for k in range(len(args))]
            [x.eval(args) for x in vars]
            mul = ff(range(len(vars)), ytr=ytr, mask=mask)
            assert mul.eval(vars)
            gp, args = create_problem_node(nargs=4, seed=0)
            vars = [Variable(k, ytr=gp._ytr, mask=gp._mask) for k in range(len(args))]
            [x.eval(args) for x in vars]
            mul2 = ff(range(len(vars)), ytr=gp._ytr, mask=gp._mask)
            assert mul2.eval(vars)
            assert mul2.hy.SSE(mul.hy[0]) == 0
            if mul.hy_test is not None:
                assert mul2.hy_test.SSE(mul.hy_test[0]) == 0
            
