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
from EvoDAG.base import tonparray
from SparseArray import SparseArray
import numpy as np
from nose.tools import assert_almost_equals


def create_problem_node(nargs=4, seed=0):
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=nargs, classifier=False, seed=seed)
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


def test_compute_weight():
    gp, args = create_problem_node()
    mask = SparseArray.fromlist(np.ones(len(gp._mask)))
    n = Add(list(range(len(args))), ytr=gp._ytr, mask=mask)
    D = np.array([tonparray(i.hy) for i in args]).T
    coef = n.compute_weight([x.hy for x in args])
    r = np.linalg.lstsq(D, tonparray(gp._ytr))[0]
    [assert_almost_equals(a, b) for a, b in zip(coef, r)]


def test_node_add():
    gp, args = create_problem_node()
    n = Add(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
    D = np.array([tonparray(i.hy) for i in args]).T
    coef = n.compute_weight([x.hy for x in args])
    assert n.eval(args)
    # a = map(lambda (a, b): a.hy * b, zip(args, coef))
    a = [_a.hy * b for _a, b in zip(args, coef)]
    r = n.cumsum(a)
    print((D * coef).sum(axis=1)[:4], "*")
    print(tonparray(n.hy)[:4])
    print(tonparray(r)[:4])
    print(tonparray(gp._mask)[:4])
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
    from EvoDAG.node import Log1p
    gp, args = create_problem_node(nargs=1)
    r = args[0].hy.log1p()
    n = Log1p(list(range(len(args))), ytr=gp._ytr, mask=gp._mask)
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
        Exp, Sqrt, Sin, Cos, Log1p,\
        Sq, Min, Max
    for f, s in zip([Add, Mul, Div, Fabs,
                     Exp, Sqrt, Sin, Cos, Log1p,
                     Sq, Min, Max],
                    ['+', '*', '/', 'fabs', 'exp',
                     'sqrt', 'sin', 'cos', 'log1p',
                     'sq', 'min', 'max']):
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
    from EvoDAG.node import Fabs, Exp, Sqrt, Sin, Cos, Log1p, Sq
    from EvoDAG.node import Acos, Asin, Atan, Tan, Cosh, Sinh, Tanh
    from EvoDAG.node import Acosh, Asinh, Atanh, Expm1, Log, Log2, Log10, Lgamma
    from EvoDAG.node import Sign, Ceil, Floor

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
        for FF in [Fabs, Exp, Sqrt, Sin, Cos, Log1p, Sq,
                   Acos, Asin, Atan, Tan, Cosh, Sinh, Tanh,
                   Acosh, Asinh, Atanh, Expm1, Log, Log2, Log10, Lgamma,
                   Sign, Ceil, Floor]:
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
    from EvoDAG.node import Variable, Mul, Div, Min, Max, Atan2, Hypot
    from EvoDAG.node import Argmax, Argmin
    for ff in [Mul, Div, Min, Max, Atan2, Hypot, Argmax, Argmin]:
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
            

def test_density_safe():
    gp, args = create_problem_node(nargs=4, seed=0)
    for i in gp._function_set:
        assert i.density_safe or not i.density_safe


def test_functions_finite():
    from EvoDAG.node import Variable, Mul, Div, Min, Max, Atan2, Hypot
    from EvoDAG.node import Argmax, Argmin
    for ff in [Mul, Div, Min, Max, Atan2, Hypot, Argmax, Argmin]:
        for flag in [False, True]:
            gp, args = create_problem_node(nargs=4, seed=0)
            for v in args:
                _ = [x for x in v._eval_tr.full_array()]
                _[0] = float('inf')
                _[1] = float('inf')
                _[3] = 0
                _[4] = 0
                v._eval_tr = SparseArray.fromlist(_)
            if flag:
                for i in args:
                    i.hy_test = None
            gp2, _ = create_problem_node(nargs=4, seed=1)
            # ytr = [gp._ytr, gp._ytr]
            ytr = gp._ytr
            # mask = [gp._mask, gp2._mask]
            mask = gp._mask.full_array()
            mask[0] = 1
            mask = SparseArray.fromlist(mask)
            vars = [Variable(k, ytr=ytr, mask=mask, finite=flag)
                    for k in range(len(args))]
            for x in vars:
                _ = x.eval(args)
                print(mask.full_array())
                assert _ == flag
            if not flag:
                continue
            mul = ff(range(len(vars)), ytr=ytr, mask=mask, finite=False)
            _ = mul.eval(vars)
            if isinstance(mul, Div):
                assert not _
                mul = ff(range(len(vars)), ytr=ytr, mask=mask, finite=True)
                _ = mul.eval(vars)
                assert _


def test_classification_regression_klass():
    from EvoDAG.node import Variable
    assert Variable.classification
    assert Variable.regression


def create_problem_node2(nargs=4, seed=0):
    from EvoDAG import RootGP
    gp = RootGP(generations=1, popsize=nargs,
                multiple_outputs=True, seed=seed)
    gp.X = X
    gp.Xtest = X
    y = cl.copy()
    gp.nclasses(y)
    gp.y = y
    return gp, [gp.X[x] for x in range(nargs)]


def test_functions_extra_args():
    from EvoDAG.node import Variable, Mul, Div, Min, Max, Atan2, Hypot
    from EvoDAG.node import Argmax, Argmin
    for ff in [Mul, Div, Min, Max, Atan2, Hypot, Argmax, Argmin]:
        gp, args = create_problem_node2(nargs=4, seed=0)
        vars = [Variable(k, ytr=gp._ytr, y_klass=gp._y_klass,
                         mask=gp._mask, finite=True)
                for k in range(len(args))]
        [x.eval(args) for x in vars]
        mul = ff(range(len(vars)), ytr=gp._ytr, klass=gp._y_klass,
                 mask=gp._mask, finite=True)
        mul.eval(vars)


def test_naive_bayes():
    import numpy as np
    from EvoDAG.node import Variable, NaiveBayes
    gp, args = create_problem_node2(nargs=4, seed=0)
    gp.random_leaf()
    vars = [Variable(k, ytr=gp._ytr, y_klass=gp._y_klass,
                     mask=gp._mask, finite=True)
            for k in range(len(args))]
    [x.eval(args) for x in vars]
    naive_bayes = NaiveBayes(range(len(vars)), ytr=gp._ytr, naive_bayes=gp._naive_bayes,
                             mask=gp._mask, finite=True)
    naive_bayes.eval(vars)
    mask = np.array(gp._mask_ts.sign().full_array(), dtype=np.bool)
    klass = np.array(gp._y_klass.full_array())[mask]
    unique_klass = np.unique(klass)
    mean = []
    std2 = []
    p_klass = []
    l = []
    for v in vars:
        l += v.hy
    for v in l:
        var = np.array(v.full_array())[mask]
        mean.append([np.mean(var[k == klass]) for k in unique_klass])
        std2.append([np.var(var[k == klass]) for k in unique_klass])
        p_klass = [(k == klass).mean() for k in unique_klass]
    mean = np.array(mean)
    std2 = np.array(std2)
    p_klass = np.array(p_klass)
    likelihood = []
    for i in range(unique_klass.shape[0]):
        a = np.log(p_klass[i])
        b = - 0.5 * np.sum([np.log(2. * np.pi * s[i]) for s in std2])
        _ = [(x + -m[i]).sq() * (1 / s[i]) for x, m, s in zip(l, mean, std2)]
        _ = SparseArray.cumsum(_) * -0.5
        likelihood.append(_ + b + a)
    for a, b in zip(likelihood, naive_bayes.hy):
        [assert_almost_equals(v, w) for v, w in zip(a.data, b.data)]


def test_naive_bayes_sklearn():
    from EvoDAG.naive_bayes import NaiveBayes as NB
    from EvoDAG.node import NaiveBayes
    try:
        from sklearn.naive_bayes import GaussianNB
    except ImportError:
        return

    class Var(object):
        def __init__(self, a):
            self.hy = a
            self.hy_test = None

    m = GaussianNB().fit(X, cl)
    hy = m._joint_log_likelihood(X)
    vars = [Var(SparseArray.fromlist(x)) for x in X.T]
    nb = NB(mask=SparseArray.fromlist([1 for _ in X[:, 0]]), klass=SparseArray.fromlist(cl),
            nclass=3)
    naive_bayes = NaiveBayes([x for x in range(4)], naive_bayes=nb)
    naive_bayes.eval(vars)
    for a, b in zip(naive_bayes.hy, hy.T):
        for a1, b1 in zip(a.full_array(), b):
            assert_almost_equals(a1, b1, 3)


def test_naive_bayes_MN():
    import numpy as np
    import math
    from EvoDAG.node import Variable, NaiveBayesMN
    from EvoDAG.naive_bayes import NaiveBayes as MN
    gp, args = create_problem_node2(nargs=3, seed=0)
    gp.random_leaf()
    vars = [Variable(k, ytr=gp._ytr, y_klass=gp._y_klass,
                     mask=gp._mask, finite=True)
            for k in range(len(args))]
    [x.eval(args) for x in vars]
    nb = MN(mask=gp._mask_ts, klass=gp._y_klass,
            nclass=gp._labels.shape[0])
    l = []
    [[l.append(y) for y in x.hy] for x in vars]
    coef = [nb.coef_MN(x) for x in l]
    p_klass = coef[0][1]
    coef = [(k, x[0]) for k, x in enumerate(coef) if np.all(np.isfinite(np.array(x[0])))]
    R = []
    for k, p in enumerate(p_klass):
        r = p
        for v, w in coef:
            r += (l[v] * w[k])
        R.append(r)
    naive_bayes = NaiveBayesMN(range(len(vars)), ytr=gp._ytr, naive_bayes=gp._naive_bayes,
                               mask=gp._mask, finite=True)
    naive_bayes.eval(vars)
    for a, b in zip(R, naive_bayes.hy):
        [assert_almost_equals(math.exp(v), w) for v, w in zip(a.data, b.data)]


def test_naive_bayes_MN_variable():
    from EvoDAG.node import NaiveBayesMN
    gp, args = create_problem_node2(nargs=3, seed=0)
    gp.random_leaf()
    naive_bayes = NaiveBayesMN(range(len(gp.X)), ytr=gp._ytr, naive_bayes=gp._naive_bayes,
                               mask=gp._mask, finite=True)
    naive_bayes.eval(gp.X)
    assert len(naive_bayes.hy) == 3


def test_multiple_variables():
    from EvoDAG.node import MultipleVariables
    gp, args = create_problem_node2(nargs=3, seed=0)
    gp.random_leaf()
    mv = MultipleVariables([x for x in range(len(gp.X))],
                           ytr=gp._ytr, naive_bayes=gp._naive_bayes,
                           mask=gp._mask, finite=True)
    mv.eval(gp.X)
    assert len(mv.hy) == 3
    mv2 = MultipleVariables([x for x in range(len(gp.X))],
                            ytr=gp._ytr[0], mask=gp._mask[0], finite=True)
    mv2.eval(gp.X)
    assert isinstance(mv2.hy, SparseArray)
    assert mv2.hy.SSE(mv.hy[0]) == 0
    l = []
    for a, b in zip(gp.X, mv2.weight):
        l.append(a.hy * b)
    r = SparseArray.cumsum(l)
    assert mv2.hy.SSE(r) == 0
