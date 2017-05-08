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
from .linalg_solve import compute_weight
from .cython_utils import naive_bayes as NB
from .cython_utils import naive_bayes_MN as MN
from .cython_utils import naive_bayes_isfinite, naive_bayes_isfinite_MN
from SparseArray import SparseArray


class Variable(object):
    classification = True
    regression = True

    def __init__(self, variable, weight=None, ytr=None,
                 mask=None, height=0, finite=True, **kwargs):
        if isinstance(variable, list):
            variable = variable if len(variable) > 1 else variable[0]
        self._variable = variable
        self._weight = weight
        self._eval_tr = None
        self._eval_ts = None
        self._ytr = ytr
        self._mask = mask
        self._fitness = None
        self._fitness_vs = None
        self._position = 0
        self._height = height
        self._multiple_outputs = False
        self._n_outputs = 1
        self._finite = finite
        if isinstance(ytr, list) and len(ytr) > 1:
            self._multiple_outputs = True
            self._n_outputs = len(ytr)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, v):
        self._height = v

    def tostore(self):
        ins = self.__class__(self.variable, weight=self.weight)
        ins.position = self.position
        ins.height = self.height
        ins._fitness = self._fitness
        ins._fitness_vs = self._fitness_vs
        ins._multiple_outputs = self._multiple_outputs
        ins._n_outputs = self._n_outputs
        return ins

    @property
    def position(self):
        "Position where this variable is in the history"
        return self._position

    @position.setter
    def position(self, v):
        self._position = v

    @property
    def fitness(self):
        "Stores the fitness on the training set"
        return self._fitness

    @fitness.setter
    def fitness(self, v):
        self._fitness = v

    @property
    def fitness_vs(self):
        "Stores the fitness on the validation set"
        return self._fitness_vs

    @fitness_vs.setter
    def fitness_vs(self, v):
        self._fitness_vs = v

    @property
    def variable(self):
        "The variable is indicated by the position in EvoDAG.X"
        return self._variable

    @variable.setter
    def variable(self, v):
        self._variable = v

    @property
    def weight(self):
        "The weight is obtained by OLS on RootGP._ytr"
        return self._weight

    @weight.setter
    def weight(self, v):
        self._weight = v

    def compute_weight(self, r, ytr=None, mask=None):
        """Returns the weight (w) using OLS of r * w = gp._ytr """
        ytr = self._ytr if ytr is None else ytr
        mask = self._mask if mask is None else mask
        return compute_weight(r, ytr, mask)

    def raw_outputs(self, X):
        r = X[self.variable].hy
        hr = X[self.variable].hy_test
        return r, hr

    def set_weight(self, r):
        if self.weight is None:
            if not self._multiple_outputs:
                ytr = [self._ytr]
                mask = [self._mask]
            else:
                ytr = self._ytr
                mask = self._mask
            W = []
            for _ytr, _mask in zip(ytr, mask):
                w = self.compute_weight([r], ytr=_ytr, mask=_mask)
                if w is None:
                    return False
                W.append(w[0])
            if not self._multiple_outputs:
                self.weight = W[0]
            else:
                self.weight = W
        return True

    def _mul(self, a, w):
        if not isinstance(w, list):
            return a * w
        if not isinstance(a, list):
            return [a * x for x in w]
        return [x * y for x, y in zip(a, w)]

    def eval(self, X):
        r, hr = self.raw_outputs(X)
        if isinstance(r, list):
            if self._finite:
                r = [x.finite(inplace=True) for x in r]
            else:
                r = [x for x in r]
        else:
            if self._finite:
                r = r.finite(inplace=True)
        if not self.set_weight(r):
            return False
        self.hy = self._mul(r, self.weight)
        if hr is not None:
            if isinstance(hr, list):
                if self._finite:
                    hr = [x.finite(inplace=True) for x in hr]
                else:
                    hr = [x for x in hr]
            else:
                if self._finite:
                    hr = hr.finite(inplace=True)
            self.hy_test = self._mul(hr, self.weight)
        return True

    def isfinite(self):
        "Test whether the predicted values are finite"
        if self._multiple_outputs:
            if self.hy_test is not None:
                r = [(hy.isfinite() and (hyt is None or hyt.isfinite()))
                     for hy, hyt in zip(self.hy, self.hy_test)]
            else:
                r = [hy.isfinite() for hy in self.hy]
            return np.all(r)
        return self.hy.isfinite() and (self.hy_test is None or
                                       self.hy_test.isfinite())

    @property
    def hy(self):
        "Predicted values of the training and validation set"
        return self._eval_tr

    @hy.setter
    def hy(self, v):
        self._eval_tr = v

    @property
    def hy_test(self):
        "Predicted values of the test set"
        return self._eval_ts

    @hy_test.setter
    def hy_test(self, v):
        self._eval_ts = v


class Function(Variable):
    nargs = 2
    unique_args = False

    def tostore(self):
        ins = super(Function, self).tostore()
        ins.nargs = self.nargs
        return ins

    def signature(self):
        vars = self._variable
        if not isinstance(vars, list):
            vars = [vars]
        c = self.symbol + '|' + '|'.join([str(x) for x in vars])
        return c

    def hy2listM(self, X):
        if not self._multiple_outputs:
            hy = [x.hy for x in X]
            if X[0].hy_test is not None:
                hyt = [x.hy_test for x in X]
            else:
                hyt = None
        else:
            hy = [list() for x in range(self._n_outputs)]
            [[hy[k].append(x.hy[k]) for x in X] for k in range(self._n_outputs)]
            if X[0].hy_test is not None:
                hyt = [list() for x in range(self._n_outputs)]
                [[hyt[k].append(x.hy_test[k]) for x in X]
                 for k in range(self._n_outputs)]
            else:
                hyt = None
        return hy, hyt

    def hy2list(self, X):
        if not self._multiple_outputs:
            hy = [X.hy]
            hyt = [X.hy_test]
        else:
            hy = X.hy
            hyt = X.hy_test
        return hy, hyt

    def return_r_hr(self, r, hr):
        if len(r) == 1:
            r = r[0]
        if hr is None:
            pass
        elif len(hr) == 0:
            hr = None
        elif len(hr) == 1:
            hr = hr[0]
        return r, hr


class Function1(Function):
    def set_weight(self, r):
        if self.weight is None:
            if not self._multiple_outputs:
                ytr = [self._ytr]
                mask = [self._mask]
                r = [r]
            else:
                ytr = self._ytr
                mask = self._mask
            W = []
            for _r, _ytr, _mask in zip(r, ytr, mask):
                w = self.compute_weight([_r], ytr=_ytr, mask=_mask)
                if w is None:
                    return False
                W.append(w[0])
            if not self._multiple_outputs:
                self.weight = W[0]
            else:
                self.weight = W
        return True

    def _raw_outputs(self, X, func):
        X = X[self.variable]
        hy, hyt = self.hy2list(X)
        r = [getattr(x, func)() for x in hy]
        hr = None
        if hyt is not None:
            hr = [getattr(x, func)() for x in hyt if x is not None]
        return self.return_r_hr(r, hr)

    def raw_outputs(self, X):
        return self._raw_outputs(X, self.symbol)


class Add(Function):
    nargs = 5
    symbol = '+'
    unique_args = True
    min_nargs = 2
    density_safe = True

    def __init__(self, *args, **kwargs):
        super(Add, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cumsum(r):
        return SparseArray.cumsum(r)

    def set_weight(self, hy):
        if self.weight is not None:
            return True
        if not self._multiple_outputs:
            w = self.compute_weight(hy, ytr=self._ytr, mask=self._mask)
            if w is None:
                return False
            self.weight = w
        else:
            W = []
            for _hy, _ytr, _mask in zip(hy, self._ytr, self._mask):
                w = self.compute_weight(_hy, ytr=_ytr, mask=_mask)
                if w is None:
                    return False
                W.append(w)
            self.weight = W
        return True

    @staticmethod
    def _mul(a, b):
        return [_a * _b for _a, _b in zip(a, b)]

    def eval(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        if not self.set_weight(hy):
            return False
        if self._multiple_outputs:
            r = [self.cumsum(self._mul(a, w)) for a, w in zip(hy, self.weight)]
            if hyt is not None:
                hr = [self.cumsum(self._mul(a, w)) for a, w in zip(hyt, self.weight)]
            else:
                hr = None
        else:
            r = self.cumsum([x * w for x, w in zip(hy, self.weight)])
            if hyt is not None:
                hr = self.cumsum([x * w for x, w in zip(hyt, self.weight)])
            else:
                hr = None
        self.hy = r
        self.hy_test = hr
        return True


class Mul(Function1):
    symbol = '*'
    density_safe = False

    def __init__(self, *args, **kwargs):
        super(Mul, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cumprod(r):
        return SparseArray.cummul(r)

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [self.cumprod(x) for x in hy]
            if hyt is not None:
                hr = [self.cumprod(x) for x in hyt]
        else:
            r = self.cumprod(hy)
            if hyt is not None:
                hr = self.cumprod(hyt)
        return r, hr


class Div(Function1):
    symbol = '/'
    density_safe = False

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [x[0] / x[1] for x in hy]
            if hyt is not None:
                hr = [x[0] / x[1] for x in hyt]
        else:
            r = hy[0] / hy[1]
            if hyt is not None:
                hr = hyt[0] / hyt[1]
        return r, hr


class Atan2(Function1):
    symbol = 'atan2'
    density_safe = False

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [x[0].atan2(x[1]) for x in hy]
            if hyt is not None:
                hr = [x[0].atan2(x[1]) for x in hyt]
        else:
            r = hy[0].atan2(hy[1])
            if hyt is not None:
                hr = hyt[0].atan2(hyt[1])
        return r, hr


class Hypot(Function1):
    symbol = 'hypot'
    density_safe = True

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [x[0].hypot(x[1]) for x in hy]
            if hyt is not None:
                hr = [x[0].hypot(x[1]) for x in hyt]
        else:
            r = hy[0].hypot(hy[1])
            if hyt is not None:
                hr = hyt[0].hypot(hyt[1])
        return r, hr


class OneArg(Function1):
    nargs = 1


class Acos(OneArg):
    symbol = 'acos'
    density_safe = False


class Asin(OneArg):
    symbol = 'asin'
    density_safe = True


class Atan(OneArg):
    symbol = 'atan'
    density_safe = True


class Cos(OneArg):
    symbol = 'cos'
    density_safe = False


class Sin(OneArg):
    symbol = 'sin'
    density_safe = True


class Tan(OneArg):
    symbol = 'tan'
    density_safe = True


class Cosh(OneArg):
    symbol = 'cosh'
    density_safe = False


class Sinh(OneArg):
    symbol = 'sinh'
    density_safe = True


class Tanh(OneArg):
    symbol = 'tanh'
    density_safe = True


class Acosh(OneArg):
    symbol = 'acosh'
    density_safe = False


class Asinh(OneArg):
    symbol = 'asinh'
    density_safe = True


class Atanh(OneArg):
    symbol = 'atanh'
    density_safe = True


class Exp(OneArg):
    symbol = 'exp'
    density_safe = False


class Expm1(OneArg):
    symbol = 'expm1'
    density_safe = True


class Log(OneArg):
    symbol = 'log'
    density_safe = False


class Log2(OneArg):
    symbol = 'log2'
    density_safe = False


class Log10(OneArg):
    symbol = 'log10'
    density_safe = False


class Log1p(OneArg):
    symbol = 'log1p'
    density_safe = True


class Lgamma(OneArg):
    symbol = 'lgamma'
    density_safe = False


class Sqrt(OneArg):
    symbol = 'sqrt'
    density_safe = True


class Sq(OneArg):
    symbol = 'sq'
    density_safe = True


class Sign(OneArg):
    symbol = 'sign'
    density_safe = True


class Fabs(OneArg):
    symbol = 'fabs'
    density_safe = True


class Ceil(OneArg):
    symbol = 'ceil'
    density_safe = True


class Floor(OneArg):
    symbol = 'floor'
    density_safe = True


class Min(Function1):
    nargs = 2
    symbol = 'min'
    unique_args = True
    min_nargs = 2
    density_safe = False

    def __init__(self, *args, **kwargs):
        super(Min, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cummin(r):
        return SparseArray.cummin(r)

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [self.cummin(x) for x in hy]
            if hyt is not None:
                hr = [self.cummin(x) for x in hyt]
        else:
            r = self.cummin(hy)
            if hyt is not None:
                hr = self.cummin(hyt)
        return r, hr


class Max(Function1):
    nargs = 2
    symbol = 'max'
    unique_args = True
    min_nargs = 2
    density_safe = True

    def __init__(self, *args, **kwargs):
        super(Max, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cummax(r):
        return SparseArray.cummax(r)

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [self.cummax(x) for x in hy]
            if hyt is not None:
                hr = [self.cummax(x) for x in hyt]
        else:
            r = self.cummax(hy)
            if hyt is not None:
                hr = self.cummax(hyt)
        return r, hr


class Argmax(Function1):
    nargs = 2
    symbol = 'argmax'
    unique_args = True
    min_nargs = 2
    density_safe = True

    @staticmethod
    def argop(r):
        return SparseArray.argmax(r)

    def raw_outputs(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        hr = None
        if self._multiple_outputs:
            r = [self.argop(x) for x in hy]
            if hyt is not None:
                hr = [self.argop(x) for x in hyt]
        else:
            r = self.argop(hy)
            if hyt is not None:
                hr = self.argop(hyt)
        return r, hr


class Argmin(Argmax):
    nargs = 2
    symbol = 'argmin'
    unique_args = True
    min_nargs = 2
    density_safe = True

    @staticmethod
    def argop(r):
        return SparseArray.argmin(r)


class NaiveBayes(Function):
    nargs = 2
    min_nargs = 2
    symbol = 'NB'
    density_safe = False
    unique_args = True
    regression = False

    def __init__(self, variable, naive_bayes=None, **kwargs):
        super(NaiveBayes, self).__init__(variable, **kwargs)
        self._variable = sorted(self._variable)
        self._naive_bayes = naive_bayes

    def hy2listM(self, X):
        if self._multiple_outputs:
            hy = []
            for x in X:
                _ = x.hy
                if isinstance(_, list):
                    hy += _
                else:
                    hy.append(_)
            hyt = None
            if X[0].hy_test is not None:
                hyt = []
                for x in X:
                    _ = x.hy_test
                    if isinstance(_, list):
                        hyt += _
                    else:
                        hyt.append(_)
            return hy, hyt
        hy = [x.hy for x in X]
        hyt = None
        if X[0].hy_test is not None:
            hyt = [x.hy_test for x in X]
        return hy, hyt

    def set_weight(self, X):
        if self.weight is not None:
            return True
        if self._naive_bayes is None:
            return False
        coef = self._naive_bayes.coef
        nclass = self._naive_bayes._nclass
        w = []
        v = []
        for k, x in enumerate(X):
            c = coef(x)
            if not naive_bayes_isfinite(c, nclass):
                continue
            w.append(c)
            v.append(k)
        if len(v) == 0:
            return False
        self.weight = [w, v, nclass]
        return True

    def eval(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        if not self.set_weight(hy):
            return False
        weight, var, nclass = self.weight
        self.hy = NB([hy[x] for x in var], weight, nclass)
        if hyt is not None:
            self.hy_test = NB([hyt[x] for x in var], weight, nclass)
        return True


class NaiveBayesMN(NaiveBayes):
    symbol = 'MN'
    density_safe = True
    regression = False

    def set_weight(self, X):
        if self.weight is not None:
            return True
        if self._naive_bayes is None:
            return False
        coef = self._naive_bayes.coef_MN
        nclass = self._naive_bayes._nclass
        w = []
        v = []
        for k, x in enumerate(X):
            c = coef(x)
            if not naive_bayes_isfinite_MN(c, nclass):
                continue
            w.append(c)
            v.append(k)
        if len(v) == 0:
            return False
        self.weight = [w, v, nclass]
        return True

    def eval(self, X):
        X = [X[x] for x in self.variable]
        hy, hyt = self.hy2listM(X)
        if not self.set_weight(hy):
            return False
        weight, var, nclass = self.weight
        self.hy = MN([hy[x] for x in var], weight, nclass)
        if hyt is not None:
            self.hy_test = MN([hyt[x] for x in var], weight, nclass)
        return True


class MultipleVariables(Add):
    symbol = 'X'

    def raw_outputs(self, X):
        r = [X[x].hy for x in self.variable]
        hr = None
        if X[0].hy_test is not None:
            hr = [X[x].hy_test for x in self.variable]
        return r, hr

    def set_weight(self, r):
        if self.weight is None:
            if not self._multiple_outputs:
                ytr = [self._ytr]
                mask = [self._mask]
            else:
                ytr = self._ytr
                mask = self._mask
            W = []
            for _ytr, _mask in zip(ytr, mask):
                w = self.compute_weight(r, ytr=_ytr, mask=_mask)
                if w is None:
                    return False
                W.append(w)
            if not self._multiple_outputs:
                self.weight = W[0]
            else:
                self.weight = W
        return True

    def _mul(self, a, w):
        cumsum = SparseArray.cumsum
        if not isinstance(w, list):
            return cumsum([x * y for x, y in zip(a, w)])
        return [cumsum([x * y1 for x, y1 in zip(a, y)]) for y in w]

    def eval(self, X):
        r, hr = self.raw_outputs(X)
        if isinstance(r, list):
            if self._finite:
                r = [x.finite(inplace=True) for x in r]
            else:
                r = [x for x in r]
        else:
            if self._finite:
                r = r.finite(inplace=True)
        if not self.set_weight(r):
            return False
        self.hy = self._mul(r, self.weight)
        if hr is not None:
            if isinstance(hr, list):
                if self._finite:
                    hr = [x.finite(inplace=True) for x in hr]
                else:
                    hr = [x for x in hr]
            else:
                if self._finite:
                    hr = hr.finite(inplace=True)
            self.hy_test = self._mul(hr, self.weight)
        return True
        
