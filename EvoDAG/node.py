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


class Variable(object):
    def __init__(self, variable, weight=None, ytr=None, mask=None,
                 height=0):
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
        "The variable is indicated by the position in RootGP.X"
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

    def compute_weight(self, r):
        """Returns the weight (w) using OLS of r * w = gp._ytr """
        A = np.empty((len(r), len(r)))
        b = np.array([(f * self._ytr).sum() for f in r])
        for i in range(len(r)):
            r[i] = r[i] * self._mask
            for j in range(i, len(r)):
                A[i, j] = (r[i] * r[j]).sum()
                A[j, i] = A[i, j]
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            return None
        try:
            coef = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        return coef

    def eval(self, X):
        if self.weight is None:
            w = self.compute_weight([X[self.variable].hy])
            if w is None:
                return False
            self.weight = w
        self.hy = X[self.variable].hy * self.weight
        xtest = X[self.variable].hy_test
        if xtest is not None:
            self.hy_test = xtest * self.weight
        return True

    def isfinite(self):
        "Test whether the predicted values are finite"
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
    color = 1

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


class Add(Function):
    nargs = 5
    symbol = '+'
    color = 1

    def __init__(self, *args, **kwargs):
        super(Add, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cumsum(r):
        a = r[0]
        for x in r[1:]:
            a = a + x
        return a

    def eval(self, X):
        X = [X[x] for x in self.variable]
        if self.weight is None:
            w = self.compute_weight([x.hy for x in X])
            if w is None:
                return False
            self.weight = w
        # r = map(lambda (v, w): v.hy * w, zip(X, self.weight))
        r = [v.hy * w1 for v, w1 in zip(X, self.weight)]
        r = self.cumsum(r)
        self.hy = r
        if X[0].hy_test is not None:
            # r = map(lambda (v, w): v.hy_test * w, zip(X, self.weight))
            r = [v.hy_test * w1 for v, w1 in zip(X, self.weight)]
            r = self.cumsum(r)
            self.hy_test = r
        return True


class Mul(Function):
    symbol = '*'
    color = 1

    def __init__(self, *args, **kwargs):
        super(Mul, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cumprod(r):
        a = r[0]
        for x in r[1:]:
            a = a * x
        return a

    def eval(self, X):
        X = [X[x] for x in self.variable]
        r = self.cumprod([x.hy for x in X])
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = self.cumprod([x.hy_test for x in X])
            self.hy_test = r * self.weight
        return True


class Div(Function):
    symbol = '/'
    color = 1

    def eval(self, X):
        a, b = X[self.variable[0]], X[self.variable[1]]
        r = a.hy / b.hy
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = a.hy_test / b.hy_test
            self.hy_test = r * self.weight
        return True


class Fabs(Function):
    nargs = 1
    symbol = 'fabs'
    color = 2

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.fabs()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.fabs() * self.weight
        return True


class Exp(Function):
    nargs = 1
    symbol = 'exp'
    color = 3

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.exp()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.exp() * self.weight
        return True


class Sqrt(Function):
    nargs = 1
    symbol = 'sqrt'
    color = 4
    
    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sqrt()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sqrt() * self.weight
        return True


class Sin(Function):
    nargs = 1
    symbol = 'sin'
    color = 5

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sin()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sin() * self.weight
        return True


class Cos(Function):
    nargs = 1
    symbol = 'cos'
    color = 5

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.cos()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.cos() * self.weight
        return True


class Ln(Function):
    nargs = 1
    symbol = 'ln'
    color = 6
    
    def eval(self, X):
        X = X[self.variable]
        r = X.hy.ln()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.ln() * self.weight
        return True


class Sq(Function):
    nargs = 1
    symbol = 'sq'
    color = 4

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sq()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sq() * self.weight
        return True


class Sigmoid(Function):
    nargs = 1
    symbol = 's'
    color = 6

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sigmoid()
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sigmoid() * self.weight
        return True


class If(Function):
    nargs = 3
    symbol = 'if'
    color = 7

    def eval(self, X):
        X = [X[x] for x in self.variable]
        a, b, c = X
        r = a.hy.if_func(b.hy, c.hy)
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if a.hy_test is not None:
            r = a.hy_test.if_func(b.hy_test, c.hy_test)
            self.hy_test = r * self.weight
        return True


class Min(Function):
    nargs = 2
    symbol = 'min'
    color = 8
    
    def __init__(self, *args, **kwargs):
        super(Min, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cummin(r):
        a = r[0]
        for x in r[1:]:
            a = a.min(x)
        return a

    def eval(self, X):
        X = [X[x] for x in self.variable]
        r = self.cummin([x.hy for x in X])
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = self.cummin([x.hy_test for x in X])
            self.hy_test = r * self.weight
        return True


class Max(Function):
    nargs = 2
    symbol = 'max'
    color = 8

    def __init__(self, *args, **kwargs):
        super(Max, self).__init__(*args, **kwargs)
        self._variable = sorted(self._variable)

    @staticmethod
    def cummax(r):
        a = r[0]
        for x in r[1:]:
            a = a.max(x)
        return a

    def eval(self, X):
        X = [X[x] for x in self.variable]
        r = self.cummax([x.hy for x in X])
        if self.weight is None:
            w = self.compute_weight([r])
            if w is None:
                return False
            self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = self.cummax([x.hy_test for x in X])
            self.hy_test = r * self.weight
        return True
        
