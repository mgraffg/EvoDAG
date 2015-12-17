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
import types


class Variable(object):
    def __init__(self, variable, weight=None, ytr=None, mask=None):
        if isinstance(variable, types.ListType):
            variable = variable if len(variable) > 1 else variable[0]
        self._variable = variable
        self._weight = weight
        self._eval_tr = None
        self._eval_ts = None
        self._ytr = ytr
        self._mask = mask

    @property
    def variable(self):
        "The variable is indicated by the position in RootGP.X"
        return self._variable

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
        b = np.array(map(lambda f: (f * self._ytr).sum(), r))
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


class Add(Variable):
    nargs = 2

    @staticmethod
    def cumsum(r):
        a = r[0]
        for x in r[1:]:
            a = a + x
        return a

    def eval(self, X):
        X = map(lambda x: X[x], self.variable)
        w = self.compute_weight(map(lambda x: x.hy, X))
        if w is None:
            return False
        self.weight = w
        r = map(lambda (v, w): v.hy * w, zip(X, self.weight))
        r = self.cumsum(r)
        self.hy = r
        if X[0].hy_test is not None:
            r = map(lambda (v, w): v.hy_test * w, zip(X, self.weight))
            r = self.cumsum(r)
            self.hy_test = r
        return True


class Mul(Variable):
    nargs = 2

    @staticmethod
    def cumprod(r):
        a = r[0]
        for x in r[1:]:
            a = a * x
        return a

    def eval(self, X):
        X = map(lambda x: X[x], self.variable)
        r = self.cumprod(map(lambda x: x.hy, X))
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = self.cumprod(map(lambda x: x.hy_test, X))
            self.hy_test = r * self.weight
        return True


class Div(Variable):
    nargs = 2

    def eval(self, X):
        a, b = X[self.variable[0]], X[self.variable[1]]
        r = a.hy / b.hy
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X[0].hy_test is not None:
            r = a.hy_test / b.hy_test
            self.hy_test = r * self.weight
        return True


class Fabs(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.fabs()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.fabs() * self.weight
        return True


class Exp(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.exp()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.exp() * self.weight
        return True


class Sqrt(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sqrt()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sqrt() * self.weight
        return True


class Sin(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sin()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sin() * self.weight
        return True


class Cos(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.cos()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.cos() * self.weight
        return True


class Ln(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.ln()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.ln() * self.weight
        return True


class Sq(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sq()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sq() * self.weight
        return True


class Sigmoid(Variable):
    nargs = 1

    def eval(self, X):
        X = X[self.variable]
        r = X.hy.sigmoid()
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if X.hy_test is not None:
            self.hy_test = X.hy_test.sigmoid() * self.weight
        return True


class If(Variable):
    nargs = 3

    def eval(self, X):
        X = map(lambda x: X[x], self.variable)
        a, b, c = X
        r = a.hy.if_func(b.hy, c.hy)
        w = self.compute_weight([r])
        if w is None:
            return False
        self.weight = w[0]
        self.hy = r * self.weight
        if a.hy_test is not None:
            r = a.hy_test.if_func(b.hy_test, c.hy_test)
            self.hy_test = r * self.weight
        return True
