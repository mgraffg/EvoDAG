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
from .sparse_array import SparseArray
from .node import Variable


class RootGP(object):
    def __init__(self, generations=10, popsize=10000,
                 seed=0,
                 tr_fraction=0.8,
                 classifier=True):
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._tr_fraction = tr_fraction
        self._seed = seed
        np.random.seed(self._seed)

    @property
    def popsize(self):
        """Population size"""
        return self._popsize

    @property
    def generations(self):
        """Number of generations"""
        return self._generations

    @property
    def X(self):
        """Features or variables used in the training and validation set"""
        return self._X

    @X.setter
    def X(self, v):
        self._X = self.convert_features(v)
        self.nvar = len(self._X)

    @property
    def Xtest(self):
        "Features or variables used in the test set"
        return map(lambda x: x.hy_test, self.X)

    @Xtest.setter
    def Xtest(self, v):
        if isinstance(v, np.ndarray):
            X = v.T
        else:
            X = v
        for var, d in zip(self.X, X):
            var._eval_ts = SparseArray.fromlist(d)

    @property
    def y(self):
        """Dependent variable"""
        return self._y

    def set_classifier_mask(self, v):
        """Computes the mask used to create the training and validation set"""
        v = v.tonparray()
        a = np.unique(v)
        mask = np.zeros_like(v)
        cnt = min(map(lambda x: (v == x).sum(), a)) * self._tr_fraction
        for i in a:
            index = np.where(v == i)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = True
        self._mask = SparseArray.fromlist(mask)

    @y.setter
    def y(self, v):
        if isinstance(v, np.ndarray):
            v = SparseArray.fromlist(v)
        if self._classifier:
            self.set_classifier_mask(v)
        elif self._tr_fraction < 1:
            index = np.arange(v.size())
            np.random.shuffle(index)
            ones = np.ones(v.size())
            ones[index[self._tr_fraction * v.size():]] = 0
            self._mask = SparseArray.fromlist(ones)
        else:
            self._mask = 1.0
        self._ytr = v * self._mask
        self._y = v

    def convert_features(self, v):
        if isinstance(v[0], Variable):
            return v
        if isinstance(v, np.ndarray):
            X = v.T
        else:
            X = v
        lst = []
        for var, d in enumerate(X):
            v = Variable(var, 1)
            v._eval_tr = SparseArray.fromlist(d)
            lst.append(v)
        return lst

    @property
    def nvar(self):
        """Number of features or variables"""
        return self._nvar

    @nvar.setter
    def nvar(self, v):
        self._nvar = v

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

    def random_leaf(self):
        "Returns a random variable with the associated weight"
        for i in range(10):
            var = np.random.randint(self.nvar)
            v = Variable(var, ytr=self._ytr, mask=self._mask)
            if not v.eval(self.X):
                continue
            if not v.isfinite():
                continue
            return v
        raise RuntimeError("Could not find a suitable random leaf")
