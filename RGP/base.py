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
from .node import Add, Mul, Div, Fabs, Exp, Sqrt, Sin, Cos, Ln
from .node import Sq, Sigmoid, If


class Population(object):
    def __init__(self, tournament_size=2):
        self._p = []
        self._hist = []
        self._bsf = None
        self._estopping = None
        self._tournament_size = tournament_size
        self._index = None

    @property
    def hist(self):
        "List containing all the individuals generated"
        return self._hist

    @property
    def bsf(self):
        "Best so far"
        return self._bsf

    @property
    def estopping(self):
        "Early stopping individual"
        return self._estopping

    @estopping.setter
    def estopping(self, v):
        if v.fitness_vs is None:
            return
        if self.estopping is None:
            self._estopping = v
        elif v.fitness_vs > self.estopping.fitness_vs:
            self._estopping = v

    @bsf.setter
    def bsf(self, v):
        if self.bsf is None:
            self._bsf = v
        elif v.fitness > self.bsf.fitness:
            self._bsf = v

    def add(self, v):
        self._p.append(v)
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v

    def replace(self, v):
        k = self.tournament(negative=True)
        self._p[k] = v
        v.position = len(self._hist)
        self._hist.append(v)
        self.bsf = v
        self.estopping = v

    @property
    def popsize(self):
        return len(self._p)

    @property
    def population(self):
        "List containing the population"
        return self._p

    def tournament(self, negative=False):
        if self._index is None or self._index.shape[0] != self.popsize:
            self._index = np.arange(self.popsize)
        np.random.shuffle(self._index)
        vars = self._index[:self._tournament_size]
        fit = map(lambda x: self.population[x].fitness, vars)
        if negative:
            index = np.argsort(fit)[0]
        else:
            index = np.argsort(fit)[-1]
        return vars[index]


class RootGP(object):
    def __init__(self, generations=np.inf, popsize=10000,
                 seed=0,
                 tournament_size=2,
                 early_stopping_rounds=10000,
                 function_set=[Add, Mul, Div, Fabs,
                               Exp, Sqrt, Sin, Cos, Ln,
                               Sq, Sigmoid, If],
                 tr_fraction=0.8,
                 classifier=True):
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._tr_fraction = tr_fraction
        self._early_stopping_rounds = early_stopping_rounds
        self._tournament_size = tournament_size
        self._seed = seed
        self._function_set = function_set
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
        self.mask_vs()

    @property
    def function_set(self):
        "List containing the functions used to create the individuals"
        return self._function_set

    def fitness(self, v):
        v.fitness = -self._ytr.SSE(v.hy * self._mask)

    def mask_vs(self):
        if not self._classifier:
            return
        if self._tr_fraction == 1:
            return
        m = ~ self._mask.tonparray().astype(np.bool)
        f = np.zeros(self._mask.size())
        y = self.y.tonparray()
        f[y == -1] = 0.5 / (y[m] == -1).sum()
        f[y == 1] = 0.5 / (y[m] == 1).sum()
        f[~m] = 0
        self._mask_vs = SparseArray.fromlist(f)

    def fitness_vs(self, v):
        if self._classifier:
            v.fitness_vs = -((self.y - v.hy.sign()).sign().fabs() *
                             self._mask_vs).sum()
        else:
            m = (self._mask - 1).fabs()
            v.fitness_vs = -(self.y * m).SSE(v.hy * m)

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

    def random_offspring(self):
        "Returns an offspring"
        for i in range(10):
            func = self.function_set
            func = func[np.random.randint(len(func))]
            args = []
            for j in range(func.nargs):
                k = self.population.tournament()
                while k in args:
                    k = self.population.tournament()
                args.append(k)
            args = map(lambda x: self.population.population[x].position, args)
            f = func(args, ytr=self._ytr, mask=self._mask)
            if not f.eval(self._p.hist):
                continue
            if not f.isfinite():
                continue
            return f
        raise RuntimeError("Could not find a suitable random offpsring")

    @property
    def population(self):
        "Class containing the population and all the individuals generated"
        return self._p

    def population_instance(self):
        "Population instance"
        self._p = Population(tournament_size=self._tournament_size)

    def create_population(self):
        "Create the initial population"
        self.population_instance()
        while self.population.popsize < self.popsize:
            v = self.random_leaf()
            self.fitness(v)
            if not np.isfinite(v.fitness):
                continue
            if self._tr_fraction < 1:
                self.fitness_vs(v)
                if not np.isfinite(v.fitness_vs):
                    continue
            self.population.add(v)
