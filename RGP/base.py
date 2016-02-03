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
import logging
import types
from .sparse_array import SparseArray
from .node import Variable, Function
from .node import Add, Mul, Div, Fabs, Exp, Sqrt, Sin, Cos, Ln
from .node import Sq, Sigmoid, If
from .model import Model, Models
from .population import Population


class RootGP(object):
    def __init__(self, generations=np.inf, popsize=10000,
                 seed=0,
                 tournament_size=2,
                 early_stopping_rounds=-1,
                 function_set=[Add, Mul, Div, Fabs,
                               Exp, Sqrt, Sin, Cos, Ln,
                               Sq, Sigmoid, If],
                 tr_fraction=0.8,
                 classifier=True,
                 labels=None):
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._tr_fraction = tr_fraction
        if early_stopping_rounds is not None and early_stopping_rounds < 0:
            early_stopping_rounds = popsize * 2
        self._early_stopping_rounds = early_stopping_rounds
        self._tournament_size = tournament_size
        self._seed = seed
        self._labels = labels
        self._multiclass = False
        self._function_set = function_set
        np.random.seed(self._seed)
        self._logger = logging.getLogger('RGP.RootGP')
        if self._generations == np.inf and tr_fraction == 1:
            raise RuntimeError("Infinite evolution, set generations\
            or tr_fraction < 1 ")

    def get_params(self):
        "Parameters used to initialize the class"
        import inspect
        a = inspect.getargspec(self.__init__)[0]
        out = dict()
        for key in a[1:]:
            value = getattr(self, "_%s" % key, None)
            out[key] = value
        return out

    def clone(self):
        "Clone the class without the population"
        return self.__class__(**self.get_params())

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
        self._X = Model.convert_features(v)
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
            if isinstance(d, SparseArray):
                var._eval_ts = d
            else:
                var._eval_ts = SparseArray.fromlist(d)

    @property
    def y(self):
        """Dependent variable"""
        return self._y

    def set_classifier_mask(self, v):
        """Computes the mask used to create the training and validation set"""
        v = v.tonparray()
        a = np.unique(v)
        if a[0] != -1 or a[1] != 1:
            raise RuntimeError("The labels must be -1 and 1")
        mask = np.zeros_like(v)
        cnt = min(map(lambda x: (v == x).sum(), a)) * self._tr_fraction
        for i in a:
            index = np.where(v == i)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = True
        self._mask = SparseArray.fromlist(mask)
        return SparseArray.fromlist(v)

    def multiclass(self, X, v, test_set=None):
        "Performing One vs All multiclass classification"
        from sklearn import preprocessing
        if not isinstance(v, np.ndarray):
            v = v.tonparray()
        a = preprocessing.LabelBinarizer().fit(v)
        mask = a.transform(v).astype(np.bool).T
        self._multiclass_instances = map(lambda x: self.clone(), mask)
        for m, gp in zip(mask, self._multiclass_instances):
            y = np.zeros_like(m) - 1
            y[m] = 1
            gp.fit(X, y, test_set=test_set)
        return self

    @y.setter
    def y(self, v):
        if isinstance(v, np.ndarray):
            v = SparseArray.fromlist(v)
        if self._classifier:
            if self._labels is not None and\
               (self._labels[0] != -1 or self._labels[1] != 1):
                v = v.tonparray()
                mask = np.ones_like(v, dtype=np.bool)
                mask[v == self._labels[0]] = False
                v[mask] = 1
                v[~mask] = -1
                v = SparseArray.fromlist(v)
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
        "Fitness function in the training set"
        v.fitness = -self._ytr.SSE(v.hy * self._mask)

    def mask_vs(self):
        """Procedure performed in classification to compute
        more efficiently BER in the validation set"""
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
        """Fitness function in the validation set
        In classification it uses BER"""
        if self._classifier:
            v.fitness_vs = -((self.y - v.hy.sign()).sign().fabs() *
                             self._mask_vs).sum()
        else:
            m = (self._mask - 1).fabs()
            v.fitness_vs = -(self.y * m).SSE(v.hy * m)

    def convert_features(self, v):
        return Model.convert_features(v)

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

    def _random_leaf(self, var):
        v = Variable(var, ytr=self._ytr, mask=self._mask)
        if not v.eval(self.X):
            return None
        if not v.isfinite():
            return None
        if not self.set_fitness(v):
            return None
        return v

    def random_leaf(self):
        "Returns a random variable with the associated weight"
        for i in range(10):
            var = np.random.randint(self.nvar)
            v = self._random_leaf(var)
            if v is None:
                continue
            return v
        raise RuntimeError("Could not find a suitable random leaf")

    def _random_offspring(self, func, args):
        f = func(args, ytr=self._ytr, mask=self._mask)
        if not f.eval(self.population.hist):
            return None
        if not f.isfinite():
            return None
        if not self.set_fitness(f):
            return None
        return f

    def random_offspring(self):
        "Returns an offspring with the associated weight(s)"
        for i in range(10):
            # self._logger.debug('Init random offspring %s' % i)
            func = self.function_set
            func = func[np.random.randint(len(func))]
            # self._logger.debug('Func %s' % func)
            args = []
            for j in range(func.nargs):
                k = self.population.tournament()
                while k in args:
                    k = self.population.tournament()
                args.append(k)
            # self._logger.debug('Args %s' % args)
            args = map(lambda x: self.population.population[x].position, args)
            f = self._random_offspring(func, args)
            if f is None:
                # self._logger.debug('Random offspring %s is None' % i)
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

    def set_fitness(self, v):
        """Set the fitness to a new node.
        Returns false in case fitness is not finite"""
        self.fitness(v)
        if not np.isfinite(v.fitness):
            return False
        if self._tr_fraction < 1:
            self.fitness_vs(v)
            if not np.isfinite(v.fitness_vs):
                return False
        return True

    def create_population(self):
        "Create the initial population"
        self.population_instance()
        vars = np.arange(len(self.X))
        np.random.shuffle(vars)
        vars = vars.tolist()
        while self.population.popsize < self.popsize:
            if len(vars):
                v = self._random_leaf(vars.pop())
                if v is None:
                    continue
            else:
                func = self.function_set
                func = func[np.random.randint(len(func))]
                args = []
                for j in range(func.nargs):
                    psize = len(self.population.population)
                    args.append(np.random.randint(psize))
                args = map(lambda x: self.population.population[x].position,
                           args)
                v = self._random_offspring(func, args)
                if v is None:
                    continue
            self.population.add(v)

    def stopping_criteria(self):
        "Test whether the stopping criteria has been achieved."
        if self.generations < np.inf:
            inds = self.popsize * self.generations
            flag = inds <= len(self.population.hist)
        else:
            flag = False
        if flag:
            return True
        if self._tr_fraction < 1:
            if self.population.estopping.fitness_vs == 0:
                return True
        esr = self._early_stopping_rounds
        if self._tr_fraction < 1 and esr is not None:
            position = self.population.estopping.position
            if position < self.popsize:
                position = self.popsize
            return (len(self.population.hist) -
                    position) > esr
        return flag

    def nclasses(self, v):
        "Number of classes of v, also sets the labes"
        if not self._classifier:
            return 0
        if not isinstance(v, np.ndarray):
            v = v.tonparray()
        self._labels = np.unique(v)
        return self._labels.shape[0]

    def fit(self, X, y, test_set=None):
        "Evolutive process"
        self.X = X
        if self.nclasses(y) > 2:
            self._multiclass = True
            return self.multiclass(X, y, test_set=test_set)
        self.y = y
        if test_set is not None:
            self.Xtest = test_set
        self._logger.info("Starting evolution")
        self.create_population()
        while not self.stopping_criteria():
            a = self.random_offspring()
            self.population.replace(a)
        return self

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        trace_map = {}
        self._trace(n, trace_map)
        s = trace_map.keys()
        s.sort()
        return s

    def _trace(self, n, trace_map):
        if n.position in trace_map:
            return
        else:
            trace_map[n.position] = 1
        if isinstance(n, Function):
            if isinstance(n.variable, types.ListType):
                map(lambda x: self._trace(self.population.hist[x], trace_map),
                    n.variable)
            else:
                self._trace(self.population.hist[n.variable], trace_map)

    def model(self, v=None):
        "Returns the model of node v"
        if self._multiclass:
            models = map(lambda gp: gp.model(v=v),
                         self._multiclass_instances)
            return Models(models, labels=self._labels)
        if v is None:
            v = self.population.estopping
        hist = self.population.hist
        trace = self.trace(v)
        m = Model(trace, hist, labels=self._labels)
        return m

    def decision_function(self, v=None, X=None):
        "Decision function i.e. the raw data of the prediction"
        m = self.model(v=v)
        return m.decision_function(X)

    def predict(self, v=None, X=None):
        """In classification this returns the classes, in
        regression it is equivalent to the decision function"""
        m = self.model(v=v)
        return m.predict(X)
