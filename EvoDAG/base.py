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
from .sparse_array import SparseArray
from .node import Variable
from .node import Add, Mul, Div, Fabs, Exp, Sqrt, Sin, Cos, Ln
from .node import Sq, Sigmoid, If, Min, Max
from .model import Model, Models
from .population import SteadyState
import time
import importlib
import inspect


class EvoDAG(object):
    def __init__(self, generations=np.inf, popsize=10000,
                 seed=0,
                 tournament_size=2,
                 early_stopping_rounds=-1,
                 function_set=[Add, Mul, Div, Fabs,
                               Exp, Sqrt, Sin, Cos, Ln,
                               Sq, Sigmoid, If, Min, Max],
                 tr_fraction=0.8,
                 population_class=SteadyState,
                 number_tries_feasible_ind=30,
                 time_limit=None,
                 unique_individuals=True,
                 classifier=True,
                 labels=None, all_inputs=False,
                 random_generations=0,
                 multiple_outputs=False,
                 **kwargs):
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._number_tries_feasible_ind = number_tries_feasible_ind
        self._unfeasible_counter = 0
        self._number_tries_unique_args = 3
        self._tr_fraction = tr_fraction
        if early_stopping_rounds is not None and early_stopping_rounds < 0:
            early_stopping_rounds = popsize
        self._early_stopping_rounds = early_stopping_rounds
        self._tournament_size = tournament_size
        self._seed = seed
        self._labels = labels
        self._multiclass = False
        self._function_set = function_set
        self._time_limit = time_limit
        self._init_time = time.time()
        self._random_generations = random_generations
        if not inspect.isclass(population_class):
            pop = importlib.import_module('EvoDAG.population')
            population_class = getattr(pop, population_class)
        self._population_class = population_class
        np.random.seed(self._seed)
        self._unique_individuals = unique_individuals
        self._unique_individuals_set = set()
        self._logger = logging.getLogger('EvoDAG')
        self._all_inputs = all_inputs
        if self._generations == np.inf and tr_fraction == 1:
            raise RuntimeError("Infinite evolution, set generations\
            or tr_fraction < 1 ")
        self._multiple_outputs = multiple_outputs
        self._extras = kwargs

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
    def signature(self):
        "Instance file name"
        kw = self.get_params()
        keys = sorted(kw.keys())
        l = []
        for k in keys:
            n = k[0] + k[-1]
            v = kw[k]
            if k == 'function_set':
                v = "_".join([x.__name__[0] +
                              x.__name__[-1] +
                              str(x.nargs) for x in kw[k]])
            elif k == 'population_class':
                v = kw[k].__name__
            else:
                v = str(v)
            l.append('{0}_{1}'.format(n, v))
        return '-'.join(l)

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
        return [x.hy_test for x in self.X]

    @Xtest.setter
    def Xtest(self, v):
        Model.convert_features_test_set(self.X, v)

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
        cnt = min([(v == x).sum() for x in a]) * self._tr_fraction
        cnt = int(round(cnt))
        for i in a:
            index = np.where(v == i)[0]
            np.random.shuffle(index)
            mask[index[:cnt]] = True
        self._mask = SparseArray.fromlist(mask)
        return SparseArray.fromlist(v)

    def set_regression_mask(self, v):
        """Computes the mask used to create the training and validation set"""
        index = np.arange(v.size())
        np.random.shuffle(index)
        ones = np.ones(v.size())
        ones[index[int(self._tr_fraction * v.size()):]] = 0
        self._mask = SparseArray.fromlist(ones)

    def test_regression_mask(self, v):
        """Test whether the average prediction is different than zero"""
        m = (self._mask - 1).fabs()
        x = v * m
        b = (x - x.sum() / x.size()).sq().sum()
        return b != 0

    def multiclass(self, X, v, test_set=None):
        "Performing One vs All multiclass classification"
        if not isinstance(v, np.ndarray):
            v = v.tonparray()
        mask = None
        for i in self._labels:
            _ = np.zeros_like(v, dtype=np.bool)
            _[v == i] = True
            mask = np.vstack((mask, _)) if mask is not None else _
        time_limit = self._time_limit
        if time_limit is not None:
            self._time_limit = time_limit / float(len(self._labels))
        self._multiclass_instances = [self.clone() for x in mask]
        self._time_limit = time_limit
        for m, gp in zip(mask, self._multiclass_instances):
            y = np.zeros_like(m) - 1
            y[m] = 1
            gp.fit(X, y, test_set=test_set)
        return self

    def transform_to_mo(self, v):
        klass = self._labels
        y = np.empty((v.shape[0], klass.shape[0]))
        y.fill(-1)
        for i, k in enumerate(klass):
            mask = k == v
            y[mask, i] = 1
        return y
        
    def multiple_outputs_y(self, v):
        if isinstance(v, list):
            assert len(v) == self._labels.shape[0]
            v = np.array([x.tonparray() for x in v]).T
        else:
            v = v.tonparray()
            v = self.transform_to_mo(v)
        mask = []
        mask_vs = []
        ytr = []
        y = []
        for _v in v.T:
            _v = SparseArray.fromlist(_v)
            self.set_classifier_mask(_v)
            mask.append(self._mask)
            ytr.append(_v * self._mask)
            y.append(_v)
            self._y = _v
            self.mask_vs()
            mask_vs.append(self._mask_vs)
        self._ytr = ytr
        self._y = y
        self._mask = mask
        self._mask_vs = mask_vs

    @y.setter
    def y(self, v):
        if isinstance(v, np.ndarray):
            v = SparseArray.fromlist(v)
        if self._classifier and self._multiple_outputs:
            return self.multiple_outputs_y(v)
        elif self._classifier:
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
            for i in range(self._number_tries_feasible_ind):
                self.set_regression_mask(v)
                flag = self.test_regression_mask(v)
                if flag:
                    break
            if not flag:
                msg = "Unsuitable validation set (RSE: average equals zero)"
                raise RuntimeError(msg)
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
        if self._classifier:
            if self._multiple_outputs:
                f = [-ytr.SSE(hy * mask) for ytr, hy, mask in
                     zip(self._ytr, v.hy, self._mask)]
                v.fitness = np.mean(f)
            else:
                v.fitness = -self._ytr.SSE(v.hy * self._mask)
        else:
            v.fitness = -self._ytr.SAE(v.hy * self._mask)

    def mask_vs(self):
        """Procedure to perform, in classification,
        more efficiently BER in the validation set"""
        if not self._classifier:
            return
        if self._tr_fraction == 1:
            return
        m = ~ self._mask.tonparray().astype(np.bool)
        f = np.zeros(self._mask.size())
        y = self.y.tonparray()
        den = (y[m] == -1).sum()
        if den:
            f[y == -1] = 0.5 / den
        else:
            f[y == -1] = 0.5
        den = (y[m] == 1).sum()
        if den:
            f[y == 1] = 0.5 / den
        else:
            f[y == 1] = 0.5
        f[~m] = 0
        self._mask_vs = SparseArray.fromlist(f)

    def fitness_vs(self, v):
        """Fitness function in the validation set
        In classification it uses BER and RSE in regression"""
        if self._classifier:
            if self._multiple_outputs:
                f = [-((y - hy.sign()).sign().fabs() * mask_vs).sum() for
                     y, hy, mask_vs in zip(self.y, v.hy, self._mask_vs)]
                v.fitness_vs = np.mean(f)
            else:
                v.fitness_vs = -((self.y - v.hy.sign()).sign().fabs() *
                                 self._mask_vs).sum()
        else:
            m = (self._mask - 1).fabs()
            x = self.y * m
            y = v.hy * m
            a = (x - y).sq().sum()
            b = (x - x.sum() / x.size()).sq().sum()
            v.fitness_vs = -a / b

    def es_extra_test(self, v):
        """This function is called from population before setting
        the early stopping individual and after the comparisons with
        the validation set fitness"""
        return True

    def convert_features(self, v):
        return Model.convert_features(v)

    @property
    def nvar(self):
        """Number of features or variables"""
        return self._nvar

    @nvar.setter
    def nvar(self, v):
        self._nvar = v

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
        for i in range(self._number_tries_feasible_ind):
            var = np.random.randint(self.nvar)
            v = self._random_leaf(var)
            if v is None:
                continue
            return v
        raise RuntimeError("Could not find a suitable random leaf")

    def unique_individual(self, v):
        "Test whether v has not been explored during the evolution"
        return v not in self._unique_individuals_set

    def unfeasible_offspring(self):
        self._unfeasible_counter += 1
        return None

    def _random_offspring(self, func, args):
        f = func(args, ytr=self._ytr, mask=self._mask)
        if self._unique_individuals:
            sig = f.signature()
            if self.unique_individual(sig):
                self._unique_individuals_set.add(sig)
            else:
                return self.unfeasible_offspring()
        f.height = max([self.population.hist[x].height for x in args]) + 1
        if not f.eval(self.population.hist):
            return self.unfeasible_offspring()
        if not f.isfinite():
            return self.unfeasible_offspring()
        if not self.set_fitness(f):
            return self.unfeasible_offspring()
        return f

    def get_unique_args(self, func):
        args = []
        for j in range(func.nargs):
            k = self.population.tournament()
            for _ in range(self._number_tries_unique_args):
                if k not in args:
                    args.append(k)
                    break
                else:
                    k = self.population.tournament()
        if len(args) < func.min_nargs:
            return None
        return args

    def get_args(self, func):
        args = []
        if func.unique_args:
            return self.get_unique_args(func)
        for j in range(func.nargs):
            k = self.population.tournament()
            for _ in range(self._number_tries_unique_args):
                if k not in args:
                    break
                else:
                    k = self.population.tournament()
            args.append(k)
        return args

    def random_offspring(self):
        "Returns an offspring with the associated weight(s)"
        for i in range(self._number_tries_feasible_ind):
            # self._logger.debug('Init random offspring %s' % i)
            func = self.function_set
            func = func[np.random.randint(len(func))]
            # self._logger.debug('Func %s' % func)
            args = self.get_args(func)
            if args is None:
                continue
            # print(args)
            # self._logger.debug('Args %s' % args)
            args = [self.population.population[x].position for x in args]
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
        self._p = self._population_class(tournament_size=self._tournament_size,
                                         classifier=self._classifier,
                                         labels=self._labels,
                                         es_extra_test=self.es_extra_test,
                                         popsize=self._popsize,
                                         random_generations=self._random_generations)

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

    @property
    def init_popsize(self):
        if self._all_inputs:
            return self._init_popsize
        return self.popsize

    def create_population(self):
        "Create the initial population"
        self.population_instance()
        vars = np.arange(len(self.X))
        np.random.shuffle(vars)
        vars = vars.tolist()
        while (self._all_inputs or
               (self.population.popsize < self.popsize and
                not self.stopping_criteria())):
            if len(vars):
                v = self._random_leaf(vars.pop())
                if v is None:
                    continue
            elif self._all_inputs:
                self._init_popsize = self.population.popsize
                break
            else:
                func = self.function_set
                func = func[np.random.randint(len(func))]
                psize = len(self.population.population)
                args = np.arange(psize)
                np.random.shuffle(args)
                args = args[:func.nargs].tolist()
                for j in range(len(args), func.nargs):
                    args.append(np.random.randint(psize))
                args = [self.population.population[x].position for x in args]
                v = self._random_offspring(func, args)
                if v is None:
                    continue
            self.add(v)

    def stopping_criteria(self):
        "Test whether the stopping criteria has been achieved."
        if self._time_limit is not None:
            if time.time() - self._init_time > self._time_limit:
                return True
        if self.generations < np.inf:
            inds = self.popsize * self.generations
            flag = inds <= len(self.population.hist)
        else:
            flag = False
        if flag:
            return True
        est = self.population.estopping
        if self._tr_fraction < 1:
            if est is not None and est.fitness_vs == 0:
                return True
        esr = self._early_stopping_rounds
        if self._tr_fraction < 1 and esr is not None and est is not None:
            position = self.population.estopping.position
            if position < self.init_popsize:
                position = self.init_popsize
            return (len(self.population.hist) +
                    self._unfeasible_counter -
                    position) > esr
        return flag

    def nclasses(self, v):
        "Number of classes of v, also sets the labes"
        if not self._classifier:
            return 0
        if isinstance(v, list):
            self._labels = np.arange(len(v))
            return
        if not isinstance(v, np.ndarray):
            v = v.tonparray()
        self._labels = np.unique(v)
        return self._labels.shape[0]

    def add(self, a):
        "Add individual a to the population"
        self.population.add(a)
        if self.population.previous_estopping:
            self._unfeasible_counter = 0

    def replace(self, a):
        "Replace an individual in the population with individual a"
        self.population.replace(a)
        if self.population.previous_estopping:
            self._unfeasible_counter = 0

    def fit(self, X, y, test_set=None):
        "Evolutive process"
        self._init_time = time.time()
        self.X = X
        nclasses = self.nclasses(y)
        if self._classifier and self._multiple_outputs:
            pass
        elif nclasses > 2:
            self._multiclass = True
            return self.multiclass(X, y, test_set=test_set)
        self.y = y
        if test_set is not None:
            self.Xtest = test_set
        self._logger.info("Starting evolution")
        self.create_population()
        while not self.stopping_criteria():
            try:
                a = self.random_offspring()
            except RuntimeError:
                return self
            self.replace(a)
        return self

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        return self.population.trace(n)

    def model(self, v=None):
        "Returns the model of node v"
        if self._multiclass:
            models = [gp.model(v=v) for gp in self._multiclass_instances]
            return Models(models, labels=self._labels)
        return self.population.model(v=v)

    def decision_function(self, v=None, X=None):
        "Decision function i.e. the raw data of the prediction"
        m = self.model(v=v)
        return m.decision_function(X)

    def predict(self, v=None, X=None):
        """In classification this returns the classes, in
        regression it is equivalent to the decision function"""
        m = self.model(v=v)
        return m.predict(X)

RGP = EvoDAG
RootGP = RGP
