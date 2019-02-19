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
from SparseArray import SparseArray
from .node import Variable
from .node import Add, Mul, Div, Fabs, Exp, Sqrt, Sin, Cos, Log1p
from .node import Sq, Min, Max
from .node import Atan2, Hypot, Acos, Asin, Atan, Tan, Cosh, Sinh
from .node import Tanh, Acosh, Asinh, Atanh, Expm1, Log, Log2, Log10
from .node import Lgamma, Sign, Ceil, Floor, NaiveBayes, NaiveBayesMN
from .node import Centroid
from .model import Model
from .population import SteadyState
from .utils import tonparray
from .function_selection import FunctionSelection
from .naive_bayes import NaiveBayes as NB
from .bagging_fitness import BaggingFitness
import time
import importlib
import inspect


class EvoDAG(object):
    def __init__(self, generations=np.inf, popsize=10000,
                 seed=0, tournament_size=2,
                 early_stopping_rounds=-1,
                 function_set=[Add, Mul, Div, Fabs,
                               Exp, Sqrt, Sin, Cos, Log1p,
                               Sq, Min, Max, Atan2, Hypot, Acos, Asin, Atan,
                               Tan, Cosh, Sinh, Tanh, Acosh, Asinh, Atanh,
                               Expm1, Log, Log2, Log10, Lgamma, Sign,
                               Ceil, Floor, NaiveBayes, NaiveBayesMN, Centroid],
                 tr_fraction=0.5, population_class=SteadyState,
                 negative_selection=True,
                 number_tries_feasible_ind=30, time_limit=None,
                 unique_individuals=True, classifier=True,
                 labels=None, all_inputs=False, random_generations=0,
                 fitness_function='BER', orthogonal_selection=False, orthogonal_dot_selection=False,
                 min_density=0.8, multiple_outputs=False, function_selection=True,
                 fs_tournament_size=2, finite=True, pr_variable=0.33,
                 share_inputs=False, input_functions=None, F1_index=-1,
                 use_all_vars_input_functions=False, remove_raw_inputs=True,
                 probability_calibration=None, **kwargs):
        self._remove_raw_inputs = remove_raw_inputs
        self._fitness_function = fitness_function
        self._bagging_fitness = BaggingFitness(base=self)
        generations = np.inf if generations is None else generations
        self._pr_variable = pr_variable
        self._share_inputs = share_inputs
        self._finite = finite
        self._generations = generations
        self._popsize = popsize
        self._classifier = classifier
        self._number_tries_feasible_ind = number_tries_feasible_ind
        self._unfeasible_counter = 0
        self._negative_selection = negative_selection
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
        self._function_selection = function_selection
        self._fs_tournament_size = fs_tournament_size
        density_safe = [k for k, v in enumerate(function_set) if v.density_safe]
        self._function_selection_ins = FunctionSelection(nfunctions=len(self._function_set),
                                                         seed=seed,
                                                         tournament_size=self._fs_tournament_size,
                                                         nargs=map(lambda x: x.nargs,
                                                                   function_set),
                                                         density_safe=density_safe)
        self._min_density = min_density
        self._function_selection_ins.min_density = self._min_density
        self._time_limit = time_limit
        self._init_time = time.time()
        self._random_generations = random_generations
        if not inspect.isclass(population_class):
            pop = importlib.import_module('EvoDAG.population')
            population_class = getattr(pop, population_class)
        self._set_input_functions(input_functions)
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
        self._F1_index = F1_index
        self._use_all_vars_input_functions = use_all_vars_input_functions
        self._probability_calibration = probability_calibration
        self._orthogonal_selection = orthogonal_selection
        self._orthogonal_dot_selection = orthogonal_dot_selection
        # orthogonal_selection is only implemented for classification
        # if self._orthogonal_selection:
        #     assert self._classifier
        self._extras = kwargs
        if self._time_limit is not None:
            self._logger.info('Time limit: %0.2f' % self._time_limit)

    def _set_input_functions(self, input_functions):
        if input_functions is not None:
            if not isinstance(input_functions, list):
                input_functions = [input_functions]
            r = []
            for f in input_functions:
                if not inspect.isclass(f):
                    _ = importlib.import_module('EvoDAG.node')
                    f = getattr(_, f)
                    r.append(f)
                else:
                    r.append(f)
            self._input_functions = r
        else:
            self._input_functions = input_functions

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
    def classifier(self):
        "Whether it is a classification problem or not"
        return self._classifier

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

    @y.setter
    def y(self, v):
        if isinstance(v, np.ndarray):
            v = SparseArray.fromlist(v)
        if self.classifier:
            self._multiple_outputs = True
            if self._labels is None:
                self.nclasses(v)
            return self._bagging_fitness.multiple_outputs_cl(v)
        elif self._multiple_outputs:
            return self._bagging_fitness.multiple_outputs_regression(v)
        elif self._tr_fraction < 1:
            for i in range(self._number_tries_feasible_ind):
                self._bagging_fitness.set_regression_mask(v)
                flag = self._bagging_fitness.test_regression_mask(v)
                if flag:
                    break
            if not flag:
                msg = "Unsuitable validation set (RSE: average equals zero)"
                raise RuntimeError(msg)
        else:
            self._mask = 1.0
        self._ytr = v * self._mask
        self._y = v

    @property
    def function_set(self):
        "List containing the functions used to create the individuals"
        return self._function_set

    @property
    def nvar(self):
        """Number of features or variables"""
        return self._nvar

    @nvar.setter
    def nvar(self, v):
        self._nvar = v

    @property
    def naive_bayes(self):
        try:
            return self._naive_bayes
        except AttributeError:
            if hasattr(self, '_y_klass'):
                self._naive_bayes = NB(mask=self._mask_ts, klass=self._y_klass,
                                       nclass=self._labels.shape[0])
            else:
                self._naive_bayes = None
        return self._naive_bayes

    @property
    def population(self):
        "Class containing the population and all the individuals generated"
        try:
            return self._p
        except AttributeError:
            self._p = self._population_class(base=self,
                                             tournament_size=self._tournament_size,
                                             classifier=self.classifier,
                                             labels=self._labels,
                                             es_extra_test=self.es_extra_test,
                                             popsize=self._popsize,
                                             random_generations=self._random_generations,
                                             negative_selection=self._negative_selection)
            return self._p

    @property
    def init_popsize(self):
        if self._all_inputs:
            return self._init_popsize
        return self.popsize

    def es_extra_test(self, v):
        """This function is called from population before setting
        the early stopping individual and after the comparisons with
        the validation set fitness"""
        return True

    def _random_leaf(self, var):
        v = Variable(var, ytr=self._ytr, finite=self._finite, mask=self._mask,
                     naive_bayes=self.naive_bayes)
        if not v.eval(self.X):
            return None
        if not v.isfinite():
            return None
        if not self._bagging_fitness.set_fitness(v):
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
        f = func(args, ytr=self._ytr, naive_bayes=self.naive_bayes,
                 finite=self._finite, mask=self._mask)
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
        if not self._bagging_fitness.set_fitness(f):
            return self.unfeasible_offspring()
        return f

    def _get_args_orthogonal(self, first):
        vars = self.population.random()
        pop = self.population.population
        score = self._bagging_fitness.score
        if self.classifier:
            fit = [(k, score.accuracy(first, SparseArray.argmax(pop[x].hy),
                                      self._mask_ts.index)[0]) for k, x in enumerate(vars)]
        else:
            fit = [(k, score.accuracy(first, pop[x].hy.sign() + 1,
                                      self._mask.index)[0]) for k, x in enumerate(vars)]
        fit = min(fit, key=lambda x: x[1])
        index = fit[0]
        return vars[index]

    def get_args_orthogonal(self, func):
        first = self.population.tournament()
        args = {first: 1}
        if self.classifier:
            first = SparseArray.argmax(self.population.population[first].hy)
        else:
            first = self.population.population[first].hy.sign() + 1.0
        res = []
        sel = self._get_args_orthogonal
        n_tries = self._number_tries_unique_args
        for j in range(func.nargs - 1):
            k = sel(first)
            for _ in range(n_tries):
                if k not in args:
                    args[k] = 1
                    res.append(k)
                    break
                else:
                    k = sel(first)
        if len(res) < func.min_nargs:
            return None
        return res

    def _get_args_orthogonal_dot(self, first):    
        vars = self.population.random()
        pop = self.population.population
        if self.classifier:
            prod = [(k,SparseArray.dot(pop[x].hy[0].mul(self._mask_ts),first)) for k,x in enumerate(vars)]
        else:
            prod = [(k,SparseArray.dot(pop[x].hy[0].mul(self._mask),first)) for k,x in enumerate(vars)]
        prod = min(prod, key=lambda x: x[1])
        index = prod[0]
        return vars[index]
    
    def get_args_orthogonal_dot(self, func):
        first = self.population.tournament()
        args = {first: 1}
        if self.classifier:
            first = self.population.population[first].hy[0].mul(self._mask_ts)
        else:
            first = self.population.population[first].hy[0].mul(self._mask)
        res = []
        sel = self._get_args_orthogonal_dot
        n_tries = self._number_tries_unique_args
        for j in range(func.nargs - 1):
            k = sel(first)
            for _ in range(n_tries):
                if k not in args:
                    args[k] = 1
                    res.append(k)
                    break
                else:
                    k = sel(first)
        try:
            min_nargs = func.min_nargs
        except AttributeError:
            min_nargs = func.nargs
        if len(res) < min_nargs:
            return None
        return res
    
    def get_unique_args(self, func):
        args = {}
        res = []
        p_tournament = self.population.tournament
        n_tries = self._number_tries_unique_args
        for j in range(func.nargs):
            k = p_tournament()
            for _ in range(n_tries):
                if k not in args:
                    args[k] = 1
                    res.append(k)
                    break
                else:
                    k = p_tournament()
        if len(res) < func.min_nargs:
            return None
        return res

    def get_args(self, func):
        args = []
        if self._orthogonal_selection and func.orthogonal_selection:
            return self.get_args_orthogonal(func)
        if self._orthogonal_dot_selection and func.orthogonal_selection:
            return self.get_args_orthogonal_dot(func)
        if func.unique_args:
            return self.get_unique_args(func)
        try:
            min_nargs = func.min_nargs
        except AttributeError:
            min_nargs = func.nargs
        p_tournament = self.population.tournament
        for j in range(func.nargs):
            k = p_tournament()
            for _ in range(self._number_tries_unique_args):
                if k not in args:
                    break
                else:
                    k = p_tournament()
            args.append(k)
        if len(args) < min_nargs:
            return None
        return args

    def random_offspring(self):
        "Returns an offspring with the associated weight(s)"
        function_set = self.function_set
        function_selection = self._function_selection_ins
        function_selection.density = self.population.density
        function_selection.unfeasible_functions.clear()
        for i in range(self._number_tries_feasible_ind):
            if self._function_selection:
                func_index = function_selection.tournament()
            else:
                func_index = function_selection.random_function()
            func = function_set[func_index]
            args = self.get_args(func)
            if args is None:
                continue
            args = [self.population.population[x].position for x in args]
            f = self._random_offspring(func, args)
            if f is None:
                function_selection.unfeasible_functions.add(func_index)
                continue
            function_selection[func_index] = f.fitness
            return f
        raise RuntimeError("Could not find a suitable random offpsring")

    def create_population(self):
        "Create the initial population"
        self.population.create_population()

    def stopping_criteria_tl(self):
        if self._time_limit is not None:
            if time.time() - self._init_time > self._time_limit:
                return True
        return False

    def stopping_criteria(self):
        "Test whether the stopping criteria has been achieved."
        if self.stopping_criteria_tl():
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
        if not self.classifier:
            return 0
        if isinstance(v, list):
            self._labels = np.arange(len(v))
            return
        if not isinstance(v, np.ndarray):
            v = tonparray(v)
        self._labels = np.unique(v)
        return self._labels.shape[0]

    def replace(self, a):
        "Replace an individual in the population with individual a"
        self.population.replace(a)

    def shuffle_tr2ts(self):
        L = []
        for x in self.X:
            l = np.array(x.hy.full_array())
            np.random.shuffle(l)
            L.append(l)
        return np.array(L).T

    def fit(self, X, y, test_set=None):
        """Evolutive process"""
        self._init_time = time.time()
        self.X = X
        if self._popsize == "nvar":
            self._popsize = self.nvar + len(self._input_functions)
        if isinstance(test_set, str) and test_set == 'shuffle':
            test_set = self.shuffle_tr2ts()
        nclasses = self.nclasses(y)
        if self.classifier and self._multiple_outputs:
            pass
        elif nclasses > 2:
            assert False
            self._multiclass = True
            return self.multiclass(X, y, test_set=test_set)
        self.y = y
        if test_set is not None:
            self.Xtest = test_set
        for _ in range(self._number_tries_feasible_ind):
            self._logger.info("Starting evolution")
            try:
                self.create_population()
                if self.stopping_criteria_tl():
                    break
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self._logger.info("Population created (hist: %s)" % len(self.population.hist))
            if len(self.population.hist) >= self._tournament_size:
                break
        if len(self.population.hist) == 0:
            raise RuntimeError("Could not find a suitable individual")
        if len(self.population.hist) < self._tournament_size:
            self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
            return self
        if self._remove_raw_inputs:
            for x in range(self.nvar):
                self._X[x] = None
        while not self.stopping_criteria():
            try:
                a = self.random_offspring()
            except RuntimeError as err:
                self._logger.info("Done evolution (RuntimeError (%s), hist: %s)" % (err, len(self.population.hist)))
                return self
            self.replace(a)
        self._logger.info("Done evolution (hist: %s)" % len(self.population.hist))
        return self

    def trace(self, n):
        "Restore the position in the history of individual v's nodes"
        return self.population.trace(n)

    def model(self, v=None):
        "Returns the model of node v"
        return self.population.model(v=v)

    def decision_function(self, v=None, X=None):
        "Decision function i.e. the raw data of the prediction"
        m = self.model(v=v)
        return m.decision_function(X)

    def predict(self, v=None, X=None):
        """In classification this returns the classes, in
        regression it is equivalent to the decision function"""
        if X is None:
            X = v
            v = None
        m = self.model(v=v)
        return m.predict(X)

    @classmethod
    def init(cls, params_fname=None, seed=None, classifier=True, **kwargs):
        import os
        from .utils import RandomParameterSearch
        import json
        import gzip
        if params_fname is None:
            if classifier:
                kw = os.path.join(os.path.dirname(__file__),
                                  'conf', 'default_parameters.json')
            else:
                kw = os.path.join(os.path.dirname(__file__),
                                  'conf', 'default_parameters_r.json')
        else:
            kw = params_fname
        if not isinstance(kw, dict):
            if kw.endswith('.gz'):
                func = gzip.open
            else:
                func = open
            with func(kw, 'rb') as fpt:
                try:
                    d = fpt.read()
                    kw = json.loads(str(d, encoding='utf-8'))
                except TypeError:
                    kw = json.loads(d)
            if isinstance(kw, list):
                kw = kw[0]
        if seed is not None:
            kw['seed'] = seed
        kw.update(kwargs)
        kw = RandomParameterSearch.process_params(kw)
        return cls(**kw)


RGP = EvoDAG
RootGP = RGP
