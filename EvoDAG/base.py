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
from . import array_utils as au
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
from .bagging_fitness import BaggingFitness, mask_mul
import time
import importlib
import inspect


class EvoDAG(object):
    """The EvoDAG search engine, also exported as ``RGP`` and ``RootGP``.

    It runs one evolution: it builds an initial population of simple
    individuals, creates one offspring at a time with the least-squares
    crossover (PrXO), and stops by early stopping on a validation split. After
    ``fit`` call ``model`` to obtain the fitted model. The class with the
    scikit-learn-like interface that wraps a single model is
    ``EvoDAG.model.EvoDAG`` (a different class with the same name), and the
    ensemble is ``EvoDAG.EvoDAGE``.

    Most users create the engine with ``init``, which reads the default
    parameters from ``EvoDAG/conf/default_parameters.json`` (or
    ``default_parameters_r.json``), instead of calling the constructor, whose
    own defaults differ from that file.

    Parameters
    ----------
    generations : int or float, optional
        Maximum number of generations; one generation is ``popsize``
        offspring. ``None`` or ``numpy.inf`` (the default) means no limit.
        Key ``generations`` of the default parameters (``null``).
    popsize : int, optional
        Number of individuals in the population (default 10000). Key
        ``popsize`` (4000). The string ``"nvar"`` sets it to the number of
        inputs plus the number of input functions.
    seed : int, optional
        Seed of the pseudo-random numbers (default 0). Key ``seed``.
    tournament_size : int, optional
        Individuals drawn in each tournament, for the arguments and for the
        individual that leaves (default 2). Key ``tournament_size``.
    early_stopping_rounds : int, optional
        The run stops when this many offspring were created without
        improving the early stopping individual (the best on the validation
        split). A negative value, the default, means ``popsize``. Key
        ``early_stopping_rounds`` (4000).
    function_set : list, optional
        Classes of ``EvoDAG.node`` that may be used to create offspring
        (default: all of them). ``init`` builds it from the function keys
        of the parameters (``Add``, ``Mul``, ...).
    tr_fraction : float, optional
        Proportion of the data used for training when it is split into the
        training and the validation part (default 0.5). With 1 there is no
        validation set and ``generations`` must be finite. Key
        ``tr_fraction``.
    population_class : class or str, optional
        ``SteadyState`` (default) or ``Generational``, or its name. Key
        ``population_class``.
    negative_selection : bool, optional
        If true (default) the individual replaced by an offspring is the
        worst of a tournament; if false it is chosen at random.
    number_tries_feasible_ind : int, optional
        Attempts to find a valid individual before giving up (default 30).
    time_limit : float, optional
        Seconds after which the evolution stops (default ``None``, no
        limit). Key ``time_limit`` (3600).
    unique_individuals : bool, optional
        Reject offspring that repeat a function applied to the same
        arguments (default true). Key ``unique_individuals`` (false).
    classifier : bool, optional
        Classification if true (default), regression otherwise. Key
        ``classifier``.
    labels : array_like, optional
        The class labels; by default they are taken from ``y`` in ``fit``.
    all_inputs : bool, optional
        Create initial individuals until every input has been used instead
        of until the population has ``popsize`` individuals (default
        false). Key ``all_inputs`` (true).
    random_generations : int, optional
        Number of generations in which the arguments are drawn at random
        instead of by tournament on fitness (default 0). Key
        ``random_generations`` (1).
    fitness_function : str, optional
        Fitness in classification (default ``"BER"``); the options are
        described in the Parameters page of the user guide. Ignored in
        regression. Key ``fitness_function``.
    orthogonal_selection : bool, optional
        Choose the arguments of the sum, Naive Bayes and Centroid by the
        smallest agreement with the first argument (default false). Not in
        the default parameters.
    orthogonal_dot_selection : bool, optional
        Choose those arguments by the smallest dot product with the first
        argument (default false). Not in the default parameters.
    min_density : float, optional
        When the average fraction of non-zero values in the population is
        below it, only density safe functions are used (default 0.8). Key
        ``min_density``.
    multiple_outputs : bool, optional
        Whether the individuals have several outputs (default false); it is
        always on in classification. Key ``multiple_outputs``.
    function_selection : bool, optional
        Choose the function of each offspring by a tournament on the
        average fitness of the offspring it produced, instead of at random
        (default true). Key ``function_selection``.
    fs_tournament_size : int, optional
        Size of that tournament (default 2). Key ``fs_tournament_size``.
    finite : bool, optional
        Replace the non-finite values of the inputs by zero (default true).
        Key ``finite``.
    pr_variable : float, optional
        Probability that an initial individual is a single input variable
        and not an input function (default 0.33). Key ``pr_variable`` (0.9).
    share_inputs : bool, optional
        Kept for compatibility; it has no effect (default false). Key
        ``share_inputs``.
    input_functions : list, optional
        Classes (or names) among ``NaiveBayes``, ``NaiveBayesMN``,
        ``Centroid`` and ``MultipleVariables`` used to create the initial
        individuals that combine inputs (default ``None``). Key
        ``input_functions``.
    F1_index : int, optional
        Class used by ``fitness_function="F1"``; -1 (default) is the
        smallest class.
    use_all_vars_input_functions : bool, optional
        Create first one individual per input function over all the
        variables (default false). Key ``use_all_vars_input_functions``
        (true).
    remove_raw_inputs : bool, optional
        Free the training inputs once the initial population exists
        (default true).
    probability_calibration : class, optional
        A class with the ``fit``/``predict_proba`` interface of
        scikit-learn used to calibrate probabilities (default ``None``).
    **kwargs
        Ignored; they are kept in the instance. This is how keys of the
        parameters that are not arguments (for example ``Argmax``) are
        accepted.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import RGP
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
    >>> model = engine.fit(X, y).model()
    >>> bool((model.predict(X) == y).mean() > 0.8)
    True
    """

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
        """Parameters used to initialize the class.

        Returns
        -------
        dict
            The name and the current value of each argument of the
            constructor, so that ``clone`` can create an equivalent engine.

        Examples
        --------
        >>> from EvoDAG import RGP
        >>> params = RGP(popsize=20, seed=3).get_params()
        >>> params["popsize"], params["seed"]
        (20, 3)
        """
        import inspect
        a = inspect.getfullargspec(self.__init__).args
        out = dict()
        for key in a[1:]:
            value = getattr(self, "_%s" % key, None)
            out[key] = value
        return out

    def clone(self):
        """Clone the class without the population.

        Returns
        -------
        EvoDAG
            A new engine, not fitted, created with ``get_params``.
        """
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
            v = np.array(v, dtype=float)
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
        self._ytr = mask_mul(v, self._mask)
        self._y = v

    @property
    def _mask_ts(self):
        "Mask of the training set (used by the metrics) and its positions"
        return self.__mask_ts

    @_mask_ts.setter
    def _mask_ts(self, v):
        self.__mask_ts = v
        self._mask_ts_index = np.flatnonzero(v).astype(np.uint32)

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
        """Naive Bayes statistics of the training part, shared by the classifier nodes.

        ``None`` in regression.
        """
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
        """Number of individuals created before the evolution proper starts."""
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
        """Count a rejected offspring (it also counts for early stopping) and return ``None``."""
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
            fit = [(k, score.accuracy(first, au.argmax(pop[x].hy),
                                      self._mask_ts_index)[0]) for k, x in enumerate(vars)]
        else:
            index = np.flatnonzero(self._mask).astype(np.uint32)
            fit = [(k, score.accuracy(first, au.sign(pop[x].hy) + 1,
                                      index)[0]) for k, x in enumerate(vars)]
        fit = min(fit, key=lambda x: x[1])
        index = fit[0]
        return vars[index]

    def get_args_orthogonal(self, func):
        """Select the arguments of ``func`` by agreement with the first one.

        The first argument is chosen with the population tournament; each
        other one is the individual, among ``tournament_size`` drawn at
        random, whose predicted labels agree least with those of the first
        argument on the training part. Returns the list of positions in the
        population of the other arguments, or ``None`` if there are fewer
        than ``func.min_nargs``.
        """
        first = self.population.tournament()
        args = {first: 1}
        if self.classifier:
            first = au.argmax(self.population.population[first].hy)
        else:
            first = au.sign(self.population.population[first].hy) + 1.0
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

    def _get_args_orthogonal_dot(self, first_hy):    
        vars = self.population.random()
        pop = self.population.population
        if self.classifier:
            mask = self._mask_ts
        else:
            mask = self._mask
            
        if isinstance(first_hy,list):
            prod = []
            for k,x in enumerate(vars):  
                pvalue = 0
                for i in range(len(first_hy)):
                    pvalue+= au.dot(au.mul(pop[x].hy[i], mask), first_hy[i])
                prod.append( (k,pvalue) )
        else:
            prod = [(k, au.dot(au.mul(pop[x].hy, mask), first_hy)) for k, x in enumerate(vars)]
        
        prod = min(prod, key=lambda x: x[1])
        index = prod[0]
        return vars[index]
    
    def get_args_orthogonal_dot(self, func):
        """Select the arguments of ``func`` by the smallest dot product with the first one.

        Same procedure as ``get_args_orthogonal`` but the measure is the
        dot product of the masked outputs. Returns the list of positions of
        the other arguments, or ``None`` if there are too few.
        """
        first = self.population.tournament()
        args = {first: 1}
        if self.classifier:
            mask = self._mask_ts
        else:
            mask = self._mask
        if isinstance(self.population.population[first].hy,list):
            first_hy = []
            for i in range(len(self.population.population[first].hy)):
                first_hy.append(au.mul(self.population.population[first].hy[i], mask))
        else:
            first_hy = au.mul(self.population.population[first].hy, mask)
            
        res = []
        sel = self._get_args_orthogonal_dot
        n_tries = self._number_tries_unique_args
        for j in range(func.nargs - 1):
            k = sel(first_hy)
            for _ in range(n_tries):
                if k not in args:
                    args[k] = 1
                    res.append(k)
                    break
                else:
                    k = sel(first_hy)
        try:
            min_nargs = func.min_nargs
        except AttributeError:
            min_nargs = func.nargs
        if len(res) < min_nargs:
            return None
        return res
    
    def get_unique_args(self, func):
        """Select up to ``func.nargs`` different arguments by tournament.

        Returns the list of positions in the population, or ``None`` when
        fewer than ``func.min_nargs`` different ones were found.
        """
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
        """Select the arguments of function ``func`` according to the selection parameters.

        It uses the agreement or the dot product heuristic when they are on
        and ``func`` supports them, the tournament with distinct arguments
        for functions that require them, and the plain tournament otherwise.
        Returns a list of positions in the population or ``None``.
        """
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
        """Whether ``time_limit`` seconds have passed since ``fit`` started."""
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
        """Test set made by shuffling each input of the training set independently.

        It is what ``fit`` uses as test set when ``test_set="shuffle"``.
        """
        L = []
        for x in self.X:
            l = np.array(x.hy)
            np.random.shuffle(l)
            L.append(l)
        return np.array(L).T

    def fit(self, X, y, test_set=None):
        """Run the evolution.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example. A list with one array per input
            (column) is also accepted.
        y : array_like
            In classification the class of each example (numbers); in
            regression the real target. With several outputs a list with
            one array per output.
        test_set : array_like or str, optional
            Inputs to be evaluated along the evolution so that predictions
            of the models are available without evaluating them again;
            ``"shuffle"`` creates one from the training inputs.

        Returns
        -------
        EvoDAG
            The engine itself. Use ``model`` to obtain the fitted model.

        Raises
        ------
        RuntimeError
            If no suitable individual can be found, or if there is no
            validation split and no limit of generations.

        Examples
        --------
        >>> import numpy as np
        >>> from EvoDAG import RGP
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
        >>> engine.fit(X, y) is engine
        True
        >>> len(engine.population.hist) > 0
        True
        """
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
        """Positions in the history of the nodes that individual ``n`` depends on, sorted.

        Parameters
        ----------
        n : EvoDAG.node.Variable
            An individual of the history.
        """
        return self.population.trace(n)

    def model(self, v=None):
        """Return the model of node ``v``.

        Parameters
        ----------
        v : EvoDAG.node.Variable, optional
            An individual found in ``population.hist``. By default the
            early stopping individual, the best on the validation split.

        Returns
        -------
        EvoDAG.model.Model
            The nodes needed to predict, with their coefficients.

        Examples
        --------
        >>> import numpy as np
        >>> from EvoDAG import RGP, Model
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
        >>> isinstance(engine.fit(X, y).model(), Model)
        True
        """
        return self.population.model(v=v)

    def decision_function(self, v=None, X=None):
        """Decision function, i.e. the raw outputs of the model.

        Parameters
        ----------
        v : EvoDAG.node.Variable, optional
            Individual whose model is used; the early stopping one by default.
        X : array_like, optional
            Inputs to evaluate. If ``None`` the output on the test set given
            to ``fit`` is returned.

        Returns
        -------
        list or numpy.ndarray
            In classification a list with one array per class.

        Examples
        --------
        >>> import numpy as np
        >>> from EvoDAG import RGP
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
        >>> df = engine.fit(X, y).decision_function(X=X)
        >>> len(df), df[0].shape
        (2, (100,))
        """
        m = self.model(v=v)
        return m.decision_function(X)

    def predict(self, v=None, X=None):
        """Predict ``X``: the classes in classification, the values in regression.

        It can be called as ``predict(X)`` or as ``predict(v, X)``.

        Parameters
        ----------
        v : EvoDAG.node.Variable or array_like, optional
            Individual whose model is used; if ``X`` is not given this is
            taken as the inputs.
        X : array_like, optional
            Inputs to predict.

        Returns
        -------
        numpy.ndarray
            One value per example, in the labels of the training data.

        Examples
        --------
        >>> import numpy as np
        >>> from EvoDAG import RGP
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> engine = RGP.init(popsize=50, early_stopping_rounds=50, seed=0)
        >>> hy = engine.fit(X, y).predict(X)
        >>> sorted(set(hy.tolist())) == [0, 1]
        True
        """
        if X is None:
            X = v
            v = None
        m = self.model(v=v)
        return m.predict(X)

    @classmethod
    def init(cls, params_fname=None, seed=None, classifier=True, **kwargs):
        """Create an engine from a file of parameters.

        Parameters
        ----------
        params_fname : str or dict, optional
            JSON file (also gzip compressed when its name ends in ``.gz``)
            or dictionary with the parameters. A file that contains a list,
            as the one written by ``EvoDAG-params``, uses its first element.
            By default ``EvoDAG/conf/default_parameters.json`` in
            classification and ``default_parameters_r.json`` in regression.
        seed : int, optional
            Overrides the seed of the parameters.
        classifier : bool, optional
            Selects the default parameters when ``params_fname`` is not
            given (default true, classification).
        **kwargs
            Parameters that override those of the file. Keys that are the
            name of a function (``Add``, ``Mul``, ...) switch it on or off
            (boolean) or set its number of arguments (integer, 0 removes it).

        Returns
        -------
        EvoDAG
            An engine that is not fitted.

        Examples
        --------
        >>> from EvoDAG import RGP
        >>> engine = RGP.init(popsize=20, Mul=0, seed=2)
        >>> engine.popsize
        20
        >>> "Mul" in [f.__name__ for f in engine.function_set]
        False
        """
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
