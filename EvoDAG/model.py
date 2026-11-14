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
from . import array_utils as au
from .node import Variable, Function
from .utils import tonparray
from multiprocessing import Pool
import logging
import gc
import os
import gzip
import pickle
import shutil
from time import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
LOGGER = logging.getLogger('EvoDAG')


def fit(X_y_evodag):
    """Fit one member of an ensemble; the argument is a tuple (used with ``Pool.imap_unordered``).

    The tuple is ``(X, y, test_set, parameters, tmpdir, init_time)``. It
    returns the fitted ``Model``, or ``None`` when there is no time left or
    the evolution fails. A model already stored in ``tmpdir`` is reused.
    """
    X, y, test_set, evodag, tmpdir, init_time = X_y_evodag
    if tmpdir is not None:
        seed = evodag['seed']
        output = os.path.join(tmpdir, '%s.evodag' % seed)
        if os.path.isfile(output):
            with gzip.open(output) as fpt:
                try:
                    return pickle.load(fpt)
                except Exception:
                    pass
    try:
        time_limit = evodag['time_limit']
        if time_limit is not None:
            evodag['time_limit'] = time_limit - (time() - init_time)
            if evodag['time_limit'] < 2:
                LOGGER.info('Not enough time (seed: %s) ' % evodag['seed'])
                return None
    except KeyError:
        pass
    try:
        evodag = EvoDAG(**evodag)
        evodag.fit(X, y, test_set=test_set)
    except RuntimeError:
        return None
    m = evodag.model
    gc.collect()
    if tmpdir is not None:
        with gzip.open(output, 'w') as fpt:
            pickle.dump(m, fpt)
    return m


def decision_function(model_X):
    """Evaluate the decision function of one model; the argument is ``(k, model, X)``.

    It returns ``[k, decision_function]`` so that results can be sorted
    after a parallel map.
    """
    k, model, X = model_X
    return [k, model.decision_function(X)]


def predict_proba(model_X):
    """Evaluate ``predict_proba`` of one model; the argument is ``(k, model, X)``.

    It returns ``[k, probabilities]`` so that results can be sorted after a
    parallel map.
    """
    k, model, X = model_X
    return [k, model.predict_proba(X)]


class Model(object):
    """A fitted model: the nodes needed to predict, with their coefficients.

    It is built from an individual (usually the early stopping individual)
    keeping only the nodes that individual depends on, in the order in which
    they are evaluated. A model is obtained with ``EvoDAG.EvoDAG.model``, as
    ``single_model.model`` after ``EvoDAG.model.EvoDAG.fit``, or as the
    elements of ``Ensemble.models``; it is not normally created directly.
    Use ``predict`` and ``decision_function`` with new inputs.

    Parameters
    ----------
    trace : list of int
        Positions in ``hist`` of the nodes that belong to the model.
    hist : list
        History of all the individuals created by the evolution.
    nvar : int, optional
        Number of inputs of the training data.
    classifier : bool, optional
        Classification (default) or regression.
    labels : numpy.ndarray, optional
        The class labels, used to translate the predictions.
    probability_calibration : object, optional
        Fitted object with ``predict_proba`` used by ``predict_proba``.
    nclasses : int, optional
        Number of classes.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import RGP
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = RGP.init(popsize=50, early_stopping_rounds=50, seed=0).fit(X, y).model()
    >>> model.nvar, model.nclasses, model.classifier
    (3, 2, True)
    >>> model.predict(X).shape
    (100,)
    """
    def __init__(self, trace, hist, nvar=None, classifier=True, labels=None,
                 probability_calibration=None, nclasses=None):
        self._classifier = classifier
        self._trace = trace
        self._hist = hist
        self._map = {}
        for k, v in enumerate(self._trace):
            self._map[v] = k
        self._hy_test = self._hist[self._trace[-1]].hy_test
        self._hist = [self.transform(self._hist[x].tostore()) for x in
                      self._trace]
        self._labels = labels
        self._nvar = nvar
        self._probability_calibration = probability_calibration
        self._nclasses = nclasses
        
    @property
    def nclasses(self):
        "Number of classes (``None`` in regression)"
        return self._nclasses

    @property
    def nvar(self):
        "Number of inputs of the training data"
        return self._nvar

    @property
    def multiple_outputs(self):
        "Whether the nodes have several outputs (one per class in classification)"
        return self._hist[0]._multiple_outputs

    @property
    def classifier(self):
        "whether this is classification or regression task"
        return self._classifier

    @property
    def fitness_vs(self):
        "Fitness in the validation set; it is minus the error, so zero is perfect"
        return self._hist[-1].fitness_vs

    @property
    def size(self):
        "Number of nodes of the model"
        return len(self._hist)

    @property
    def height(self):
        "Height of the last node, the longest path down to an input"
        return self._hist[-1].height

    def inputs(self, counter=None):
        """Count how many times each input variable is used by the model.

        Parameters
        ----------
        counter : collections.Counter, optional
            Counter to update; a new one is created by default.

        Returns
        -------
        collections.Counter
            Maps the index of an input (a column of ``X``) to the number of
            nodes that read it.
        """
        from collections import Counter
        if counter is None:
            counter = Counter()
        for node in self._hist:
            if node.height == 0:
                if isinstance(node._variable, list):
                    for _ in node._variable:
                        counter[_] += 1
                else:
                    counter[node._variable] += 1
        return counter

    def transform(self, v):
        """Replace, in node ``v``, positions in the history by positions in the model."""
        if v.height == 0:
            return v
        if v.nargs == 1:
            v.variable = self._map[v.variable]
        else:
            v.variable = [self._map[x] for x in v.variable]
        return v

    def predict_proba(self, X, **kwargs):
        """Probabilities of each class, calibrated on the validation split.

        It needs the ``probability_calibration`` parameter at training time;
        without it this raises ``AttributeError``. Use ``EvoDAGE`` for the
        fraction of votes of an ensemble.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.
        """
        X = self.decision_function(X, **kwargs)
        return self._probability_calibration.predict_proba(X)

    def decision_function(self, X, **kwargs):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            return self._hy_test
        X = self.convert_features(X)
        if len(X) < self.nvar:
            _ = 'Number of variables differ, trained with %s given %s' % (self.nvar, len(X))
            raise RuntimeError(_)
        hist = self._hist
        for node in hist:
            if node.height:
                node.eval(hist)
            else:
                node.eval(X)
        node.normalize()
        r = node.hy
        # dropping the references frees the arrays; there are no cycles to
        # collect (a gc.collect() here took a quarter of a small predict)
        for i in hist[:-1]:
            i.hy = None
            i.hy_test = None
        return r

    def predict(self, X, **kwargs):
        """Predict ``X``: the class with the largest output, or the value in regression.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.

        Returns
        -------
        numpy.ndarray
            One element per example, in the labels of the training data.

        Examples
        --------
        >>> import numpy as np
        >>> from EvoDAG import RGP
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> model = RGP.init(popsize=50, early_stopping_rounds=50, seed=0).fit(X, y).model()
        >>> hy = model.predict(X)
        >>> hy.shape, bool((hy == y).mean() > 0.8)
        ((100,), True)
        """
        hy = self.decision_function(X, **kwargs)
        if self._classifier:
            [au.finite(x) for x in hy]
            hy = np.array(au.argmax(hy), dtype=int)
            if self._labels is not None:
                hy = self._labels[hy]
        else:
            hy = tonparray(hy)
        return hy

    def graphviz(self, fpt, terminals=True):
        """Write the model in the dot language of Graphviz.

        Parameters
        ----------
        fpt : str or file-like
            File name, or an object with a ``write`` method.
        terminals : bool, optional
            Include the input variables in the graph (default true).

        Examples
        --------
        >>> import io
        >>> import numpy as np
        >>> from EvoDAG import RGP
        >>> rng = np.random.RandomState(0)
        >>> X = rng.randn(100, 3)
        >>> y = (X[:, 0] > 0).astype(int)
        >>> model = RGP.init(popsize=50, early_stopping_rounds=50, seed=0).fit(X, y).model()
        >>> out = io.StringIO()
        >>> model.graphviz(out)
        >>> out.getvalue().startswith("digraph EvoDAG {")
        True
        """
        flag = False
        if isinstance(fpt, str):
            flag = True
            fpt = open(fpt, 'w')
        fpt.write("digraph EvoDAG {\n")
        last = len(self._hist) - 1
        height = self._hist[-1].height
        try:
            b, m = np.linalg.solve([[0, height-1], [1, 1]], [9, 1])
        except np.linalg.LinAlgError:  # review fix-1: numpy 2 has no np.linalg.linalg
            b, m = 0, 1
        done = {}
        for k, n in enumerate(self._hist):
            if isinstance(n, Function):
                done[k] = 1
                name = n.__class__.__name__
                if n.height == 0:
                    cdn = "n{0} [label=\"{1}\" fillcolor=red style=filled];\n"
                    fpt.write(cdn.format(k, name))
                    continue
                color = int(np.round(n.height * m + b))
                extra = "colorscheme=blues9 style=filled color={0}".format(color)
                if k == last:
                    extra = "fillcolor=green style=filled"
                fpt.write("n{0} [label=\"{1}\" {2}];\n".format(k,
                                                               name,
                                                               extra))
                vars = n._variable
                if not isinstance(vars, list):
                    vars = [vars]
                for j in vars:
                    if j in done:
                        fpt.write("n{0} -> n{1};\n".format(k, j))
            elif terminals:
                cdn = "n{0} [label=\"X{1}\" fillcolor=red style=filled];\n"
                fpt.write(cdn.format(k, n._variable))
                done[k] = 1
        fpt.write("}\n")
        if flag:
            fpt.close()

    @staticmethod
    def convert_features(v):
        """Convert inputs (rows are examples) into the list of ``Variable`` nodes used to evaluate a model."""
        if v is None:
            return None
        if isinstance(v[0], Variable):
            return v
        if isinstance(v, np.ndarray):
            X = v.T
        elif isinstance(v[0], np.ndarray) and v[0].ndim == 1:
            X = v
        else:
            X = np.array(v).T
        lst = []
        for var, d in enumerate(X):
            v = Variable(var, 1)
            v._eval_tr = np.array(d, dtype=float)
            lst.append(v)
        return lst

    @staticmethod
    def convert_features_test_set(vars, v):
        """Store the test inputs ``v`` as the test set of the ``Variable`` nodes ``vars``."""
        if isinstance(v, np.ndarray):
            X = v.T
        else:
            X = v
        for var, d in zip(vars, X):
            var._eval_ts = np.array(d, dtype=float)


class Ensemble(object):
    """Fitted ensemble: the models obtained with different seeds and how they are combined.

    Classification predicts the class that most models choose and the
    decision function is the median of the models' outputs; in regression
    the prediction is that median. It is what ``EvoDAGE.fit`` returns and
    is not normally created directly: ``Ensemble.init`` creates an unfitted
    one that ``fit`` trains.

    Parameters
    ----------
    models : list of Model or None
        Fitted models; ``None`` for an ensemble to be trained with ``fit``.
    n_jobs : int, optional
        Processes used to train and to evaluate the models (default 1).
    evodags : list of dict, optional
        Parameters of each member, used by ``fit``.
    tmpdir : str, optional
        Directory where finished members are stored; ``fit`` removes it at
        the end.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EvoDAGE
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> ensemble = EvoDAGE(n_estimators=3, popsize=50, early_stopping_rounds=50,
    ...                    seed=0).fit(X, y)
    >>> type(ensemble).__name__, len(ensemble.models)
    ('Ensemble', 3)
    """
    def __init__(self, models, n_jobs=1, evodags=None, tmpdir=None):
        self._models = models
        self._n_jobs = n_jobs
        self._evodags = evodags
        self._tmpdir = tmpdir
        if models is not None:
            self._init()

    def training_size(self, args):
        """Subsample the training set of each member when ``max_training_size`` is given.

        ``args`` is the list of tuples that ``fit`` maps; the same list is
        returned when the parameter is not set.
        """
        X, y, test_set, evodag, self._tmpdir, init_time = args[0]
        max_training_size = evodag.get("max_training_size", None)
        if max_training_size is None:
            return args
        R = []
        for arg in args:
            X, y, test_set, evodag, self._tmpdir, init_time = arg
            X = np.atleast_2d(X)
            y = np.atleast_1d(y)
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X = X[index[:max_training_size]]
            y = y[index[:max_training_size]]
            R.append((X, y, test_set, evodag, self._tmpdir, init_time))
        return R

    def fit(self, X, y, test_set=None):
        """Train the members, in parallel when ``n_jobs`` is greater than one.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.
        y : array_like
            Classes (numbers) or real targets.
        test_set : array_like, optional
            Inputs evaluated along the evolution of each member.

        Returns
        -------
        Ensemble
            The ensemble itself; members that could not be trained in the
            time limit are left out.
        """
        evodags = self._evodags
        init_time = time()
        args = [(X, y, test_set, evodag, self._tmpdir, init_time) for evodag in evodags]
        try:
            time_limit = evodags[0]['time_limit']
        except KeyError:
            time_limit = None
        if time_limit is not None:
            LOGGER.info('time_limit in Ensemble: %0.2f' % time_limit)
        if self._n_jobs == 1:
            _ = [fit(x) for x in tqdm(args)]
            self._models = [x for x in _ if x is not None]
        else:
            p = Pool(self._n_jobs, maxtasksperchild=1)
            self._models = []
            args = self.training_size(args)
            for x in tqdm(p.imap_unordered(fit, args),
                          total=len(args)):
                if x is not None:
                    self._models.append(x)
                if time_limit is not None and time() - init_time > time_limit:
                    p.terminate()
                    break
            p.close()
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir)
        self._init()
        if time_limit is not None:
            LOGGER.info('Used time in Ensemble: %0.2f' % (time() - init_time))
        return self

    def _init(self):
        self._labels = self._models[0]._labels
        self._classifier = False
        flag = False
        if self._models[0]._classifier:
            flag = True
        self._classifier = flag

    @property
    def nclasses(self):
        "Number of classes"
        return self.models[0].nclasses

    @property
    def probability_calibration(self):
        "Whether the models have a probability calibration"
        return self.models[0]._probability_calibration is not None

    @property
    def models(self):
        "List containing the models that compose the ensemble"
        return self._models

    @property
    def multiple_outputs(self):
        "Whether the models have several outputs"
        return self.models[0].multiple_outputs

    @property
    def classifier(self):
        "Whether this is classification or regression"
        return self._classifier

    @property
    def fitness_vs(self):
        "Median Fitness in the validation set"
        l = [x.fitness_vs for x in self.models]
        return np.median(l)

    @property
    def size(self):
        "Median number of nodes of the models"
        l = [x.size for x in self.models]
        return np.median(l)

    @property
    def height(self):
        "Median height of the models"
        l = [x.height for x in self.models]
        return np.median(l)

    def inputs(self, counter=None):
        """Count how many times each input variable is used by all the models.

        See ``Model.inputs``; the counts of the models are added.
        """
        from collections import Counter
        if counter is None:
            counter = Counter()
        for m in self.models:
            m.inputs(counter=counter)
        return counter

    def _decision_function_raw(self, X, cpu_cores=1):
        if cpu_cores == 1:
            r = [m.decision_function(X) for m in self._models]
        else:
            p = Pool(cpu_cores, maxtasksperchild=1)
            args = [(k, m, X) for k, m in enumerate(self._models)]
            r = [x for x in tqdm(p.imap_unordered(decision_function,
                                                  args),
                                 total=len(args))]
            r.sort(key=lambda x: x[0])
            r = [x[1] for x in r]
            p.close()
        return r

    def raw_decision_function(self, X):
        """Outputs of every model side by side.

        Returns
        -------
        numpy.ndarray
            One row per example and, for each model in turn, one column per
            class (a classifier) or per output.
        """
        hy = self._decision_function_raw(X, cpu_cores=self._n_jobs)
        if isinstance(hy[0], list):
            _ = []
            [[_.append(y) for y in x] for x in hy]
            hy = _
        if self.classifier:
            [au.finite(x) for x in hy]
        return np.array([tonparray(x) for x in hy]).T

    def _predict_proba_raw(self, X, cpu_cores=1):
        if cpu_cores == 1:
            r = [m.predict_proba(X) for m in self._models]
        else:
            p = Pool(cpu_cores, maxtasksperchild=1)
            args = [(k, m, X) for k, m in enumerate(self._models)]
            r = [x for x in tqdm(p.imap_unordered(predict_proba,
                                                  args),
                                 total=len(args))]
            r.sort(key=lambda x: x[0])
            r = [x[1] for x in r]
            p.close()
        return r

    def predict_proba(self, X):
        """Probability of each class.

        Without calibration it is the fraction of the models that predict
        the class; with ``probability_calibration`` it is the average of
        the calibrated probabilities.

        Returns
        -------
        numpy.ndarray
            One row per example, one column per class, in the order of the
            sorted labels.
        """
        if self.probability_calibration:
            proba = np.array(self._predict_proba_raw(X, cpu_cores=self._n_jobs))
            proba = np.mean(proba, axis=0)
            proba /= np.sum(proba, axis=1)[:, np.newaxis]
            proba[np.isnan(proba)] = 1. / self.nclasses
            return proba
        hy = self._decision_function_raw(X, cpu_cores=self._n_jobs)
        minlength = len(hy[0])
        hy = [au.argmax(x) for x in hy]
        hy = np.array(hy, dtype=int).T
        hy = [np.bincount(x, minlength=minlength) for x in hy]
        return np.array([x / float(x.sum()) for x in hy])

    def decision_function(self, X, cpu_cores=1):
        """Median, over the models, of their outputs.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.
        cpu_cores : int, optional
            Processes used to evaluate the models; the larger of this and
            ``n_jobs`` is used.

        Returns
        -------
        numpy.ndarray
            One row per example and one column per class (classification).
        """
        cpu_cores = max(cpu_cores, self._n_jobs)
        r = self._decision_function_raw(X, cpu_cores=cpu_cores)
        if isinstance(r[0], np.ndarray):
            r = np.array([tonparray(x) for x in r if au.isfinite(x)])
            r = np.median(r, axis=0)
        else:
            [[au.finite(x) for x in o] for o in r]
            r = np.array([[tonparray(y) for y in x] for x in r])
            r = np.median(r, axis=0)
        return r.T

    def predict(self, X, cpu_cores=1):
        """Predict ``X``: the class chosen by most models, or the median in regression.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.
        cpu_cores : int, optional
            Processes used to evaluate the models.

        Returns
        -------
        numpy.ndarray
            One element per example, in the labels of the training data.
        """
        cpu_cores = max(cpu_cores, self._n_jobs)
        if self.classifier:
            return self.predict_cl(X, cpu_cores=cpu_cores)
        return self.decision_function(X, cpu_cores=cpu_cores)

    def predict_cl(self, X, cpu_cores=1):
        """Classes of ``X`` by majority vote of the models; ``predict`` uses it in classification."""
        cpu_cores = max(cpu_cores, self._n_jobs)
        hy = [au.argmax(x) for x in
              self._decision_function_raw(X, cpu_cores=cpu_cores)]
        hy = np.array(hy, dtype=int).T
        hy = [np.bincount(x).argmax() for x in hy]
        if self._labels is not None:
            hy = self._labels[hy]
        return hy

    def graphviz(self, directory, **kwargs):
        "Directory to store the graphviz models"
        import os
        if not os.path.isdir(directory):
            os.mkdir(directory)
        output = os.path.join(directory, 'evodag-%s')
        for k, m in enumerate(self.models):
            m.graphviz(output % k, **kwargs)

    @classmethod
    def init(cls, n_estimators=30, n_jobs=1, tmpdir=None, **kwargs):
        """Create an ensemble that is ready to be trained with ``fit``.

        Parameters
        ----------
        n_estimators : int, optional
            Number of members (default 30). Member ``i`` uses seed
            ``seed + i``.
        n_jobs : int, optional
            Processes used to train and to evaluate the members.
        tmpdir : str, optional
            Directory where finished members are stored.
        **kwargs
            Parameters of every member (see ``EvoDAG.EvoDAG``).

        Returns
        -------
        Ensemble
            An ensemble without models.
        """
        try:
            init_seed = kwargs['seed']
            del kwargs['seed']
        except KeyError:
            init_seed = 0
        lst = []
        for x in range(init_seed, init_seed + n_estimators):
            kwargs['seed'] = x
            lst.append(kwargs.copy())
        if tmpdir is not None and not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
        return cls(None, evodags=lst, n_jobs=n_jobs, tmpdir=tmpdir)


class EvoDAGE(object):
    """Ensemble of EvoDAG models with a scikit-learn-like interface.

    It is exported by the top-level package: ``from EvoDAG import EvoDAGE``.
    ``n_estimators`` runs are trained with consecutive seeds; each one
    learns from its own random split of the training data. The default
    parameters are those of ``EvoDAG/conf/default_parameters.json`` (or
    ``default_parameters_r.json`` when ``classifier=False``); in classification
    they include the agreement selection of the parents and the random choice
    of the individual that leaves (``orthogonal_selection=True``,
    ``negative_selection=False``). Note that
    ``fit`` returns an ``Ensemble``, not the ``EvoDAGE`` instance.

    Parameters
    ----------
    time_limit : float, optional
        Seconds for the whole ensemble (default ``None``, no limit). It
        replaces the ``time_limit`` of the parameters file.
    n_estimators : int, optional
        Number of members (default 30).
    n_jobs : int, optional
        Processes used to train and to evaluate the members (default 1).
    tmpdir : str, optional
        Directory where finished members are stored, so that an interrupted
        fit can be resumed.
    seed : int, optional
        Seed of the first member (default 0).
    params_fname : str, optional
        JSON file with parameters, see ``EvoDAG.EvoDAG.init``.
    **kwargs
        Any parameter of the engine, see ``EvoDAG.EvoDAG``: ``popsize``,
        ``early_stopping_rounds``, ``classifier``, the function names...

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EvoDAGE
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> ensemble = EvoDAGE(n_estimators=3, popsize=50, early_stopping_rounds=50,
    ...                    seed=0).fit(X, y)
    >>> hy = ensemble.predict(X)
    >>> hy.shape, bool((hy == y).mean() > 0.8)
    ((100,), True)
    >>> ensemble.predict_proba(X).shape
    (100, 2)
    >>> ensemble.decision_function(X).shape
    (100, 2)
    >>> ensemble.raw_decision_function(X).shape
    (100, 6)
    """

    def __init__(self, time_limit=None, **kwargs):
        self._m = Ensemble.init(time_limit=time_limit, **kwargs)
        self._time_limit = time_limit

    @property
    def time_limit(self):
        "Seconds for the whole ensemble (``None``: no limit); it can be set"
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit):
        self._time_limit = time_limit
        for x in self._m._evodags:
            x['time_limit'] = self._time_limit

    def fit(self, *args, **kwargs):
        """Train the members; see ``Ensemble.fit`` (arguments ``X``, ``y``, ``test_set``).

        Returns
        -------
        Ensemble
            The fitted ensemble (not this ``EvoDAGE`` instance), which has
            the ``predict`` family of methods and ``models``.
        """
        return self._m.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        "Predict the inputs; see ``Ensemble.predict``"
        return self._m.predict(*args, **kwargs)

    def decision_function(self, *args, **kwargs):
        "Median of the outputs of the models; see ``Ensemble.decision_function``"
        return self._m.decision_function(*args, **kwargs)

    def raw_decision_function(self, *args, **kwargs):
        "Outputs of every model side by side; see ``Ensemble.raw_decision_function``"
        return self._m.raw_decision_function(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        "Probability of each class; see ``Ensemble.predict_proba``"
        return self._m.predict_proba(*args, **kwargs)


class EvoDAG(EvoDAGE):
    """A single model with the scikit-learn-like ``fit``/``predict`` interface.

    It is **not** the class exported as ``EvoDAG.EvoDAG`` (that one,
    ``EvoDAG.base.EvoDAG``, is the search engine, also available as ``RGP``
    and ``RootGP``); this one is reached as ``EvoDAG.model.EvoDAG``. It
    inherits ``predict``, ``decision_function`` and the other methods of
    ``EvoDAGE``, which after ``fit`` work on the fitted ``Model``; in
    classification ``decision_function`` returns a list with one array per
    class.

    Parameters
    ----------
    **kwargs
        Parameters of the engine (see ``EvoDAG.EvoDAG`` and
        ``EvoDAG.EvoDAG.init``), including ``params_fname``, ``seed`` and
        ``classifier``.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG.model import EvoDAG as SingleEvoDAG
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> single = SingleEvoDAG(popsize=50, early_stopping_rounds=50, seed=0).fit(X, y)
    >>> single.predict(X).shape
    (100,)
    >>> type(single.model).__name__
    'Model'
    """

    def __init__(self, **kwargs):
        from .base import EvoDAG as evodag
        self._m = evodag.init(**kwargs)

    def fit(self, *args, **kwargs):
        """Run the evolution and keep the model of the early stopping individual.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example.
        y : array_like
            Classes (numbers) or real targets.
        test_set : array_like, optional
            Inputs evaluated along the evolution.

        Returns
        -------
        EvoDAG
            This instance, whose ``model`` is now the fitted ``Model``.
        """
        self._m.fit(*args, **kwargs)
        self._m = self._m.model()
        return self

    @property
    def model(self):
        "The engine before ``fit`` and the fitted ``Model`` after it"
        return self._m
