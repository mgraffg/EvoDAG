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
"""scikit-learn estimators that fit **one** EvoDAG model.

``EvoDAGClassifier`` and ``EvoDAGRegressor`` follow the conventions of
scikit-learn (``get_params``/``set_params``, ``clone``, pipelines, grid
search, ``score``) and pass its estimator checks. Each instance runs one
evolution and keeps one :py:class:`~EvoDAG.model.Model`; they do not build
ensembles. This module needs scikit-learn, which is an optional dependency::

    pip install 'EvoDAG[sklearn]'

The two classes are reachable from the top-level package without importing
this module until they are used: ``from EvoDAG import EvoDAGClassifier``.
Without scikit-learn that statement raises an ``ImportError`` that says how
to install it, and the rest of the package keeps working.
"""
import copy
import numbers
import numpy as np
try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.validation import check_is_fitted, validate_data
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError("EvoDAG.estimators needs scikit-learn: "
                      "pip install 'EvoDAG[sklearn]'") from error
from . import array_utils as au
from .base import EvoDAG as _Engine
from .utils import tonparray


class _EvoDAGBase(BaseEstimator):
    """Parameters, ``fit`` and the evaluation of the model, shared by both estimators."""

    # classification (True) or regression (False); set by the subclasses
    _classifier = True

    def __init__(self, popsize=None, early_stopping_rounds=None,
                 time_limit=None, random_state=0, params_fname=None,
                 params=None):
        self.popsize = popsize
        self.early_stopping_rounds = early_stopping_rounds
        self.time_limit = time_limit
        self.random_state = random_state
        self.params_fname = params_fname
        self.params = params

    def _engine(self):
        """The (unfitted) engine for the parameters of this instance."""
        random_state = self.random_state
        if random_state is None:
            random_state = 0
        if (isinstance(random_state, bool)
                or not isinstance(random_state, numbers.Integral)):
            raise ValueError("random_state must be an int or None; got %r"
                             % (random_state,))
        params = self.params
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be a dict or None; got %r"
                             % type(params).__name__)
        reserved = {"seed", "classifier"} & set(params)
        if reserved:
            raise ValueError("params cannot contain %s; use random_state and "
                             "the class of the estimator" % sorted(reserved))
        kwargs = dict(params)
        for name in ("popsize", "early_stopping_rounds", "time_limit"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        params_fname = self.params_fname
        if isinstance(params_fname, dict):
            # the engine updates the dictionary it is given
            params_fname = dict(params_fname)
        return _Engine.init(params_fname=params_fname,
                            seed=int(random_state),
                            classifier=self._classifier, **kwargs)

    def _fit(self, X, y):
        "Run one evolution on the validated ``X`` and the encoded ``y``."
        self.model_ = self._engine().fit(X, y).model()
        return self

    def _outputs(self, X):
        """Outputs of the model on ``X``, validated: a list with one array per class."""
        check_is_fitted(self, "model_")
        X = validate_data(self, X, reset=False)
        return self.model_.decision_function(X)

    def __setstate__(self, state):
        super().__setstate__(state)
        model = self.__dict__.get("model_")
        if model is not None:
            # An estimator loaded with joblib's mmap_mode="r" (also what
            # scikit-learn's check_estimators_pickle does) has read-only
            # arrays, which the Cython kernels of the nodes reject; a copy
            # of the (small) model has ordinary writable arrays.
            self.model_ = copy.deepcopy(model)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags


class EvoDAGClassifier(ClassifierMixin, _EvoDAGBase):
    """Classifier that evolves one EvoDAG model (scikit-learn interface).

    One instance is one evolution and one :py:class:`~EvoDAG.model.Model`; to
    combine several of them see the ensembles of scikit-learn or
    :py:class:`~EvoDAG.model.EvoDAGE`. With integer labels ``0..k-1`` and the
    same parameters and seed, ``predict`` equals that of
    ``EvoDAG.model.EvoDAG(seed=random_state, ...)`` (the single model of
    :py:mod:`EvoDAG.model`, not the search engine ``EvoDAG.EvoDAG``).
    Labels of any type are accepted; they are encoded with ``LabelEncoder``
    and the engine receives ``0..k-1``.

    Parameters
    ----------
    popsize : int, optional
        Population size. ``None`` (default) uses the value of the file of
        default parameters ``EvoDAG/conf/default_parameters.json``.
    early_stopping_rounds : int, optional
        Individuals created without improvement before the evolution stops.
        ``None`` uses the default parameters.
    time_limit : float, optional
        Seconds allowed for the evolution. ``None`` uses the default
        parameters.
    random_state : int, optional
        Seed of the evolution (the engine's ``seed``; default 0). Only an
        integer is accepted and ``None`` is the same as 0, so a fit is always
        deterministic.
    params_fname : str or dict, optional
        JSON file (or dictionary) used instead of the default parameters, see
        ``EvoDAG.EvoDAG.init``.
    params : dict, optional
        Any other parameter of the engine, for example
        ``{"Add": 30, "tr_fraction": 0.4}``. ``popsize``,
        ``early_stopping_rounds`` and ``time_limit`` given as arguments take
        precedence; the keys ``seed`` and ``classifier`` are not allowed. A
        ``probability_calibration`` class given here makes ``predict_proba``
        return calibrated probabilities.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The class labels, sorted.
    n_features_in_ : int
        Number of columns of ``X`` seen in ``fit``.
    model_ : EvoDAG.model.Model
        The fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EvoDAGClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = np.where(X[:, 0] > 0, "up", "down")
    >>> clf = EvoDAGClassifier(popsize=50, early_stopping_rounds=50,
    ...                        random_state=0).fit(X, y)
    >>> clf.classes_.tolist()
    ['down', 'up']
    >>> clf.predict(X).shape
    (100,)
    >>> bool(clf.score(X, y) > 0.8)
    True
    >>> clf.decision_function(X).shape
    (100,)
    >>> clf.predict_proba(X).shape
    (100, 2)
    """

    _classifier = True

    def fit(self, X, y):
        """Run one evolution.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``); NaN,
            infinite values and sparse matrices are rejected.
        y : array_like
            Class labels, numbers or strings.

        Returns
        -------
        EvoDAGClassifier
            This instance, with ``model_``, ``classes_`` and
            ``n_features_in_`` set.

        Raises
        ------
        ValueError
            If there are fewer than two examples, ``y`` has a single class
            or ``y`` is not a classification target.
        """
        X, y = validate_data(self, X, y, ensure_min_samples=2)
        check_classification_targets(y)
        encoder = LabelEncoder().fit(y)
        if encoder.classes_.shape[0] < 2:
            raise ValueError("This solver needs samples of at least 2 "
                             "classes in the data, but the data contains "
                             "only one class: %r" % encoder.classes_[0])
        self.classes_ = encoder.classes_
        return self._fit(X, encoder.transform(y))

    def _scores(self, X):
        "Outputs as a matrix with one column per class, without non-finite values"
        outputs = self._outputs(X)
        return np.array([au.finite(np.array(tonparray(x), dtype=float))
                         for x in outputs]).T

    def predict(self, X):
        """Predict the class of each example: the one with the largest output.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One element of ``classes_`` per example.
        """
        scores = self._scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def decision_function(self, X):
        """Outputs of the model.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)`` with two classes, the output of the second
            class minus that of the first (positive means ``classes_[1]``);
            ``(n_samples, n_classes)`` otherwise. Non-finite outputs count as
            zero, as in ``predict``.
        """
        scores = self._scores(X)
        if scores.shape[1] == 2:
            return scores[:, 1] - scores[:, 0]
        return scores

    def predict_proba(self, X):
        """Probability of each class.

        A single model has no graded probability. If the engine parameter
        ``probability_calibration`` was given, the calibrated probabilities
        are returned (they are not guaranteed to have the same argmax as
        ``predict``); otherwise the result is the one-hot encoding of
        ``predict``. The rows add up to 1.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One row per example and one column per class, in the order of
            ``classes_``.
        """
        scores = self._scores(X)
        calibration = self.model_._probability_calibration
        if calibration is not None:
            proba = np.asarray(calibration.predict_proba(scores), dtype=float)
            total = proba.sum(axis=1)[:, np.newaxis]
            with np.errstate(divide="ignore", invalid="ignore"):
                proba = proba / total
            proba[~np.isfinite(proba)] = 1. / proba.shape[1]
            return proba
        proba = np.zeros(scores.shape)
        proba[np.arange(scores.shape[0]), np.argmax(scores, axis=1)] = 1.
        return proba


class EvoDAGRegressor(RegressorMixin, _EvoDAGBase):
    """Regressor that evolves one EvoDAG model (scikit-learn interface).

    One instance is one evolution and one :py:class:`~EvoDAG.model.Model`.
    With the same parameters and seed, ``predict`` equals that of
    ``EvoDAG.model.EvoDAG(seed=random_state, classifier=False, ...)``.
    The defaults come from ``EvoDAG/conf/default_parameters_r.json``.

    Parameters
    ----------
    popsize : int, optional
        Population size. ``None`` (default) uses the default parameters.
    early_stopping_rounds : int, optional
        Individuals created without improvement before the evolution stops.
        ``None`` uses the default parameters.
    time_limit : float, optional
        Seconds allowed for the evolution. ``None`` uses the default
        parameters.
    random_state : int, optional
        Seed of the evolution (the engine's ``seed``; default 0). Only an
        integer is accepted and ``None`` is the same as 0.
    params_fname : str or dict, optional
        JSON file (or dictionary) used instead of the default parameters, see
        ``EvoDAG.EvoDAG.init``.
    params : dict, optional
        Any other parameter of the engine, for example ``{"Add": 30}``; the
        keys ``seed`` and ``classifier`` are not allowed.

    Attributes
    ----------
    n_features_in_ : int
        Number of columns of ``X`` seen in ``fit``.
    model_ : EvoDAG.model.Model
        The fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EvoDAGRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = 2 * X[:, 0] - X[:, 1]
    >>> reg = EvoDAGRegressor(popsize=50, early_stopping_rounds=50,
    ...                       random_state=0).fit(X, y)
    >>> reg.predict(X).shape
    (100,)
    >>> bool(reg.score(X, y) > 0.9)
    True
    """

    _classifier = False

    def fit(self, X, y):
        """Run one evolution.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``); NaN,
            infinite values and sparse matrices are rejected.
        y : array_like
            Real targets.

        Returns
        -------
        EvoDAGRegressor
            This instance, with ``model_`` and ``n_features_in_`` set.
        """
        X, y = validate_data(self, X, y, y_numeric=True, ensure_min_samples=2)
        return self._fit(X, y)

    def predict(self, X):
        """Predict the target of each example.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One predicted value per example.
        """
        return np.asarray(tonparray(self._outputs(X)), dtype=float)
