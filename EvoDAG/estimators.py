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
"""scikit-learn estimators: one EvoDAG model, and ensembles of any estimator.

``EvoDAGClassifier`` and ``EvoDAGRegressor`` follow the conventions of
scikit-learn (``get_params``/``set_params``, ``clone``, pipelines, grid
search, ``score``) and pass its estimator checks. Each instance runs one
evolution and keeps one :py:class:`~EvoDAG.model.Model`. ``EnsembleClassifier``
and ``EnsembleRegressor`` clone any estimator several times and combine the
members as ``EvoDAGE`` does, so that
``EnsembleClassifier(EvoDAGClassifier())`` is its scikit-learn form. This
module needs scikit-learn, which is an optional dependency::

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
    from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                              clone)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.metaestimators import available_if
    from sklearn.utils.multiclass import check_classification_targets
    from sklearn.utils.parallel import Parallel, delayed
    from sklearn.utils.validation import check_is_fitted, validate_data
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError("EvoDAG.estimators needs scikit-learn: "
                      "pip install 'EvoDAG[sklearn]'") from error
from . import array_utils as au
from .base import EvoDAG as _Engine
from .utils import tonparray


def _int_seed(random_state):
    "``random_state`` as an ``int``: only integers are accepted and ``None`` is 0."
    if random_state is None:
        return 0
    if (isinstance(random_state, bool)
            or not isinstance(random_state, numbers.Integral)):
        raise ValueError("random_state must be an int or None; got %r"
                         % (random_state,))
    return int(random_state)


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
        random_state = _int_seed(self.random_state)
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
                            seed=random_state,
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


def _fit_member(member, X, y):
    "Fit ``member`` and return it (the function that ``joblib`` runs for each member)."
    return member.fit(X, y)


def _predict_member(member, method, X):
    "Call ``method`` of a fitted ``member`` (run by ``joblib``)."
    return getattr(member, method)(X)


def _member_has(method):
    "``available_if`` check: the members of the ensemble have ``method``."
    def check(self):
        return hasattr(self._base_estimator(), method)
    return check


class _EnsembleBase(BaseEstimator):
    """Members, ``fit`` and evaluation of the members, shared by both ensembles."""

    def __init__(self, estimator=None, n_estimators=30, random_state=0,
                 n_jobs=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _default_estimator(self):
        raise NotImplementedError  # pragma: no cover

    def _base_estimator(self):
        "The estimator that is cloned: the one given or the default of the class."
        if self.estimator is None:
            return self._default_estimator()
        return self.estimator

    def _check_parameters(self):
        n_estimators = self.n_estimators
        if (isinstance(n_estimators, bool)
                or not isinstance(n_estimators, numbers.Integral)
                or n_estimators < 1):
            raise ValueError("n_estimators must be an integer >= 1; got %r"
                             % (n_estimators,))
        return _int_seed(self.random_state)

    def _fit_members(self, X, y):
        "Fit ``n_estimators`` clones, member ``i`` with ``random_state + i``."
        seed = self._check_parameters()
        members = []
        for i in range(self.n_estimators):
            member = clone(self._base_estimator())
            if "random_state" in member.get_params(deep=False):
                member.set_params(random_state=seed + i)
            members.append(member)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_member)(member, X, y) for member in members)
        return self

    def _members_output(self, method, X):
        """``method`` of every member on ``X`` (validated), in the order of the members."""
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False)
        return Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_member)(member, method, X)
            for member in self.estimators_)


class EnsembleClassifier(ClassifierMixin, _EnsembleBase):
    """Ensemble of clones of a classifier, combined as ``EvoDAGE`` does.

    ``EnsembleClassifier(EvoDAGClassifier(...), n_estimators=n, random_state=s)``
    is the scikit-learn form of ``EvoDAG.EvoDAGE(n_estimators=n, seed=s, ...)``
    and gives the same predictions. It also works with any other classifier.
    Member ``i`` is a clone of ``estimator`` fitted on the whole training set
    with ``random_state + i`` (the diversity comes from the seed); ``predict``
    is the majority vote of the members, a tie going to the class with the
    lowest index, and ``decision_function`` is the median over the members.
    scikit-learn's bagging and voting ensembles cannot be used instead: they
    average the outputs and draw random seeds.

    Parameters
    ----------
    estimator : object, optional
        Classifier that is cloned; ``None`` (default) is
        :py:class:`~EvoDAG.estimators.EvoDAGClassifier` with its default parameters. If it has
        no ``random_state`` parameter the members are identical clones. Only
        its own ``random_state`` is set (not that of nested estimators).
    n_estimators : int, optional
        Number of members (default 30).
    random_state : int, optional
        Seed of the first member (default 0); only an integer is accepted
        and ``None`` is the same as 0.
    n_jobs : int, optional
        Processes used by ``joblib`` to fit and evaluate the members
        (default 1); the results do not depend on it.
    voting : str, optional
        Either ``"hard"`` or ``"soft"``; how ``predict_proba`` combines the
        members. ``"hard"`` (default) is the fraction of members that vote for each class. ``"soft"`` is the
        mean of the ``predict_proba`` of the members, renormalised (rows
        that are not finite become ``1 / n_classes``); it is the way
        ``EvoDAGE`` combines calibrated probabilities and needs members with
        ``predict_proba``. ``predict`` is always the majority vote, so with
        ``"soft"`` the argmax of ``predict_proba`` may differ from it.

    Attributes
    ----------
    estimators_ : list
        The fitted members, in order.
    classes_ : numpy.ndarray
        The class labels, sorted.
    n_features_in_ : int
        Number of columns of ``X`` seen in ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EnsembleClassifier, EvoDAGClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = np.where(X[:, 0] > 0, "up", "down")
    >>> tiny = EvoDAGClassifier(popsize=50, early_stopping_rounds=50)
    >>> clf = EnsembleClassifier(tiny, n_estimators=3, random_state=0).fit(X, y)
    >>> [m.random_state for m in clf.estimators_]
    [0, 1, 2]
    >>> clf.predict(X).shape
    (100,)
    >>> bool(clf.score(X, y) > 0.8)
    True
    >>> clf.predict_proba(X).shape
    (100, 2)
    """

    def __init__(self, estimator=None, n_estimators=30, random_state=0,
                 n_jobs=1, voting="hard"):
        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         random_state=random_state, n_jobs=n_jobs)
        self.voting = voting

    def _default_estimator(self):
        return EvoDAGClassifier()

    def fit(self, X, y):
        """Fit the members on the whole training set.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``); NaN,
            infinite values and sparse matrices are rejected.
        y : array_like
            Class labels, numbers or strings. The members receive them
            encoded as ``0..k-1``.

        Returns
        -------
        EnsembleClassifier
            This instance, with ``estimators_``, ``classes_`` and
            ``n_features_in_`` set.

        Raises
        ------
        ValueError
            If ``n_estimators`` is not an integer >= 1, ``voting`` is not
            ``"hard"`` or ``"soft"``, ``random_state`` is not an integer, or
            ``y`` is not a classification target.
        """
        if self.voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'; got %r"
                             % (self.voting,))
        X, y = validate_data(self, X, y, ensure_min_samples=2)
        check_classification_targets(y)
        encoder = LabelEncoder().fit(y)
        if encoder.classes_.shape[0] < 2:
            raise ValueError("This solver needs samples of at least 2 "
                             "classes in the data, but the data contains "
                             "only one class: %r" % encoder.classes_[0])
        self.classes_ = encoder.classes_
        return self._fit_members(X, encoder.transform(y))

    def _votes(self, X):
        "Number of members that vote for each class: ``(n_samples, n_classes)``."
        predictions = self._members_output("predict", X)
        k = self.classes_.shape[0]
        votes = np.zeros((np.shape(predictions[0])[0], k))
        rows = np.arange(votes.shape[0])
        for prediction in predictions:
            votes[rows, np.asarray(prediction).astype(int)] += 1
        return votes

    def predict(self, X):
        """Predict the class chosen by most members.

        A tie goes to the class with the lowest index (the smallest label).

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One element of ``classes_`` per example.
        """
        votes = self._votes(X)
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X):
        """Probability of each class, see ``voting``.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One row per example, one column per class in the order of
            ``classes_``; the rows add up to 1.

        Raises
        ------
        AttributeError
            With ``voting="soft"`` if the members have no ``predict_proba``.
        """
        if self.voting == "hard":
            votes = self._votes(X)
            return votes / votes.sum(axis=1)[:, np.newaxis]
        if not hasattr(self._base_estimator(), "predict_proba"):
            raise AttributeError("voting='soft' needs members with "
                                 "predict_proba")
        probabilities = self._members_output("predict_proba", X)
        k = self.classes_.shape[0]
        total = np.zeros((np.shape(probabilities[0])[0], k))
        for member, proba in zip(self.estimators_, probabilities):
            # a member that did not see every class has fewer columns
            columns = np.asarray(getattr(member, "classes_", np.arange(k)))
            total[:, columns.astype(int)] += proba
        proba = total / len(probabilities)
        with np.errstate(divide="ignore", invalid="ignore"):
            proba = proba / proba.sum(axis=1)[:, np.newaxis]
        proba[~np.isfinite(proba).all(axis=1)] = 1. / k
        return proba

    @available_if(_member_has("decision_function"))
    def decision_function(self, X):
        """Median, over the members, of their ``decision_function``.

        Only available when the estimator has a ``decision_function``.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)`` with two classes and
            ``(n_samples, n_classes)`` otherwise, as the members return it.
        """
        outputs = self._members_output("decision_function", X)
        return np.median(np.array(outputs), axis=0)


class EnsembleRegressor(RegressorMixin, _EnsembleBase):
    """Ensemble of clones of a regressor, combined as ``EvoDAGE`` does.

    ``EnsembleRegressor(EvoDAGRegressor(...), n_estimators=n, random_state=s)``
    is the scikit-learn form of
    ``EvoDAG.EvoDAGE(n_estimators=n, seed=s, classifier=False, ...)`` and gives
    the same predictions. It also works with any other regressor. Member ``i``
    is a clone of ``estimator`` fitted on the whole training set with
    ``random_state + i``, and ``predict`` is the median of the members'
    predictions, which are left out when they are not all finite (if none is
    finite all of them are used).

    Parameters
    ----------
    estimator : object, optional
        Regressor that is cloned; ``None`` (default) is
        :py:class:`~EvoDAG.estimators.EvoDAGRegressor` with its default parameters. If it has no
        ``random_state`` parameter the members are identical clones. Only its
        own ``random_state`` is set (not that of nested estimators).
    n_estimators : int, optional
        Number of members (default 30).
    random_state : int, optional
        Seed of the first member (default 0); only an integer is accepted
        and ``None`` is the same as 0.
    n_jobs : int, optional
        Processes used by ``joblib`` to fit and evaluate the members
        (default 1); the results do not depend on it.

    Attributes
    ----------
    estimators_ : list
        The fitted members, in order.
    n_features_in_ : int
        Number of columns of ``X`` seen in ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG import EnsembleRegressor, EvoDAGRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(100, 3)
    >>> y = 2 * X[:, 0] - X[:, 1]
    >>> tiny = EvoDAGRegressor(popsize=50, early_stopping_rounds=50)
    >>> reg = EnsembleRegressor(tiny, n_estimators=3, random_state=0).fit(X, y)
    >>> reg.predict(X).shape
    (100,)
    >>> bool(reg.score(X, y) > 0.9)
    True
    """

    def _default_estimator(self):
        return EvoDAGRegressor()

    def fit(self, X, y):
        """Fit the members on the whole training set.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``); NaN,
            infinite values and sparse matrices are rejected.
        y : array_like
            Real targets.

        Returns
        -------
        EnsembleRegressor
            This instance, with ``estimators_`` and ``n_features_in_`` set.

        Raises
        ------
        ValueError
            If ``n_estimators`` is not an integer >= 1 or ``random_state``
            is not an integer.
        """
        X, y = validate_data(self, X, y, y_numeric=True, ensure_min_samples=2)
        return self._fit_members(X, y)

    def predict(self, X):
        """Median of the predictions of the members.

        Parameters
        ----------
        X : array_like
            Inputs, one row per example (``n_samples, n_features``).

        Returns
        -------
        numpy.ndarray
            One predicted value per example.
        """
        predictions = np.array(self._members_output("predict", X), dtype=float)
        finite = np.isfinite(predictions).all(axis=1)
        if finite.any():
            predictions = predictions[finite]
        return np.median(predictions, axis=0)
