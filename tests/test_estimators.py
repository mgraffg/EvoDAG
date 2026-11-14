"""Tests of the scikit-learn estimators ``EvoDAGClassifier`` and ``EvoDAGRegressor``."""
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.datasets import load_diabetes, load_iris  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.base import clone, is_classifier, is_regressor  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from EvoDAG import EvoDAGClassifier, EvoDAGRegressor  # noqa: E402
from EvoDAG.model import EvoDAG as SingleEvoDAG  # noqa: E402

TINY = dict(popsize=10, early_stopping_rounds=5)


@parametrize_with_checks([EvoDAGClassifier(**TINY), EvoDAGRegressor(**TINY)])
def test_sklearn_checks(estimator, check):
    check(estimator)


def test_pipeline_cross_val_grid_search():
    X, y = load_iris(return_X_y=True)
    pipe = make_pipeline(StandardScaler(), EvoDAGClassifier(**TINY))
    assert is_classifier(pipe)
    scores = cross_val_score(pipe, X, y, cv=2)
    assert scores.shape == (2,) and (scores > 0.3).all()
    grid = GridSearchCV(EvoDAGClassifier(early_stopping_rounds=5),
                        {"popsize": [10, 20]}, cv=2).fit(X, y)
    assert grid.best_params_["popsize"] in (10, 20)
    assert grid.predict(X).shape == y.shape

    X, y = load_diabetes(return_X_y=True)
    pipe = make_pipeline(StandardScaler(), EvoDAGRegressor(**TINY))
    assert is_regressor(pipe)
    assert cross_val_score(pipe, X, y, cv=2).shape == (2,)
    grid = GridSearchCV(EvoDAGRegressor(early_stopping_rounds=5),
                        {"popsize": [10, 20]}, cv=2).fit(X, y)
    assert grid.best_params_["popsize"] in (10, 20)


def test_params_and_clone():
    clf = EvoDAGClassifier(popsize=10, params={"Add": 5})
    assert clf.get_params()["params"] == {"Add": 5}
    assert clone(clf).get_params() == clf.get_params()
    assert clf.set_params(popsize=20).popsize == 20
    X, y = load_iris(return_X_y=True)
    for bad in ({"seed": 3}, {"classifier": False}, [1]):
        with pytest.raises(ValueError):
            EvoDAGClassifier(params=bad, **TINY).fit(X, y)
    with pytest.raises(ValueError):
        EvoDAGClassifier(random_state=np.random.RandomState(0),
                         **TINY).fit(X, y)
    # None is the same as 0, and the dictionary of the user is not modified
    params = {"Add": 5}
    a = EvoDAGClassifier(random_state=None, params=params, **TINY).fit(X, y)
    b = EvoDAGClassifier(random_state=0, params=params, **TINY).fit(X, y)
    assert params == {"Add": 5}
    assert (a.predict(X) == b.predict(X)).all()


@pytest.mark.parametrize("labels", [["a", "b", "c"], [5, 7, 9]])
def test_string_and_shifted_labels(labels):
    X, y = load_iris(return_X_y=True)
    labels = np.array(labels)
    clf = EvoDAGClassifier(**TINY)
    assert clf.fit(X, labels[y]) is clf
    assert clf.classes_.tolist() == sorted(labels.tolist())
    assert clf.n_features_in_ == 4
    hy = clf.predict(X)
    assert set(hy.tolist()) <= set(labels.tolist())
    proba = clf.predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert (clf.classes_[proba.argmax(axis=1)] == hy).all()


def test_decision_function_shapes():
    X, y = load_iris(return_X_y=True)
    clf = EvoDAGClassifier(**TINY).fit(X, y)
    assert clf.decision_function(X).shape == (150, 3)
    binary = y != 2
    clf = EvoDAGClassifier(**TINY).fit(X[binary], y[binary])
    d = clf.decision_function(X[binary])
    assert d.shape == (100,)
    hy = clf.predict(X[binary])
    assert ((d > 0) == (hy == clf.classes_[1])).all()


def test_fit_sets_attributes():
    from EvoDAG import Model
    X, y = load_diabetes(return_X_y=True)
    reg = EvoDAGRegressor(**TINY)
    assert reg.fit(X, y) is reg
    assert reg.n_features_in_ == 10
    assert isinstance(reg.model_, Model)
    assert reg.predict(X).shape == y.shape


def test_probability_calibration():
    class Calibration(object):
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.tile([0.25, 0.75], (len(X), 1))

    X, y = load_iris(return_X_y=True)
    binary = y != 2
    clf = EvoDAGClassifier(params={"probability_calibration": Calibration},
                           **TINY).fit(X[binary], y[binary])
    np.testing.assert_allclose(clf.predict_proba(X[binary]),
                               np.tile([0.25, 0.75], (100, 1)))


def test_same_predictions_as_single_evodag():
    X, y = load_iris(return_X_y=True)
    for seed in (0, 3):
        ours = EvoDAGClassifier(random_state=seed, **TINY).fit(X, y)
        single = SingleEvoDAG(seed=seed, **TINY).fit(X, y)
        assert (ours.predict(X) == single.predict(X)).all()
    X, y = load_diabetes(return_X_y=True)
    for seed in (0, 3):
        ours = EvoDAGRegressor(random_state=seed, **TINY).fit(X, y)
        single = SingleEvoDAG(seed=seed, classifier=False, **TINY).fit(X, y)
        np.testing.assert_array_equal(ours.predict(X), single.predict(X))


def test_import_error_without_sklearn():
    code = (
        "import sys\n"
        "sys.modules['sklearn'] = None\n"
        "import EvoDAG\n"
        "EvoDAG.EvoDAGE\n"
        "assert 'EvoDAGClassifier' not in EvoDAG.__all__\n"
        "ns = {}\n"
        "exec('from EvoDAG import *', ns)\n"
        "try:\n"
        "    from EvoDAG import EvoDAGClassifier\n"
        "except ImportError as error:\n"
        "    assert 'EvoDAG[sklearn]' in str(error), error\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    assert result.returncode == 0, result.stdout


def test_unknown_attribute():
    import EvoDAG
    with pytest.raises(AttributeError):
        EvoDAG.NotAName


def test_equals_the_member_of_evodage():
    from EvoDAG import EvoDAGE
    X, y = load_iris(return_X_y=True)
    members = EvoDAGE(n_estimators=2, seed=1, **TINY).fit(X, y).models
    for k, member in enumerate(members):
        ours = EvoDAGClassifier(random_state=1 + k, **TINY).fit(X, y)
        assert (ours.predict(X) == member.predict(X)).all()


# ---------------------------------------------------------------------------
# EnsembleClassifier and EnsembleRegressor
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from EvoDAG import EnsembleClassifier, EnsembleRegressor, EvoDAGE  # noqa: E402


@parametrize_with_checks([
    EnsembleClassifier(EvoDAGClassifier(**TINY), n_estimators=2),
    EnsembleRegressor(EvoDAGRegressor(**TINY), n_estimators=2),
    EnsembleClassifier(DecisionTreeClassifier(), n_estimators=3),
    EnsembleRegressor(DecisionTreeRegressor(), n_estimators=3),
])
def test_ensemble_sklearn_checks(estimator, check):
    check(estimator)


def test_ensemble_matches_evodage():
    X, y = load_iris(return_X_y=True)
    ours = EnsembleClassifier(EvoDAGClassifier(**TINY), n_estimators=3,
                              random_state=2).fit(X, y)
    theirs = EvoDAGE(n_estimators=3, seed=2, **TINY).fit(X, y)
    # the decision function is exactly the median of EvoDAGE ...
    median = theirs.decision_function(X)
    np.testing.assert_allclose(ours.decision_function(X), median)
    # ... and predict is its argmax (not the majority vote of EvoDAGE)
    assert (ours.predict(X) == ours.classes_[median.argmax(axis=1)]).all()
    assert (ours.predict_proba(X).argmax(axis=1) == median.argmax(axis=1)).all()
    np.testing.assert_allclose(ours.predict_proba(X).sum(axis=1), 1)
    X, y = load_diabetes(return_X_y=True)
    ours = EnsembleRegressor(EvoDAGRegressor(**TINY), n_estimators=3,
                             random_state=2).fit(X, y)
    theirs = EvoDAGE(n_estimators=3, seed=2, classifier=False,
                     **TINY).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X))


def test_default_estimator_and_member_seeds():
    X, y = load_iris(return_X_y=True)
    ens = EnsembleClassifier(EvoDAGClassifier(**TINY), n_estimators=3,
                             random_state=5)
    assert ens.fit(X, y) is ens
    assert [m.random_state for m in ens.estimators_] == [5, 6, 7]
    assert ens.get_params()["estimator__popsize"] == 10
    assert ens.n_features_in_ == 4 and ens.classes_.tolist() == [0, 1, 2]
    # the parameter of the user is not modified
    assert ens.estimator.random_state == 0
    ens = EnsembleRegressor(EvoDAGRegressor(**TINY), n_estimators=2,
                            random_state=None)
    X, y = load_diabetes(return_X_y=True)
    ens.fit(X, y)
    assert [m.random_state for m in ens.estimators_] == [0, 1]
    # the default estimator is the one of the class
    assert isinstance(EnsembleClassifier()._base_estimator(), EvoDAGClassifier)
    assert isinstance(EnsembleRegressor()._base_estimator(), EvoDAGRegressor)


def test_n_jobs_does_not_change_results():
    X, y = load_iris(return_X_y=True)
    kw = dict(n_estimators=3, random_state=1)
    a = EnsembleClassifier(EvoDAGClassifier(**TINY), n_jobs=1, **kw).fit(X, y)
    b = EnsembleClassifier(EvoDAGClassifier(**TINY), n_jobs=2, **kw).fit(X, y)
    assert (a.predict(X) == b.predict(X)).all()
    np.testing.assert_array_equal(a.decision_function(X),
                                  b.decision_function(X))
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    X, y = load_diabetes(return_X_y=True)
    a = EnsembleRegressor(EvoDAGRegressor(**TINY), n_jobs=1, **kw).fit(X, y)
    b = EnsembleRegressor(EvoDAGRegressor(**TINY), n_jobs=2, **kw).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


class _Voter(ClassifierMixin, BaseEstimator):
    """Classifier without ``decision_function`` whose answers are given:
    ``votes`` is the class it predicts for every example and ``proba`` its
    probabilities (one row for all)."""

    def __init__(self, votes=0, proba=(0.5, 0.5)):
        self.votes = votes
        self.proba = proba

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.votes)

    def predict_proba(self, X):
        return np.tile(self.proba, (len(X), 1))


class _Fixed(_Voter):
    "A ``_Voter`` that has a ``decision_function``: ``score`` for every example"

    def __init__(self, votes=0, proba=(0.5, 0.5), score=0.0):
        super().__init__(votes=votes, proba=proba)
        self.score = score

    def decision_function(self, X):
        return np.tile(np.asarray(self.score, dtype=float), (len(X), 1)
                       ).reshape((len(X),) + np.shape(self.score))


def _ensemble_with_members(cls, members, y=(0, 1, 2, 0), **kwargs):
    ens = cls(n_estimators=len(members), **kwargs)
    ens.fit(np.zeros((len(y), 2)), np.array(y))
    ens.estimators_ = members
    return ens


def test_majority_vote_without_decision_function():
    X = np.zeros((3, 2))
    members = [_Voter(votes=v, proba=p) for v, p in
               [(2, (0., 0., 1.)), (1, (0., 1., 0.)), (1, (0.2, 0.6, 0.2))]]
    ens = _ensemble_with_members(EnsembleClassifier, members,
                                 estimator=_Voter())
    assert not hasattr(ens, "decision_function")
    assert (ens.predict(X) == 1).all()
    np.testing.assert_allclose(ens.predict_proba(X)[0], [0, 2 / 3, 1 / 3])
    # soft voting: the mean of the probabilities, and predict is its argmax
    ens.set_params(voting="soft")
    np.testing.assert_allclose(ens.predict_proba(X)[0],
                               [0.2 / 3, 1.6 / 3, 1.2 / 3])
    np.testing.assert_allclose(ens.predict_proba(X).sum(axis=1), 1)
    assert (ens.predict(X) == 1).all()
    # a tie goes to the lowest class
    members = [_Voter(votes=v) for v in (2, 1, 2, 1)]
    ens = _ensemble_with_members(EnsembleClassifier, members,
                                 estimator=_Voter())
    assert (ens.predict(X) == 1).all()
    ens.set_params(voting="hard")
    np.testing.assert_allclose(ens.predict_proba(X)[0], [0, 0.5, 0.5])


def test_predict_follows_the_median_decision():
    X = np.zeros((3, 2))
    # the plurality (two votes) is class 1, the median decision says class 2
    members = [_Fixed(votes=1, score=(0., 5., 1.)),
               _Fixed(votes=1, score=(0., 2., 1.)),
               _Fixed(votes=2, score=(0., 0., 9.))]
    ens = _ensemble_with_members(EnsembleClassifier, members,
                                 estimator=_Fixed())
    np.testing.assert_allclose(ens.decision_function(X)[0], [0, 2, 1])
    assert (ens.predict(X) == 1).all()
    members[1] = _Fixed(votes=1, score=(0., 0.5, 1.))
    np.testing.assert_allclose(ens.decision_function(X)[0], [0, 0.5, 1])
    assert (ens.predict(X) == 2).all()
    proba = ens.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_allclose(proba[0], np.exp([0, .5, 1]) / np.exp(
        [0, .5, 1]).sum())
    assert (proba.argmax(axis=1) == 2).all()


def test_tied_votes_with_positive_median_decision():
    "The case of the review: the votes tie (class 0 wins a vote) but the median is positive"
    X = np.zeros((3, 2))
    members = [_Fixed(votes=0, score=-1.), _Fixed(votes=1, score=3.)]
    ens = _ensemble_with_members(EnsembleClassifier, members, y=(0, 1, 0, 1),
                                 estimator=_Fixed())
    assert (ens.decision_function(X) == 1.).all()
    assert (ens.predict(X) == 1).all()
    proba = ens.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_allclose(proba[:, 1], 1 / (1 + np.exp(-1.)))
    assert (proba.argmax(axis=1) == 1).all()
    # a decision of exactly zero is the first class in the three outputs
    members = [_Fixed(votes=0, score=-1.), _Fixed(votes=1, score=1.)]
    ens = _ensemble_with_members(EnsembleClassifier, members, y=(0, 1, 0, 1),
                                 estimator=_Fixed())
    assert (ens.predict(X) == 0).all()
    assert (ens.decision_function(X) == 0).all()
    assert (ens.predict_proba(X).argmax(axis=1) == 0).all()


def test_soft_voting_needs_predict_proba_and_validation():
    X, y = load_iris(return_X_y=True)
    from sklearn.svm import LinearSVC
    ens = EnsembleClassifier(LinearSVC(), n_estimators=2, voting="soft")
    ens.fit(X, y)
    with pytest.raises(AttributeError):
        ens.predict_proba(X)
    assert ens.decision_function(X).shape == (150, 3)
    # decision_function is only there when the estimator has it
    assert not hasattr(EnsembleClassifier(_NoDecision()), "decision_function")
    for kw in ({"voting": "other"}, {"n_estimators": 0},
               {"n_estimators": 1.5}, {"random_state": 0.5}):
        with pytest.raises(ValueError):
            EnsembleClassifier(DecisionTreeClassifier(), **kw).fit(X, y)


class _NoDecision(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        return self


class _Reg(RegressorMixin, BaseEstimator):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def test_median_regressor_drops_non_finite_members():
    X = np.zeros((3, 2))
    values = [1., 100., 3., np.nan, np.inf]
    members = [_Reg(v) for v in values]
    ens = _ensemble_with_members(EnsembleRegressor, members, estimator=_Reg())
    assert (ens.predict(X) == 3.).all()
    members = [_Reg(np.inf), _Reg(np.inf)]
    ens = _ensemble_with_members(EnsembleRegressor, members, estimator=_Reg())
    # every member is non-finite: they are all used
    assert np.isinf(ens.predict(X)).all()


def test_ensemble_with_other_estimators():
    X, y = load_iris(return_X_y=True)
    pipe = make_pipeline(StandardScaler(),
                         EnsembleClassifier(DecisionTreeClassifier(),
                                            n_estimators=3))
    grid = GridSearchCV(pipe, {"ensembleclassifier__n_estimators": [2, 3]},
                        cv=2).fit(X, y)
    assert grid.best_params_["ensembleclassifier__n_estimators"] in (2, 3)
    clf = EnsembleClassifier(DecisionTreeClassifier(random_state=0),
                             n_estimators=3, random_state=4).fit(X, y)
    assert [m.random_state for m in clf.estimators_] == [4, 5, 6]
    assert cross_val_score(clone(clf), X, y, cv=2).shape == (2,)

    X, y = load_diabetes(return_X_y=True)
    ens = EnsembleRegressor(Ridge(), n_estimators=3).fit(X, y)
    # Ridge has no random_state: the clones are identical
    np.testing.assert_allclose(ens.predict(X), Ridge().fit(X, y).predict(X))
    grid = GridSearchCV(EnsembleRegressor(EvoDAGRegressor(early_stopping_rounds=5),
                                          n_estimators=2),
                        {"n_estimators": [1, 2], "estimator__popsize": [10, 20]},
                        cv=2).fit(X, y)
    assert grid.best_params_["estimator__popsize"] in (10, 20)
    assert is_classifier(clf) and is_regressor(ens)


def test_ensemble_string_labels():
    X, y = load_iris(return_X_y=True)
    labels = np.array(["a", "b", "c"])
    ens = EnsembleClassifier(EvoDAGClassifier(**TINY), n_estimators=3)
    ens.fit(X, labels[y])
    assert ens.classes_.tolist() == ["a", "b", "c"]
    proba = ens.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert (ens.classes_[proba.argmax(axis=1)] == ens.predict(X)).all()
