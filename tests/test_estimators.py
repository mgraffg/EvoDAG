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
