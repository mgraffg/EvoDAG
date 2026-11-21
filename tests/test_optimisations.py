"""The optimisations of the engine against the straightforward computation.

Every optimisation must give the same bits as the code it replaced; these
tests compare the fast path with the plain computation on the same inputs
(``benchmarks/ecj2022/parity.py`` checks whole fits). ``test_linalg_solve.py``
has the one of ``compute_weight``."""
import pickle

import numpy as np
import pytest

from EvoDAG import array_utils as au
from EvoDAG.base import EvoDAG as Engine
from EvoDAG.cython_utils import Score


def fitted(classifier=True, **kwargs):
    rs = np.random.RandomState(0)
    X = rs.randn(200, 5)
    if classifier:
        y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1)
    else:
        y = X[:, 0] * X[:, 1] + X[:, 2]
    params = dict(popsize=60, early_stopping_rounds=60, time_limit=None,
                  seed=0)
    params.update(kwargs)
    return Engine.init(classifier=classifier, **params).fit(X, y)


@pytest.mark.parametrize("classifier", [True, False])
def test_the_cached_labels_are_the_plain_ones(classifier):
    engine = fitted(classifier)
    checked = 0
    for node in engine.population.population:
        plain = (au.argmax(node.hy) if classifier
                 else au.sign(node.hy) + 1.0)
        labels = node.labels(classifier)
        assert np.array_equal(labels, plain) and labels.dtype == plain.dtype
        assert node.labels(classifier) is labels  # remembered
        checked += 1
    assert checked > 0


def test_the_labels_are_dropped_with_the_outputs():
    engine = fitted()
    node = engine.population.population[0]
    node.labels()
    assert node._labels_cache is not None
    node.hy = [x * 2.0 for x in node.hy]
    assert node._labels_cache is None
    node.labels()
    node.hy = None
    assert node._labels_cache is None
    # the nodes released by the population keep nothing of their outputs
    released = [v for v in engine.population.hist
                if v.hy is None and v is not node]
    assert released
    for v in released:
        assert v._labels_cache is None and v._density_cache is None


def test_models_do_not_carry_the_caches():
    engine = fitted()
    for node in engine.population.population:
        node.labels()
    model = engine.model()
    for node in model._hist:
        assert "_labels_cache" not in vars(node)
        assert "_density_cache" not in vars(node)
    clone = pickle.loads(pickle.dumps(model))
    assert len(pickle.dumps(clone)) == len(pickle.dumps(model))


def test_selection_uses_the_same_numbers():
    "The agreement tournament picks what the plain computation would"
    engine = fitted()
    pop = engine.population.population
    first = pop[3].labels()
    score = engine._bagging_fitness.score
    index = engine._mask_ts_index
    for x in pop:
        plain = score.accuracy(first, au.argmax(x.hy), index)[0]
        fast = score._accuracy_index(first, x.labels(), index)
        assert plain == fast


def test_accuracy_index_equals_the_first_accuracy():
    rs = np.random.RandomState(0)
    for k in (2, 3, 7):
        score = Score(k)
        for n in (1, 50, 1000):
            y = rs.randint(0, k, n).astype(float)
            hy = rs.randint(0, k, n).astype(float)
            for index in (np.arange(n)[rs.rand(n) < 0.5],
                          np.arange(n), np.arange(n)[::3],
                          np.array([0, n + 5, n + 9])):
                index = index.astype(np.uint32)
                expected = score.accuracy(y, hy, index)[0]
                got = score._accuracy_index(y, hy, index)
                assert np.array_equal(expected, got, equal_nan=True)
    # no positions at all: 0 / 0 in both
    empty = np.zeros(0, dtype=np.uint32)
    score = Score(2)
    y = np.array([0.0, 1.0])
    assert np.isnan(score.accuracy(y, y, empty)[0])
    assert np.isnan(score._accuracy_index(y, y, empty))


def test_regression_index_is_remembered_for_one_mask():
    engine = fitted(False)
    index = engine._regression_index()
    assert index.dtype == np.uint32
    assert np.array_equal(index, np.flatnonzero(engine._mask).astype(np.uint32))
    assert engine._regression_index() is index
    engine._mask = np.ones(10)
    other = engine._regression_index()
    assert other is not index and other.shape == (10,)


def test_the_density_does_not_keep_the_outputs_alive():
    import gc
    import weakref
    engine = fitted()
    pop = engine.population
    node = [v for v in pop.population if v is not pop.estopping][0]
    node.density
    assert node._density_cache is not None
    ref = weakref.ref(node.hy[0])
    pop.clean(node)
    assert node.hy is None and node._density_cache is None
    gc.collect()
    assert ref() is None


def test_mul_is_the_product_with_zeros_that_stay_zero():
    "``array_utils.mul`` against the plain loop, with the sign of zero"
    rs = np.random.RandomState(0)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-200, -1e-200,
                        1e200, -1e200, 3.0, -2.5])
    a = np.concatenate([special.repeat(len(special)),
                        rs.randn(100) * (rs.rand(100) < 0.5)])
    b = np.concatenate([np.tile(special, len(special)),
                        rs.randn(100) * (rs.rand(100) < 0.5)])
    expected = np.array([0.0 if x == 0 or y == 0 else x * y
                         for x, y in zip(a, b)])
    with np.errstate(all="ignore"):
        got = au.mul(a, b)
    assert np.array_equal(got, expected, equal_nan=True)
    assert (np.signbit(got) == np.signbit(expected)).all()
    assert not np.signbit(got[(a == 0) | (b == 0)]).any()


def test_mean_without_the_method_overhead():
    rs = np.random.RandomState(0)
    for n in (1, 2, 3, 7, 64, 3823, 10000):
        for _ in range(50):
            u = rs.randn(n) * 10.0 ** rs.randint(-6, 6)
            got = Engine._mean(u)
            assert got == u.mean() and np.signbit(got) == np.signbit(u.mean())


def plain_pearson(engine, reference, x):
    "The measure as it was written before the reference was prepared"
    value = 0.0
    for u, v in zip(engine._masked_outputs(engine.population.population[x]),
                    reference):
        if np.dot(u, u) == 0 or np.dot(v, v) == 0:
            value += 99.0
            continue
        uc = u - u.mean()
        vc = v - v.mean()
        nu = np.sqrt(np.dot(uc, uc))
        nv = np.sqrt(np.dot(vc, vc))
        if nu == 0 or nv == 0:
            value += 99.0
        else:
            value += abs(np.dot(uc, vc) / (nu * nv))
    return value


def plain_angle(engine, reference, x, target):
    value = 0.0
    for e, r in zip(engine._relative_semantics(
            engine.population.population[x], target), reference):
        norm = np.sqrt(np.dot(e, e))
        if norm > 0.0:
            value += float(np.arccos(np.clip(np.dot(e / norm, r), -1.0, 1.0)))
    return value


def test_pearson_and_angle_equal_the_plain_formulas():
    engine = fitted(selection="pearson")
    pop = engine.population.population
    for first in (0, 3, 11):
        reference = engine._pearson_reference(first)
        prepared = engine._pearson_prepare(reference)
        for x in range(len(pop)):
            plain = plain_pearson(engine, reference, x)
            assert engine._pearson_prepared_value(prepared, x) == plain
            assert engine._pearson_value(reference, x) == plain
    # a reference without variation adds 99 per output
    zero = [np.zeros(len(engine._mask_ts))] * 3
    assert engine._pearson_value(zero, 0) == 99.0 * 3
    target = engine._angle_target()
    for first in (0, 3, 11):
        reference = engine._angle_reference(first, target)
        if reference is None:
            continue
        for x in range(len(pop)):
            assert engine._angle_value(reference, x, target) == \
                plain_angle(engine, reference, x, target)
