# Copyright 2026 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of the selection schemes of the heuristics article (``selection``)."""
import itertools
import numpy as np
import pytest
from test_root import X, cl
from EvoDAG import RGP
from EvoDAG.node import Add, Mul, Min, Max, Hypot, NaiveBayes, Centroid
from EvoDAG.node import Atan2, Sin

SELECTIONS = ('fitness', 'random', 'agreement', 'cosine', 'pearson',
              'angle', 'novelty')


def fit(seed=0, **kwargs):
    "Fit a tiny classification engine and return it"
    engine = RGP.init(popsize=30, early_stopping_rounds=30, time_limit=None,
                      seed=seed, **kwargs)
    return engine.fit(X, cl)


def test_defaults():
    for engine in (RGP.init(), RGP()):
        assert engine._selection is None
        assert engine._first_selection == 'random'
        assert engine._heuristic_functions == 'tailored'
    params = RGP(selection='cosine', first_selection='fitness',
                 heuristic_functions='all').get_params()
    assert params['selection'] == 'cosine'
    assert params['first_selection'] == 'fitness'
    assert params['heuristic_functions'] == 'all'


def test_invalid_values():
    with pytest.raises(ValueError, match="selection"):
        RGP(selection='x')
    with pytest.raises(ValueError, match="first_selection"):
        RGP(first_selection='x')
    with pytest.raises(ValueError, match="heuristic_functions"):
        RGP(heuristic_functions='x')
    with pytest.raises(ValueError, match="novelty"):
        RGP(selection='novelty', classifier=False)
    with pytest.raises(ValueError, match="SteadyState"):
        RGP(selection='novelty', population_class='Generational')


@pytest.mark.parametrize("selection,first,functions,negative", list(
    itertools.product(SELECTIONS, ('random', 'fitness'),
                      ('tailored', 'all'), (True, False))))
def test_every_selection_fits_and_is_deterministic(selection, first,
                                                   functions, negative):
    kw = dict(selection=selection, first_selection=first,
              heuristic_functions=functions, negative_selection=negative)
    a = fit(**kw).model().decision_function(X)
    b = fit(**kw).model().decision_function(X)
    assert np.array_equal(np.array(a), np.array(b))


def test_fitness_selection_equals_legacy():
    a = fit(selection='fitness', orthogonal_selection=True)
    b = fit(orthogonal_selection=False, orthogonal_dot_selection=False)
    assert np.array_equal(np.array(a.model().decision_function(X)),
                          np.array(b.model().decision_function(X)))
    assert len(a.population.hist) == len(b.population.hist)


def test_selection_overrides_legacy_flags():
    a = fit(selection='random', orthogonal_selection=True)
    b = fit(selection='random', orthogonal_selection=False)
    assert np.array_equal(np.array(a.model().decision_function(X)),
                          np.array(b.model().decision_function(X)))


def masked(engine, x):
    m = engine._mask_ts
    return [h * m for h in engine.population.population[x].hy]


def brute_cosine(engine, first, x):
    value, counter = 0.0, 0
    for u, f in zip(masked(engine, x), masked(engine, first)):
        nu, nf = np.linalg.norm(u), np.linalg.norm(f)
        if nu and nf:
            value += abs(u @ f) / (nu * nf)
        else:
            counter += 1
    return 99.0 if counter == len(masked(engine, first)) else value


def brute_pearson(engine, first, x):
    value = 0.0
    for u, f in zip(masked(engine, x), masked(engine, first)):
        uc, fc = u - u.mean(), f - f.mean()
        if u @ u == 0 or f @ f == 0 or not np.linalg.norm(uc) \
                or not np.linalg.norm(fc):
            value += 99.0
        else:
            value += abs(uc @ fc / (np.linalg.norm(uc) * np.linalg.norm(fc)))
    return value


def brute_angle(engine, first, x):
    m = engine._mask_ts
    target = [y * m for y in engine.y]

    def rel(p):
        return [t - h for t, h in zip(target, masked(engine, p))]
    value = 0.0
    for e, r in zip(rel(x), rel(first)):
        ne = np.linalg.norm(e)
        if ne > 0:
            value += np.arccos(np.clip((e / ne) @ (r / np.linalg.norm(r)),
                                       -1, 1))
    return value


def test_measures_against_brute_force():
    engine = fit()
    first = 3
    others = [0, 1, 5, 7, 10, 20, 29]
    ref = engine._cosine_reference(first)
    for x in others:
        assert abs(engine._cosine_value(ref, x) -
                   brute_cosine(engine, first, x)) < 1e-12
    ref = engine._pearson_reference(first)
    for x in others:
        assert abs(engine._pearson_value(ref, x) -
                   brute_pearson(engine, first, x)) < 1e-12
    target = engine._angle_target()
    ref = engine._angle_reference(first, target)
    assert ref is not None
    for x in others:
        assert abs(engine._angle_value(ref, x, target) -
                   brute_angle(engine, first, x)) < 1e-12


def test_degenerate_measures():
    engine = fit()
    zero = [np.zeros(len(engine._mask_ts))] * 3
    assert engine._cosine_value((zero, [0.0] * 3), 0) == 99.0
    assert engine._pearson_value(zero, 0) == 99.0 * 3


def novelty_state(engine):
    pop = engine.population.population
    index = engine._mask_ts_index
    labels = np.array([np.argmax([h[index] for h in v.hy], axis=0)
                       for v in pop])
    counts = np.zeros((len(index), len(pop[0].hy)), dtype=int)
    for j in range(len(index)):
        counts[j] = np.bincount(labels[:, j], minlength=counts.shape[1])
    return labels, counts


def test_novelty_counts_are_consistent():
    engine = RGP.init(popsize=30, early_stopping_rounds=60, time_limit=None,
                      seed=0, selection='novelty').fit(X, cl)
    assert len(engine.population.hist) > 60
    labels, counts = novelty_state(engine)
    n = engine._novelty_n
    assert n == len(engine.population.population)
    assert np.array_equal(engine._novelty_labels[:n], labels)
    assert np.array_equal(engine._novelty_counts, counts)
    assert engine._novelty_labels.dtype == np.uint8
    # the value of an individual follows its definition
    x = 4
    p = counts[np.arange(labels.shape[1]), labels[x]] / n
    assert abs(engine._novelty_value(x) - np.sum(1 / (p + 1e-5))) < 1e-6 * \
        engine._novelty_value(x)


def test_no_novelty_state_unless_requested():
    engine = fit(selection='cosine')
    assert getattr(engine, '_novelty_labels', None) is None


def test_novelty_state_grows():
    engine = fit(selection='novelty')
    n = engine._novelty_n
    engine._novelty_labels = engine._novelty_labels[:n]
    v = engine.population.population[0]
    engine._novelty_add(v)
    assert engine._novelty_labels.shape[0] == 2 * n
    assert engine._novelty_n == n + 1


def test_heuristic_functions_all(monkeypatch):
    # ``init`` sets the arity of the node classes from the parameters, so
    # the arities are fixed here
    for f in (Mul, Min, Max, Hypot, Atan2):
        monkeypatch.setattr(f, 'nargs', 2)
    monkeypatch.setattr(Sin, 'nargs', 1)
    tailored = RGP(selection='cosine')
    everything = RGP(selection='cosine', heuristic_functions='all')
    for f in (Add, NaiveBayes, Centroid):
        assert tailored._heuristic_applies(f)
        assert everything._heuristic_applies(f)
    for f in (Mul, Min, Max, Hypot, Atan2):
        assert not tailored._heuristic_applies(f)
        assert everything._heuristic_applies(f)
    assert not everything._heuristic_applies(Sin)


@pytest.mark.parametrize("selection", ('agreement', 'cosine', 'pearson',
                                       'angle'))
def test_reference_is_not_an_argument(selection, monkeypatch):
    engine = fit()
    monkeypatch.setattr(Mul, 'nargs', 2)
    engine._first_draw = lambda: (lambda: 3)
    method = getattr(engine, '_get_args_' + selection)
    for _ in range(50):
        args = method(Add)
        assert args is None or (3 not in args and len(args) <= Add.nargs - 1)
    assert method(Mul) is None


def test_random_selection_for_the_other_functions(monkeypatch):
    engine = fit(selection='cosine')
    monkeypatch.setattr(Mul, 'nargs', 2)
    calls = []
    monkeypatch.setattr(engine, '_get_args_cosine',
                        lambda func: calls.append(func))
    engine.get_args(Add)
    engine.get_args(Mul)
    assert calls == [Add]
    args = engine.get_args(Mul)
    assert len(args) == 2
