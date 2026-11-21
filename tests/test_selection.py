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
import warnings
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


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_fitness_selection_equals_legacy(seed):
    a = fit(seed=seed, selection='fitness', orthogonal_selection=True)
    b = fit(seed=seed, orthogonal_selection=False,
            orthogonal_dot_selection=False)
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
    # a zero candidate against a reference that is not zero: for the cosine
    # the zero outputs add nothing and the value is 99 only when all are zero
    ref = engine._cosine_reference(3)
    v = engine.population.population[0]
    n = len(engine._mask_ts)
    v.hy = [np.zeros(n) for _ in v.hy]
    assert engine._cosine_value(ref, 0) == 99.0
    saved = engine.population.population[7].hy
    v.hy = [np.zeros(n)] + [h.copy() for h in saved[1:]]
    assert abs(engine._cosine_value(ref, 0) - brute_cosine(engine, 3, 0)) \
        < 1e-12
    assert 0.0 < engine._cosine_value(ref, 0) < 99.0
    # for Pearson each zero output adds 99
    v.hy = [np.zeros(n) for _ in v.hy]
    assert engine._pearson_value(engine._pearson_reference(3), 0) == 99.0 * 3
    v.hy = [np.zeros(n)] + [h.copy() for h in saved[1:]]
    value = engine._pearson_value(engine._pearson_reference(3), 0)
    assert 99.0 <= value < 99.0 + 2 and \
        abs(value - brute_pearson(engine, 3, 0)) < 1e-12
    # a constant output is constant after the mask only where the mask is
    # constant, so it is a normal output for the measure
    v.hy = [np.full(n, 2.0) for _ in v.hy]
    assert abs(engine._pearson_value(engine._pearson_reference(3), 0) -
               brute_pearson(engine, 3, 0)) < 1e-12


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
def test_reference_is_the_first_argument(selection, monkeypatch):
    engine = fit()
    monkeypatch.setattr(Mul, 'nargs', 2)
    monkeypatch.setattr(Add, 'nargs', 5)

    def first_draw():
        # the reference is individual 3; the angle scheme draws its
        # candidates with the same function, so the rest are random
        calls = []

        def draw():
            calls.append(1)
            if len(calls) == 1:
                return 3
            return np.random.randint(engine.population.popsize)
        return draw
    engine._first_draw = first_draw
    method = getattr(engine, '_get_args_' + selection)
    sizes = set()
    for _ in range(50):
        args = method(Add)
        # a tournament whose winner was already chosen is repeated a limited
        # number of times, so a call may return fewer than five arguments
        assert args is not None and 2 <= len(args) <= 5 and args[0] == 3 \
            and len(set(args)) == len(args)
        sizes.add(len(args))
    if selection != 'angle':
        # the reference and four winners: the five arguments of the sum
        assert max(sizes) == 5
    # a function of two arguments gets the reference and one winner
    args = method(Mul)
    assert len(args) == 2 and args[0] == 3 and args[1] != 3


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


def two_candidates(engine, monkeypatch, *lists):
    "The tournaments draw these lists of candidates, one after the other"
    draws = itertools.cycle(lists)
    monkeypatch.setattr(engine.population, 'random',
                        lambda: list(next(draws)))


@pytest.mark.parametrize("selection,name", (
    ('cosine', '_cosine_value'), ('pearson', '_pearson_prepared_value')))
def test_tournament_min_picks_the_smallest(selection, name, monkeypatch):
    engine = fit(selection=selection)
    monkeypatch.setattr(Add, 'nargs', 2)
    monkeypatch.setattr(Add, 'min_nargs', 1)
    engine._first_draw = lambda: (lambda: 3)
    method = getattr(engine, '_get_args_' + selection)
    for values, expected in (({4: 0.9, 7: 0.1}, [3, 7]),
                             ({4: 0.1, 7: 0.9}, [3, 4]),
                             ({4: 0.5, 7: 0.5}, [3, 4])):
        two_candidates(engine, monkeypatch, [4, 7])
        monkeypatch.setattr(engine, name, lambda ref, x, v=values: v[x])
        assert method(Add) == expected
    # the generic tournament
    two_candidates(engine, monkeypatch, [4, 7])
    assert engine._tournament_min({4: 0.9, 7: 0.1}.get) == 7
    two_candidates(engine, monkeypatch, [4, 7])
    assert engine._tournament_min({4: 0.5, 7: 0.5}.get) == 4


# the paper draws ten candidates; the number is spelled out so that a change
# of the constant of the engine is noticed
ANGLE_CANDIDATES = 10


def test_angle_draws_ten_candidates_and_picks_the_largest(monkeypatch):
    engine = fit(selection='angle')
    monkeypatch.setattr(Add, 'nargs', 2)
    monkeypatch.setattr(Add, 'min_nargs', 1)
    draws = []

    def first_draw():
        counter = itertools.count()

        def draw():
            draws.append(next(counter))
            return draws[-1]
        return draw
    monkeypatch.setattr(engine, '_first_draw', first_draw)
    monkeypatch.setattr(engine, '_angle_value',
                        lambda reference, x, target: float(x))
    args = engine._get_args_angle(Add)
    assert len(draws) == 1 + ANGLE_CANDIDATES
    assert args == [0, ANGLE_CANDIDATES]
    monkeypatch.setattr(engine, '_angle_value',
                        lambda reference, x, target: -float(x))
    del draws[:]
    assert engine._get_args_angle(Add) == [0, 1]


def test_novelty_picks_the_largest(monkeypatch):
    engine = fit(selection='novelty')
    values = {4: 1.0, 7: 3.0, 2: 0.5}
    monkeypatch.setattr(engine, '_novelty_value', lambda x: values[x])
    two_candidates(engine, monkeypatch, [4, 7], [4, 2])
    assert engine._get_args_novelty(Add)[:2] == [7, 4]
    values.update({4: 5.0})
    two_candidates(engine, monkeypatch, [4, 7], [2, 7])
    assert engine._get_args_novelty(Add)[:2] == [4, 7]


def test_first_selection_chooses_the_draw():
    engine = fit(selection='cosine', first_selection='fitness')
    assert engine._first_draw() == engine.population.tournament
    engine = fit(selection='cosine', first_selection='random')
    assert engine._first_draw() == engine.population.random_selection


def forbid(*_, **__):
    raise AssertionError("the wrong draw was used")


def test_other_functions_use_the_right_draw(monkeypatch):
    engine = fit(selection='cosine')
    monkeypatch.setattr(Mul, 'nargs', 2)
    monkeypatch.setattr(engine.population, 'tournament', forbid)
    assert len(engine.get_args(Mul)) == 2
    # without the random generations of the defaults the tournament is used
    engine = fit(selection='fitness', random_generations=0)
    monkeypatch.setattr(Mul, 'nargs', 2)
    monkeypatch.setattr(engine.population, 'random_selection', forbid)
    assert engine.get_args(Add) is not None
    assert len(engine.get_args(Mul)) == 2
    engine = fit(selection='random')
    monkeypatch.setattr(engine.population, 'tournament', forbid)
    assert engine.get_args(Add) is not None


def test_angle_falls_back_to_random(monkeypatch):
    engine = fit(selection='angle')
    monkeypatch.setattr(engine, '_angle_reference',
                        lambda first, target: None)
    monkeypatch.setattr(engine.population, 'tournament', forbid)
    args = engine._get_args_angle(Add)
    assert isinstance(args, list) and len(args) >= Add.min_nargs


def test_huge_outputs_do_not_warn(monkeypatch):
    engine = fit(selection='cosine')
    v = engine.population.population[5]
    v.hy = [h * 1e200 for h in v.hy]
    engine._first_draw = lambda: (lambda: 5)
    pop = engine.population
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for name in ('cosine', 'pearson', 'angle'):
            engine._selection = name
            for _ in range(20):
                engine._get_args_selection(Add)
            # the huge individual is a candidate and the reference
            monkeypatch.setattr(pop, 'random', lambda: [5, 6, 7])
            engine._get_args_selection(Add)
            monkeypatch.undo()
    with np.errstate(over='ignore', invalid='ignore'):
        assert engine._cosine_value(engine._cosine_reference(3), 5) == 0.0


def test_agreement_in_regression():
    rng = np.random.RandomState(0)
    Xr = rng.randn(80, 3)
    yr = Xr[:, 0] + 0.1 * rng.randn(80)

    def run(**kw):
        engine = RGP.init(classifier=False, popsize=20, seed=0,
                          early_stopping_rounds=20, time_limit=None, **kw)
        return engine.fit(Xr, yr).predict(Xr)
    a = run(selection='agreement')
    assert np.isfinite(a).all() and np.array_equal(a, run(
        selection='agreement'))
    # the same as the legacy flag
    assert np.array_equal(a, run(orthogonal_selection=True))


@pytest.mark.parametrize("kwargs", (
    dict(selection='agreement'), dict(selection='cosine'),
    dict(selection='pearson'), dict(selection='angle'),
    dict(selection=None, orthogonal_selection=True),
    dict(selection=None, orthogonal_selection=False,
         orthogonal_dot_selection=True)))
def test_a_fit_builds_sums_with_all_their_arguments(kwargs):
    # the sum is given three arguments: the reference and two winners. When
    # the reference was left out no sum could have more than two
    engine = fit(Add=3, **kwargs)
    sizes = [len(v.variable) for v in engine.population.hist
             if isinstance(v, Add)]
    assert sizes and max(sizes) == 3
