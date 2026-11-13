# Copyright 2024 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Golden reference outputs for EvoDAG.

This module defines every golden case and regenerates the golden files that
``test_golden.py`` compares the library against::

    python EvoDAG/tests/golden/record.py            # rewrite the golden files
    python EvoDAG/tests/golden/record.py --output D # write them into D

Each *group* (see ``GROUPS``) is stored as a pair of files:

* ``<group>.npz``: every number, as dense numpy arrays (one array per key);
* ``<group>.json``: everything that is not a number: structure (class names,
  variables), booleans (did ``eval`` succeed?), the length of every list and
  the exceptions raised, if any.

Everything the library returns goes through ``store`` which converts it to
dense numpy, whether it is a ``SparseArray`` (anything with ``full_array``)
or a plain ``np.ndarray``; the same recorder and the same tests therefore
work with the SparseArray implementation and with a numpy one. The few
places that must *build* library inputs are ``vec`` and ``kernel_index``;
``vec`` follows the library (see ``USES_SPARSEARRAY``).

Training is deterministic for a fixed seed, so the end to end cases do not
depend on ``PYTHONHASHSEED``.
"""
import argparse
import array
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

import EvoDAG.node as node_module
from EvoDAG import RGP, EvoDAG
from EvoDAG.model import EvoDAGE
from EvoDAG.node import (Variable, Function, Add, Mul, Div, Atan2, Hypot,
                         Acos, Asin, Atan, Cos, Sin, Tan, Cosh, Sinh, Tanh,
                         Acosh, Asinh, Atanh, Exp, Expm1, Log, Log2, Log10,
                         Log1p, Lgamma, Sqrt, Sq, Sign, Fabs, Ceil, Floor,
                         Min, Max, Argmax, Argmin, NaiveBayes, NaiveBayesMN,
                         Centroid, MultipleVariables)

GOLDEN_DIR = Path(__file__).resolve().parent

# Whether the library still stores its vectors as SparseArray objects. The
# only thing that depends on it is how ``vec`` builds library inputs.
USES_SPARSEARRAY = hasattr(node_module, 'SparseArray')

# ``nargs`` of every node class as defined in EvoDAG/node.py. Other tests, and
# ``EvoDAG.init`` (through ``RandomParameterSearch.process_params``), change
# these class attributes, so every case starts from this table.
PRISTINE_NARGS = {
    'Acos': 1, 'Acosh': 1, 'Add': 5, 'Argmax': 2, 'Argmin': 2, 'Asin': 1,
    'Asinh': 1, 'Atan': 1, 'Atan2': 2, 'Atanh': 1, 'Ceil': 1, 'Centroid': 2,
    'Cos': 1, 'Cosh': 1, 'Div': 2, 'Exp': 1, 'Expm1': 1, 'Fabs': 1,
    'Floor': 1, 'Hypot': 2, 'Lgamma': 1, 'Log': 1, 'Log10': 1, 'Log1p': 1,
    'Log2': 1, 'Max': 2, 'Min': 2, 'Mul': 2, 'MultipleVariables': 5,
    'NaiveBayes': 2, 'NaiveBayesMN': 2, 'Sign': 1, 'Sin': 1, 'Sinh': 1,
    'Sq': 1, 'Sqrt': 1, 'Tan': 1, 'Tanh': 1}

ONE_ARG = [Acos, Asin, Atan, Cos, Sin, Tan, Cosh, Sinh, Tanh, Acosh, Asinh,
           Atanh, Exp, Expm1, Log, Log2, Log10, Log1p, Lgamma, Sqrt, Sq, Sign,
           Fabs, Ceil, Floor]

# variable lists evaluated by every function that takes several arguments
MULTI_ARGS = {
    Add: [[0, 3], [0, 1, 4], [1, 2, 3], [0, 1, 3, 4, 6]],
    Mul: [[0, 3], [1, 4], [0, 2], [5, 6]],
    Div: [[0, 3], [3, 0], [0, 1], [4, 2], [5, 3]],
    Atan2: [[0, 3], [1, 4], [0, 2]],
    Hypot: [[0, 3], [1, 4], [0, 2]],
    Min: [[0, 3], [1, 4, 6]],
    Max: [[0, 3], [1, 4, 6]],
    Argmax: [[0, 3, 4], [1, 6]],
    Argmin: [[0, 3, 4], [1, 6]],
    MultipleVariables: [[0, 3, 4], [1, 2, 6]]}

NAIVE_BAYES_ARGS = {
    NaiveBayes: [[0, 3], [3, 4, 6], [1, 3, 5]],
    NaiveBayesMN: [[0, 3], [3, 5], [1, 3, 4]],
    Centroid: [[0, 3], [1, 4, 5]]}

# every node class that has to be covered by a golden case
NODE_CLASSES = ONE_ARG + list(MULTI_ARGS) + list(NAIVE_BAYES_ARGS)


def reset_nargs():
    for name, nargs in PRISTINE_NARGS.items():
        getattr(node_module, name).nargs = nargs


# ---------------------------------------------------------------------------
# Conversion between numpy and whatever vectors the library uses
# ---------------------------------------------------------------------------

def vec(a):
    """A library vector with the values of ``a`` (used to build inputs)."""
    a = np.asarray(a, dtype=float)
    if USES_SPARSEARRAY:
        from SparseArray import SparseArray
        return SparseArray.fromlist(a)
    return a


def kernel_index(index):
    """The ``mask`` index the Cython kernels take: ``array('I')`` with the
    sorted positions of the mask."""
    return array.array('I', [int(i) for i in index])


def dense(x):
    "Dense float numpy array of a library vector"
    if hasattr(x, 'full_array'):
        return np.array(x.full_array(), dtype=float)
    return np.array(x)


def density_of(x):
    "Fraction of non zero elements of a vector (or mean over a list)"
    if isinstance(x, (list, tuple)):
        return float(np.mean([density_of(i) for i in x]))
    if hasattr(x, 'density'):
        return float(x.density)
    return float(np.mean(np.asarray(x) != 0))


def jsonable(x):
    if isinstance(x, (list, tuple)):
        return [jsonable(i) for i in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def store(arrays, meta, key, obj):
    """Records ``obj`` under ``key``: numbers into ``arrays`` (dense
    numpy), everything else into ``meta``. Lists are stored recursively,
    each item under ``key.<position>``, and their length in ``meta``."""
    if obj is None or isinstance(obj, (bool, str)):
        meta[key] = obj
    elif isinstance(obj, (list, tuple)):
        meta[key + '#len'] = len(obj)
        for i, o in enumerate(obj):
            store(arrays, meta, '%s.%s' % (key, i), o)
    else:
        arrays[key] = dense(obj)


def pack(obj, leaves):
    "Structure of ``obj`` (JSON) with its numbers appended to ``leaves``"
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [pack(o, leaves) for o in obj]
    a = dense(obj)
    leaves.append(a.ravel())
    return 'x'.join(str(i) for i in a.shape) or 's'


def store_packed(arrays, meta, key, **parts):
    """Like ``store`` for several objects at once: one array with all their
    numbers (the zip overhead of a few hundred small arrays is larger than
    the numbers themselves) and their structure in ``meta``."""
    leaves = []
    meta[key + '.layout'] = {name: pack(obj, leaves)
                             for name, obj in parts.items()}
    arrays[key] = np.concatenate(leaves) if leaves else np.zeros(0)


def run(arrays, meta, key, func):
    """Calls ``func`` and records the exception it raises, if any. Returns
    the result or ``None``."""
    try:
        return func()
    except Exception as exc:  # the exception is part of the behaviour
        meta[key + '.exception'] = type(exc).__name__
        return None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

N_TR = 60
N_TS = 24


def raw_variables(n, rs):
    """Eight variables: gaussian, 70% zeros, all zeros, positive, in
    (-1, 1), huge (overflows exp / cosh), small integers, tiny with zeros."""
    v = np.zeros((n, 8))
    v[:, 0] = rs.randn(n)
    v[:, 1] = rs.randn(n) * (rs.rand(n) > 0.7)
    v[:, 3] = rs.uniform(0.1, 3, n)
    v[:, 4] = rs.uniform(-1, 1, n) * (rs.rand(n) > 0.3)
    v[:, 5] = rs.randn(n) * 400
    v[::4, 5] = 0
    v[:, 6] = rs.randint(-3, 4, n)
    v[:, 7] = rs.randn(n) * 1e-3 * (rs.rand(n) > 0.5)
    return v


def node_problem():
    "Train / test inputs and the targets used at node level"
    rs = np.random.RandomState(0)
    X = raw_variables(N_TR, rs)
    Xts = raw_variables(N_TS, rs)
    y_reg = 1.5 * X[:, 0] + 0.7 * X[:, 3] - X[:, 4] + 0.1 * rs.randn(N_TR)
    score = np.c_[X[:, 0] + X[:, 6] * 0.3, X[:, 3] - 1.5, -X[:, 4] * 2]
    y_cl = (score + 0.3 * rs.randn(N_TR, 3)).argmax(axis=1)
    return X, Xts, y_reg, y_cl


def make_gp(classifier, X, Xts, y, **kwargs):
    reset_nargs()
    gp = RGP(generations=1, popsize=8, classifier=classifier, seed=0, **kwargs)
    gp.X = X
    gp.Xtest = Xts
    gp.y = y
    return gp


def multiple_output_nodes(gp, X, Xts):
    """The raw variables expressed as nodes with one output per class, each
    output a scaled copy of the variable."""
    scales = [1.0, -0.6, 0.35]
    nodes = []
    for j in range(X.shape[1]):
        n = Variable(j)
        n.hy = [vec(X[:, j] * s) for s in scales]
        n.hy_test = [vec(Xts[:, j] * s) for s in scales]
        nodes.append(n)
    return nodes


def record_node(arrays, meta, key, make, args):
    n = run(arrays, meta, key, make)
    if n is None:
        return
    ok = run(arrays, meta, key, lambda: n.eval(args))
    if ok is None:
        return
    meta[key + '.ok'] = bool(ok)
    if not ok:
        return
    parts = dict(weight=n.weight, hy=n.hy, hy_test=n.hy_test)
    meta[key + '.isfinite'] = bool(n.isfinite())
    if isinstance(n, (NaiveBayes, NaiveBayesMN, Centroid)):
        n.normalize()
        parts['normalized'] = n.hy
    store_packed(arrays, meta, key, **parts)


def node_cases(arrays, meta, gp, args, multiple, with_naive_bayes):
    kw = dict(ytr=gp._ytr, mask=gp._mask)
    # the variables themselves, evaluated as the first generation does
    for finite in (True, False):
        for j in range(len(gp.X)):
            key = 'Variable.%s.finite%s' % (j, finite)
            record_node(arrays, meta, key,
                        lambda: Variable(j, finite=finite,
                                         naive_bayes=gp.naive_bayes, **kw),
                        gp.X)
    one_arg_variables = [0, 3, 5] if multiple else range(7)
    for cls in ONE_ARG:
        for finite in (True, False):
            if multiple and not finite:
                continue
            for j in one_arg_variables:
                if not finite and j not in (0, 3, 5):
                    continue
                key = '%s.%s.finite%s' % (cls.__name__, j, finite)
                record_node(arrays, meta, key,
                            lambda: cls(j, finite=finite, **kw), args)
    for cls, lst in MULTI_ARGS.items():
        for variables in lst:
            key = '%s.%s' % (cls.__name__, '-'.join(str(i) for i in variables))
            # MultipleVariables takes the raw variables, also with several
            # outputs: its outputs are created from them
            inputs = gp.X if cls is MultipleVariables else args
            record_node(arrays, meta, key,
                        lambda: cls(variables, finite=True, **kw), inputs)
    if not with_naive_bayes:
        return
    for cls, lst in NAIVE_BAYES_ARGS.items():
        for variables in lst:
            key = '%s.%s' % (cls.__name__, '-'.join(str(i) for i in variables))
            record_node(arrays, meta, key,
                        lambda: cls(variables, finite=True,
                                    naive_bayes=gp.naive_bayes, **kw), args)


def build_nodes_single():
    arrays, meta = {}, {}
    X, Xts, y_reg, y_cl = node_problem()
    gp = make_gp(False, X, Xts, y_reg)
    node_cases(arrays, meta, gp, gp.X, False, False)
    return arrays, meta


def build_nodes_multiple():
    arrays, meta = {}, {}
    X, Xts, y_reg, y_cl = node_problem()
    gp = make_gp(True, X, Xts, y_cl)
    args = multiple_output_nodes(gp, X, Xts)
    node_cases(arrays, meta, gp, args, True, True)
    return arrays, meta


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def kernel_problem():
    rs = np.random.RandomState(1)
    n = 60
    dense_var = rs.randn(n)
    sparse_var = rs.randn(n) * (rs.rand(n) > 0.7)
    positive = rs.uniform(0.1, 3, n) * (rs.rand(n) > 0.4)
    zero = np.zeros(n)
    y = rs.randint(0, 3, n)
    flip = rs.rand(n) < 0.3
    hy = np.where(flip, rs.randint(0, 3, n), y)
    index = np.sort(rs.permutation(n)[:30])
    return dict(n=n, dense=dense_var, sparse=sparse_var, positive=positive,
                zero=zero, y=y, hy=hy, index=index, rs=rs)


def build_kernels():
    from EvoDAG.linalg_solve import compute_weight
    from EvoDAG.cython_utils import (fitness_SSE, fitness_SAE, Score,
                                     SelectNumbers, naive_bayes,
                                     naive_bayes_MN, naive_bayes_Nc,
                                     naive_bayes_mean_std2,
                                     naive_bayes_isfinite,
                                     naive_bayes_isfinite_MN, centroid_coef)
    from EvoDAG.function_selection import FunctionSelection
    from EvoDAG.utils import BER, RSE, tonparray
    arrays, meta = {}, {}
    p = kernel_problem()
    n = p['n']
    rs = p['rs']
    mask = np.zeros(n)
    mask[p['index']] = 1.0
    ytr = 2 * p['dense'] - p['positive'] + 0.1 * rs.randn(n)
    ytr = ytr * mask

    # --- compute_weight
    regressors = {
        'three': [p['dense'], p['sparse'], p['positive']],
        'one': [p['dense']],
        'sparse_only': [p['sparse']],
        'collinear': [p['dense'], p['dense'] * 2.0],
        'with_zero': [p['zero'], p['dense']],
        'only_zero': [p['zero']],
        'nonfinite': [np.where(np.arange(n) == 3, np.inf, p['dense'])],
        'masked_out': [np.where(mask == 0, p['dense'], 0.0)]}
    for name, r in regressors.items():
        key = 'compute_weight.%s' % name
        w = run(arrays, meta, key,
                lambda: compute_weight([vec(x) for x in r], vec(ytr),
                                       vec(mask)))
        if w is None and (key + '.exception') not in meta:
            meta[key] = None
        elif w is not None:
            store(arrays, meta, key, w)

    # --- fitness
    ytrs = [vec(ytr), vec(ytr * -0.5 + 1.0 * mask)]
    hys = [vec(p['dense'] * 0.8), vec(p['sparse'])]
    store(arrays, meta, 'fitness_SSE',
          np.array(fitness_SSE(ytrs, hys, vec(mask))))
    store(arrays, meta, 'fitness_SAE',
          np.array(fitness_SAE(ytrs, hys, [vec(mask), vec(mask * 0.5)])))

    # --- Score
    datasets = {
        'balanced': (3, p['y'], p['hy']),
        'missing_prediction': (3, p['y'], np.where(p['hy'] == 2, 1, p['hy'])),
        'mostly_zero': (3, p['y'], np.where(rs.rand(n) < 0.8, 0, p['hy'])),
        'empty_class': (4, p['y'], p['hy'])}
    for name, (nclasses, y, hy) in datasets.items():
        index = kernel_index(p['index'])
        calls = dict(
            accuracy=lambda s: s.accuracy(vec(y), vec(hy), index),
            a_recall=lambda s: s.a_recall(vec(y), vec(hy), index),
            a_precision=lambda s: s.a_precision(vec(y), vec(hy), index),
            g_recall=lambda s: s.g_recall(vec(y), vec(hy), index),
            g_precision=lambda s: s.g_precision(vec(y), vec(hy), index),
            macroRecallF1=lambda s: s.macroRecallF1(vec(y), vec(hy), index),
            accDotMacroF1=lambda s: s.accDotMacroF1(vec(y), vec(hy), index),
            g_g_recall_precision=lambda s: s.g_g_recall_precision(
                vec(y), vec(hy), index),
            g_F1=lambda s: s.g_F1(vec(y), vec(hy), index),
            a_F1=lambda s: s.a_F1(vec(y), vec(hy), index))
        for i in range(nclasses):
            calls['F1.%s' % i] = lambda s, i=i: s.F1(i, vec(y), vec(hy), index)
            calls['g_recall_precision.%s' % i] = \
                lambda s, i=i: s.g_recall_precision(i, vec(y), vec(hy), index)
        for method, call in calls.items():
            key = 'Score.%s.%s' % (name, method)
            score = Score(nclasses)
            r = run(arrays, meta, key, lambda: call(score))
            if r is None:
                continue
            store(arrays, meta, key, np.array(r))
            store(arrays, meta, key + '.counts', np.concatenate(
                [np.array(getattr(score, attr), dtype=float) for attr in
                 ('precision', 'recall', 'precision2', 'recall2',
                  'precision_den', 'recall_den', 'precision2_den',
                  'recall2_den')]))

    # --- naive bayes coefficients and outputs
    klass = array.array('I', [int(i) for i in p['y']])
    index = kernel_index(p['index'])
    nclass = 3
    variables = {'dense': p['dense'], 'sparse': p['sparse'],
                 'positive': p['positive'], 'zero': p['zero']}
    for name, v in variables.items():
        key = 'naive_bayes_mean_std2.%s' % name
        store(arrays, meta, key,
              naive_bayes_mean_std2(vec(v), klass, index, nclass))
        key = 'naive_bayes_Nc.%s' % name
        store(arrays, meta, key,
              naive_bayes_Nc(vec(v), klass, index, nclass))
        key = 'centroid_coef.%s' % name
        store(arrays, meta, key,
              centroid_coef([vec(v)], klass, index, nclass))
    names = ['dense', 'sparse', 'positive']
    coef = [naive_bayes_mean_std2(vec(variables[k]), klass, index, nclass)
            for k in names]
    store(arrays, meta, 'naive_bayes.three',
          naive_bayes([vec(variables[k]) for k in names], coef, nclass))
    coef_zero = [naive_bayes_mean_std2(vec(variables[k]), klass, index, nclass)
                 for k in ['dense', 'zero']]
    store(arrays, meta, 'naive_bayes.with_zero',
          naive_bayes([vec(variables[k]) for k in ['dense', 'zero']],
                      coef_zero, nclass))
    meta['naive_bayes_isfinite.three'] = [
        bool(naive_bayes_isfinite(c, nclass)) for c in coef]
    meta['naive_bayes_isfinite.with_zero'] = [
        bool(naive_bayes_isfinite(c, nclass)) for c in coef_zero]
    coef = [naive_bayes_Nc(vec(variables[k]), klass, index, nclass)
            for k in names]
    store(arrays, meta, 'naive_bayes_MN.three',
          naive_bayes_MN([vec(variables[k]) for k in names], coef, nclass))
    meta['naive_bayes_isfinite_MN.three'] = [
        bool(naive_bayes_isfinite_MN(c, nclass)) for c in coef]
    coef_zero = [naive_bayes_Nc(vec(variables[k]), klass, index, nclass)
                 for k in ['dense', 'zero']]
    meta['naive_bayes_isfinite_MN.with_zero'] = [
        bool(naive_bayes_isfinite_MN(c, nclass)) for c in coef_zero]
    nonfinite = [array.array('d', [0.1, float('nan'), 0.2]),
                 array.array('d', [0.1, 0.2, 0.3])]
    meta['naive_bayes_isfinite_MN.nan'] = bool(
        naive_bayes_isfinite_MN(nonfinite, nclass))
    zero_std = [array.array('d', [1., 2., 3.]), array.array('d', [1., 0., 3.]),
                array.array('d', [1., 2., 3.])]
    meta['naive_bayes_isfinite.zero_std'] = bool(
        naive_bayes_isfinite(zero_std, nclass))
    store(arrays, meta, 'centroid_coef.three',
          centroid_coef([vec(variables[k]) for k in names], klass, index,
                        nclass))

    # --- utils
    y = p['y']
    store(arrays, meta, 'utils.BER', np.array(BER(y, p['hy'])))
    store(arrays, meta, 'utils.RSE',
          np.array(RSE(p['dense'], p['dense'] * 0.9 + 0.1)))
    store(arrays, meta, 'utils.tonparray', tonparray(vec(p['sparse'])))

    # --- SelectNumbers (python's random module)
    random.seed(0)
    s = SelectNumbers(list(range(20)))
    store(arrays, meta, 'SelectNumbers.get5', np.array(s.get(5)))
    store(arrays, meta, 'SelectNumbers.one', np.array(s.get_one()))
    store(arrays, meta, 'SelectNumbers.get20', np.array(s.get(20)))
    meta['SelectNumbers.empty'] = bool(s.empty())

    # --- FunctionSelection
    reset_nargs()
    fs = FunctionSelection(nfunctions=6, seed=3, tournament_size=3,
                           nargs=[2, 1, 5, 2, 1, 3], density_safe=[0, 2, 4])
    picks = []
    for k in range(12):
        fs[k % 6] = float(k) / 7 - 0.5
        if k == 6:
            fs.density = 0.5
            fs.min_density = 0.8
            fs.unfeasible_functions.add(2)
        picks.append([fs.random_function(), fs.tournament()])
    store(arrays, meta, 'FunctionSelection.picks', np.array(picks))
    store(arrays, meta, 'FunctionSelection.avg_fitness',
          np.array([fs.avg_fitness(i) for i in range(6)]))
    meta['FunctionSelection.comparison'] = [bool(fs.comparison(i, j))
                                            for i in range(6)
                                            for j in range(6)]
    return arrays, meta


# ---------------------------------------------------------------------------
# Masks produced by bagging_fitness
# ---------------------------------------------------------------------------

def record_masks(arrays, meta, prefix, gp):
    for attr in ('_y_klass', '_mask_vs', '_mask_ts', '_mask', '_ytr', '_y',
                 '_labels'):
        if hasattr(gp, attr):
            store(arrays, meta, '%s.%s' % (prefix, attr), getattr(gp, attr))


def build_masks():
    arrays, meta = {}, {}
    X, Xts, y_reg, y_cl = node_problem()
    for fitness_function in ['BER', 'macro-F1', 'macro-RecallF1']:
        reset_nargs()
        gp = RGP(generations=1, popsize=8, classifier=True, seed=2,
                 fitness_function=fitness_function)
        gp.X = X
        gp.y = y_cl
        record_masks(arrays, meta, 'classifier.%s' % fitness_function, gp)
    reset_nargs()
    gp = RGP(generations=1, popsize=8, classifier=False, seed=2)
    gp.X = X
    gp.y = y_reg
    record_masks(arrays, meta, 'regression', gp)
    reset_nargs()
    gp = RGP(generations=1, popsize=8, classifier=False, seed=2,
             multiple_outputs=True)
    gp.X = X
    gp.y = [vec(y_reg), vec(-y_reg * 0.5 + X[:, 3])]
    record_masks(arrays, meta, 'regression_mo', gp)
    reset_nargs()
    gp = RGP(generations=1, popsize=8, classifier=True, seed=2,
             tr_fraction=0.7)
    gp.X = X
    gp.y = [vec(np.where(y_cl == k, 1, -1)) for k in range(3)]
    record_masks(arrays, meta, 'classifier_list', gp)
    return arrays, meta


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def end_to_end_problem():
    rs = np.random.RandomState(7)
    X = rs.randn(300, 8)
    X[:, 2] *= rs.rand(300) > 0.7
    X[:, 5] *= rs.rand(300) > 0.7
    score = np.c_[X[:, 0] + X[:, 1], X[:, 3] - X[:, 4], -X[:, 6] + 0.5 * X[:, 7]]
    y_cl = (score + 0.5 * rs.randn(300, 3)).argmax(axis=1)
    y_reg = 2 * X[:, 0] - X[:, 3] ** 2 + 0.5 * X[:, 1] * X[:, 4] \
        + 0.1 * rs.randn(300)
    return X[:200], X[200:], y_cl[:200], y_reg[:200]


def record_model(arrays, meta, prefix, m):
    meta[prefix + '.classes'] = [n.__class__.__name__ for n in m._hist]
    meta[prefix + '.variables'] = [jsonable(n.variable) for n in m._hist]
    meta[prefix + '.heights'] = [int(n.height) for n in m._hist]
    meta[prefix + '.positions'] = [int(n.position) for n in m._hist]
    meta[prefix + '.classifier'] = bool(m.classifier)
    meta[prefix + '.multiple_outputs'] = bool(m.multiple_outputs)
    meta[prefix + '.nvar'] = int(m.nvar)
    parts = {}
    for k, n in enumerate(m._hist):
        parts['weight.%s' % k] = n.weight
        parts['fitness.%s' % k] = n.fitness
        parts['fitness_vs.%s' % k] = n.fitness_vs
    parts['fitness_vs'] = m.fitness_vs
    store_packed(arrays, meta, prefix + '.nodes', **parts)


def record_predictions(arrays, meta, prefix, m, Xtr, Xte):
    store(arrays, meta, prefix + '.decision_function.train',
          m.decision_function(Xtr))
    store(arrays, meta, prefix + '.decision_function.test',
          m.decision_function(Xte))
    store(arrays, meta, prefix + '.decision_function.stored',
          m.decision_function(None))
    if m.multiple_outputs and not m.classifier:
        return  # Model.predict does not support it: tonparray(list)
    store(arrays, meta, prefix + '.predict.train', m.predict(Xtr))
    store(arrays, meta, prefix + '.predict.test', m.predict(Xte))


def record_fit(arrays, meta, prefix, gp, Xtr, ytr, Xte):
    gp.fit(Xtr, ytr, test_set=Xte)
    m = gp.model()
    record_model(arrays, meta, prefix, m)
    record_predictions(arrays, meta, prefix, m, Xtr, Xte)
    meta[prefix + '.hist'] = len(gp.population.hist)
    store(arrays, meta, prefix + '.density', gp.population.density)
    return m


def end_to_end_classifier(**kwargs):
    arrays, meta = {}, {}
    Xtr, Xte, ytr, _ = end_to_end_problem()
    reset_nargs()
    params = dict(seed=0, classifier=True, popsize=150,
                  early_stopping_rounds=150, time_limit=None,
                  multiple_outputs=True)
    params.update(kwargs)
    gp = EvoDAG(**params)
    record_fit(arrays, meta, 'model', gp, Xtr, ytr, Xte)
    return arrays, meta


def build_endtoend_classifier():
    return end_to_end_classifier()


def build_endtoend_classifier_init():
    "The default parameters of the classifier (input functions included)"
    arrays, meta = {}, {}
    Xtr, Xte, ytr, _ = end_to_end_problem()
    reset_nargs()
    gp = EvoDAG.init(classifier=True, seed=1, popsize=150,
                     early_stopping_rounds=150, time_limit=None)
    record_fit(arrays, meta, 'model', gp, Xtr, ytr, Xte)
    return arrays, meta


def build_endtoend_regression():
    arrays, meta = {}, {}
    Xtr, Xte, _, ytr = end_to_end_problem()
    reset_nargs()
    # seed 1: with seed 0 the best individual is a single variable
    gp = EvoDAG.init(classifier=False, seed=1, popsize=150,
                     early_stopping_rounds=150, time_limit=None)
    record_fit(arrays, meta, 'model', gp, Xtr, ytr, Xte)
    return arrays, meta


def build_endtoend_regression_mo():
    arrays, meta = {}, {}
    Xtr, Xte, _, ytr = end_to_end_problem()
    reset_nargs()
    gp = EvoDAG.init(classifier=False, seed=0, popsize=100,
                     early_stopping_rounds=100, time_limit=None,
                     multiple_outputs=True)
    y = [vec(ytr), vec(-0.5 * ytr + Xtr[:, 3])]
    record_fit(arrays, meta, 'model', gp, Xtr, y, Xte)
    return arrays, meta


def build_endtoend_ensemble():
    arrays, meta = {}, {}
    Xtr, Xte, ytr, _ = end_to_end_problem()
    reset_nargs()
    ens = EvoDAGE(n_estimators=3, n_jobs=1, popsize=100,
                  early_stopping_rounds=100, time_limit=None,
                  classifier=True, multiple_outputs=True)
    ens.fit(Xtr, ytr, test_set=Xte)
    meta['models'] = len(ens._m.models)
    for k, m in enumerate(ens._m.models):
        record_model(arrays, meta, 'model%s' % k, m)
    for name, X in [('train', Xtr), ('test', Xte)]:
        store(arrays, meta, 'decision_function.' + name,
              ens.decision_function(X))
        store(arrays, meta, 'raw_decision_function.' + name,
              ens.raw_decision_function(X))
        store(arrays, meta, 'predict.' + name, ens.predict(X))
        store(arrays, meta, 'predict_proba.' + name, ens.predict_proba(X))
    return arrays, meta


def build_endtoend_ensemble_regression():
    arrays, meta = {}, {}
    Xtr, Xte, _, ytr = end_to_end_problem()
    reset_nargs()
    ens = EvoDAGE(n_estimators=3, n_jobs=1, popsize=100,
                  early_stopping_rounds=100, time_limit=None,
                  classifier=False)
    ens.fit(Xtr, ytr, test_set=Xte)
    meta['models'] = len(ens._m.models)
    for k, m in enumerate(ens._m.models):
        record_model(arrays, meta, 'model%s' % k, m)
    for name, X in [('train', Xtr), ('test', Xte)]:
        store(arrays, meta, 'decision_function.' + name,
              ens.decision_function(X))
        store(arrays, meta, 'predict.' + name, ens.predict(X))
    return arrays, meta


def fitness_case(fitness_function, **kwargs):
    def build():
        return end_to_end_classifier(fitness_function=fitness_function,
                                     **kwargs)
    return build


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

def build_density():
    arrays, meta = {}, {}
    rs = np.random.RandomState(11)
    n = 150
    X = rs.randn(n, 8) * (rs.rand(n, 8) > 0.6)
    y = ((X[:, 0] + X[:, 3] - X[:, 6] + 0.3 * rs.randn(n)) > 0).astype(int) \
        + (X[:, 1] > 0.5)
    for name, classifier, min_density in [('classifier', True, 0.8),
                                          ('regression', False, 0.95)]:
        reset_nargs()
        gp = EvoDAG.init(classifier=classifier, seed=0, popsize=60,
                         early_stopping_rounds=60, time_limit=None,
                         min_density=min_density)
        gp.X = X
        gp.y = y if classifier else y + 0.1 * X[:, 0]
        store(arrays, meta, name + '.inputs',
              np.array([density_of(x.hy) for x in gp.X]))
        gp = EvoDAG.init(classifier=classifier, seed=0, popsize=60,
                         early_stopping_rounds=60, time_limit=None,
                         min_density=min_density)
        gp.fit(X, y if classifier else y + 0.1 * X[:, 0])
        hist = gp.population.hist
        meta[name + '.hist'] = len(hist)
        store(arrays, meta, name + '.population_density',
              gp.population.density)
        store(arrays, meta, name + '.node_density',
              np.array([density_of(h.hy) for h in hist if h.hy is not None]))
    return arrays, meta


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

CSV = """1.5,0,3,a
0,2.5,0,b
-1,0,0,a
0,0,7,c
2,1,1,b
"""

CSV_MULTI = """1,0,0.5,1.5
0,2,-1,0
3,0,0,2
"""

JSON_SPARSE = """{"klass": "x", "num_terms": 6, "0": 1.5, "3": 2}
{"klass": "y", "num_terms": 6, "2": -1}
{"klass": "x", "num_terms": 6, "5": 4, "1": 0.5}
{"klass": "z", "num_terms": 6}
"""

JSON_VEC = """{"klass": 1, "vec": [1, 0, 3, 0]}
{"klass": 0, "vec": [0, 2, 0, 0]}
{"klass": 1, "vec": [0, 0, 0, 5]}
"""

JSON_VEC_SIZE = """{"klass": "a", "vecsize": 5, "vec": [[0, 1.5], [4, 2]]}
{"klass": "b", "vecsize": 5, "vec": [[2, -3]]}
{"klass": "a", "vecsize": 5, "vec": [[1, 7], [3, 0.5]]}
"""


def build_readers():
    from EvoDAG.utils import Inputs
    arrays, meta = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, text):
            fname = os.path.join(tmp, name)
            with open(fname, 'w') as fpt:
                fpt.write(text)
            return fname

        inputs = Inputs()
        X, y = inputs.read_csv(write('a.csv', CSV), 1)
        store(arrays, meta, 'read_csv.X', X)
        store(arrays, meta, 'read_csv.y', y)
        meta['read_csv.word2id'] = inputs.word2id
        meta['read_csv.label2id'] = inputs.label2id
        inputs = Inputs()
        X, y = inputs.read_csv(write('b.csv', CSV_MULTI), 2)
        store(arrays, meta, 'read_csv_multi.X', X)
        store(arrays, meta, 'read_csv_multi.y', [dense(i) for i in y])
        inputs = Inputs()
        X, y = inputs.read_csv(write('c.csv', CSV), 0)
        meta['read_csv_nolabel.X'] = None
        store(arrays, meta, 'read_csv_nolabel.y', y)
        for name, text in [('json_sparse', JSON_SPARSE),
                           ('json_vec', JSON_VEC),
                           ('json_vec_size', JSON_VEC_SIZE)]:
            inputs = Inputs()
            fname = write(name + '.json', text)
            X, y = inputs.read_data_json(fname)
            store(arrays, meta, 'read_data_json.%s.X' % name,
                  [dense(i) for i in X])
            store(arrays, meta, 'read_data_json.%s.y' % name, y)
            meta['read_data_json.%s.label2id' % name] = inputs.label2id
        for name, text in [('json_vec', JSON_VEC),
                           ('json_vec_size', JSON_VEC_SIZE)]:
            inputs = Inputs()
            X, y = inputs.read_data_json_vec(write(name + '.json', text))
            store(arrays, meta, 'read_data_json_vec.%s.X' % name,
                  [dense(i) for i in X])
            store(arrays, meta, 'read_data_json_vec.%s.y' % name, y)
    return arrays, meta


# ---------------------------------------------------------------------------
# Registry, writing and comparing
# ---------------------------------------------------------------------------

GROUPS = {
    'nodes_single': build_nodes_single,
    'nodes_multiple': build_nodes_multiple,
    'kernels': build_kernels,
    'masks': build_masks,
    'endtoend_classifier': build_endtoend_classifier,
    'endtoend_classifier_init': build_endtoend_classifier_init,
    'endtoend_regression': build_endtoend_regression,
    'endtoend_regression_mo': build_endtoend_regression_mo,
    'endtoend_ensemble': build_endtoend_ensemble,
    'endtoend_ensemble_regression': build_endtoend_ensemble_regression,
    'endtoend_fitness_macro_F1': fitness_case('macro-F1'),
    'endtoend_fitness_DotRecallDotPrecision':
        fitness_case('DotRecallDotPrecision'),
    'endtoend_fitness_accDotMacroF1': fitness_case('accDotMacroF1'),
    'endtoend_fitness_macro_RecallF1': fitness_case('macro-RecallF1'),
    'endtoend_orthogonal_selection':
        fitness_case('BER', orthogonal_selection=True),
    'endtoend_orthogonal_dot_selection':
        fitness_case('BER', orthogonal_dot_selection=True),
    'density': build_density,
    'readers': build_readers}


def build(group):
    "Recomputes a group: returns ``(arrays, meta)``"
    arrays, meta = GROUPS[group]()
    reset_nargs()
    return arrays, meta


def paths(group, directory=GOLDEN_DIR):
    directory = Path(directory)
    return directory / (group + '.npz'), directory / (group + '.json')


def write(group, directory=GOLDEN_DIR):
    arrays, meta = build(group)
    npz, js = paths(group, directory)
    np.savez_compressed(npz, **arrays)
    with open(js, 'w') as fpt:
        json.dump(meta, fpt, sort_keys=True, separators=(',', ':'))
        fpt.write('\n')
    return npz, js


def load(group, directory=GOLDEN_DIR):
    npz, js = paths(group, directory)
    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    with open(js) as fpt:
        meta = json.load(fpt)
    return arrays, meta


def roundtrip_meta(meta):
    "``meta`` as it comes back from json (tuples are lists, keys are str)"
    return json.loads(json.dumps(meta, sort_keys=True))


def compare(group, directory=GOLDEN_DIR, atol=1e-6):
    """Compares the recomputed group with the stored one. Returns the list
    of differences (empty when they agree): structure must be identical,
    numbers must agree within ``atol`` and non finite values must be in the
    same places."""
    arrays, meta = build(group)
    ref_arrays, ref_meta = load(group, directory)
    meta = roundtrip_meta(meta)
    problems = []
    for key in sorted(set(meta) | set(ref_meta)):
        if meta.get(key, '<missing>') != ref_meta.get(key, '<missing>'):
            problems.append('meta %s: %r != %r' % (
                key, meta.get(key, '<missing>'),
                ref_meta.get(key, '<missing>')))
    for key in sorted(set(arrays) | set(ref_arrays)):
        if key not in arrays or key not in ref_arrays:
            problems.append('array %s only in %s' % (
                key, 'recomputed' if key in arrays else 'golden'))
            continue
        a, b = arrays[key], ref_arrays[key]
        if a.shape != b.shape:
            problems.append('array %s: shape %s != %s' % (key, a.shape,
                                                          b.shape))
        elif a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            if not np.array_equal(a, b):
                problems.append('array %s differs' % key)
        else:
            a = a.astype(float)
            b = b.astype(float)
            if not (np.array_equal(np.isnan(a), np.isnan(b)) and
                    np.array_equal(np.isposinf(a), np.isposinf(b)) and
                    np.array_equal(np.isneginf(a), np.isneginf(b))):
                problems.append('array %s: non finite values differ' % key)
                continue
            fin = np.isfinite(a)
            if not np.allclose(a[fin], b[fin], rtol=0, atol=atol):
                problems.append('array %s: max |diff| %g' % (
                    key, np.max(np.abs(a[fin] - b[fin]))))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', default=str(GOLDEN_DIR),
                        help='directory where the golden files are written')
    parser.add_argument('groups', nargs='*', help='groups (default: all)')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for group in args.groups or GROUPS:
        npz, js = write(group, args.output)
        print(npz, os.path.getsize(npz))


if __name__ == '__main__':
    main()
