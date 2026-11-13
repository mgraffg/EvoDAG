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
"""Speed benchmark of EvoDAG (a script, not a pytest test).

    python benchmarks/bench.py --output benchmarks/results.json

Fits EvoDAG on fixed synthetic data and times ``fit`` and ``predict``:

* ``classification_1000x20_3c`` and ``classification_10000x50_5c``: seeds 0,
  1 and 2, multiple outputs, ``popsize=1000``;
* ``regression_1000x20``: seeds 0, 1 and 2;
* ``ensemble_1000x20_4``: one ``EvoDAGE(n_estimators=4, n_jobs=1)`` fit.

Everything runs in a single process. The JSON written has, per case, the
list of times (one per seed), their minimum, and the training accuracy
(classification) or relative squared error (regression) per seed; plus the
Python, numpy and EvoDAG versions and the machine. Run it with nothing else
heavy running: the times are only comparable on the same machine.

It does not import SparseArray, so it works with any EvoDAG implementation.
"""
import argparse
import json
import os
import platform
import time
import warnings

import numpy as np

import EvoDAG as evodag_package
from EvoDAG import EvoDAG
from EvoDAG.model import EvoDAGE

SEEDS = (0, 1, 2)
CLASSIFIER = dict(classifier=True, popsize=1000, early_stopping_rounds=1000,
                  time_limit=None, multiple_outputs=True)


def classification_data(n, nvar, nclasses):
    "Gaussian data, labels = argmax of a noisy linear map"
    rs = np.random.RandomState(0)
    X = rs.randn(n, nvar)
    W = rs.randn(nvar, nclasses)
    y = (X.dot(W) + 0.5 * rs.randn(n, nclasses)).argmax(axis=1)
    return X, y


def regression_data(n, nvar):
    rs = np.random.RandomState(0)
    X = rs.randn(n, nvar)
    w = rs.randn(nvar)
    y = X.dot(w) + X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.1 * rs.randn(n)
    return X, y


def relative_squared_error(y, hy):
    return float(((y - hy) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def timed(func):
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


def summary(fit_times, predict_times, scores, score_name):
    return {'fit_times': fit_times, 'fit_min': min(fit_times),
            'predict_times': predict_times, 'predict_min': min(predict_times),
            score_name: scores}


def bench_classification(X, y):
    fit_times, predict_times, scores = [], [], []
    for seed in SEEDS:
        gp = EvoDAG(seed=seed, **CLASSIFIER)
        _, t = timed(lambda: gp.fit(X, y))
        fit_times.append(t)
        model = gp.model()
        hy, t = timed(lambda: model.predict(X))
        predict_times.append(t)
        scores.append(float((hy == y).mean()))
    return summary(fit_times, predict_times, scores, 'train_accuracy')


def bench_regression(X, y):
    fit_times, predict_times, scores = [], [], []
    for seed in SEEDS:
        # EvoDAG(classifier=False) with the default arguments does not work:
        # the regression configuration comes from EvoDAG.init
        gp = EvoDAG.init(classifier=False, seed=seed, popsize=1000,
                         early_stopping_rounds=1000, time_limit=None)
        _, t = timed(lambda: gp.fit(X, y))
        fit_times.append(t)
        model = gp.model()
        hy, t = timed(lambda: model.predict(X))
        predict_times.append(t)
        scores.append(relative_squared_error(y, hy))
    return summary(fit_times, predict_times, scores, 'train_rse')


def bench_ensemble(X, y):
    ens = EvoDAGE(n_estimators=4, n_jobs=1, **CLASSIFIER)
    _, fit_time = timed(lambda: ens.fit(X, y))
    hy, predict_time = timed(lambda: ens.predict(X))
    return summary([fit_time], [predict_time], [float((hy == y).mean())],
                   'train_accuracy')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', required=True,
                        help='JSON file where the results are written')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    X, y = classification_data(1000, 20, 3)
    XL, yL = classification_data(10000, 50, 5)
    XR, yR = regression_data(1000, 20)
    cases = {}
    cases['classification_1000x20_3c'] = bench_classification(X, y)
    cases['classification_10000x50_5c'] = bench_classification(XL, yL)
    cases['regression_1000x20'] = bench_regression(XR, yR)
    cases['ensemble_1000x20_4'] = bench_ensemble(X, y)
    result = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'EvoDAG': evodag_package.__version__,
        'processor': platform.processor(),
        'machine': platform.machine(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'seeds': list(SEEDS),
        'cases': cases}
    with open(args.output, 'w') as fpt:
        json.dump(result, fpt, indent=2)
        fpt.write('\n')
    for name, case in cases.items():
        print('%-30s fit_min %7.2f s  predict_min %6.2f s' % (
            name, case['fit_min'], case['predict_min']))
    print(os.path.abspath(args.output))


if __name__ == '__main__':
    main()
