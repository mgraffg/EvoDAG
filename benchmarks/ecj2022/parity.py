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
"""Bit-level parity check of the engine: the check for any optimisation.

    python benchmarks/ecj2022/parity.py record --output /tmp/parity
    python benchmarks/ecj2022/parity.py check --reference /tmp/parity

``record`` runs a fixed list of small fits on synthetic data and stores
everything observable of each one, exactly (arrays are saved with NumPy,
never rounded): the decision function of the model on the training inputs
and on a held-out part, the predictions, the size and height of the model,
its validation fitness, the number of individuals created, a digest of every
individual of the history (structure, weights and fitness) and the size in
bytes of the pickled model. ``check`` runs the same fits again and compares
every stored value with ``np.array_equal`` (NaN in the same place is equal);
it exits with status 1 and lists the cases that differ.

The cases are the 22 configurations of ``configs.py`` with the parameters of
the experiments (population and early stopping reduced to 200) on a 3-class
and a 2-class problem with seeds 0 and 1, the released defaults for
classification (with and without multiple outputs) and regression, two
``EvoDAGE`` ensembles, cases with a column of huge values and a constant
column, and three cases of paths the others do not take (the agreement
heuristic in regression, the generational population and a test set given to
``fit``, whose outputs on it are recorded too); 106 cases in all. Record the reference from the tree **before** an optimisation and
check after each change; the golden files of the tests do not cover the
selection schemes nor these configurations.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402

N_TRAIN = 300
N_TEST = 100
N_VARIABLES = 8
POPSIZE = 200
SMALL = dict(popsize=POPSIZE, early_stopping_rounds=POPSIZE, time_limit=None)
SEEDS = (0, 1)


# -- the data --------------------------------------------------------------

def classification_data(n_classes, n_variables=N_VARIABLES, seed=0):
    "``X, y, Xtest`` of a noisy linear problem with ``n_classes`` classes"
    rs = np.random.RandomState(seed + 100 * n_classes)
    n = N_TRAIN + N_TEST
    X = rs.randn(n, n_variables)
    W = rs.randn(n_variables, n_classes)
    y = (X.dot(W) + 0.5 * rs.randn(n, n_classes)).argmax(axis=1)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]


def regression_data(seed=0):
    "``X, y, Xtest`` of a nonlinear regression problem"
    rs = np.random.RandomState(seed + 7)
    n = N_TRAIN + N_TEST
    X = rs.randn(n, N_VARIABLES)
    w = rs.randn(N_VARIABLES)
    y = X.dot(w) + X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.1 * rs.randn(n)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:]


def with_extremes(X, Xtest):
    "A column of huge values, one of tiny values and a constant column"
    X, Xtest = X.copy(), Xtest.copy()
    for A in (X, Xtest):
        A[:, 3] *= 1e150
        A[:, 4] *= 1e-150
        A[:, 5] = 1.0
    return X, Xtest


# -- the cases -------------------------------------------------------------

def make_cases():
    """The list of cases: dictionaries with ``name``, ``kind`` (``engine`` or
    ``ensemble``), the parameters and the data."""
    cases = []
    problems = {"3c": classification_data(3), "2c": classification_data(2)}
    for config in configs.CONFIGS:
        for label, (X, y, Xt) in problems.items():
            for seed in SEEDS:
                params = {**configs.params_for(config), **SMALL}
                cases.append(dict(name="%s-%s-s%d" % (config, label, seed),
                                  kind="engine", classifier=True, seed=seed,
                                  params=params, data=(X, y, Xt)))
    for seed in SEEDS:
        for label, multiple in (("3c", True), ("2c", True), ("2c", False)):
            X, y, Xt = problems[label]
            cases.append(dict(
                name="default-%s-multiple%s-s%d" % (label, multiple, seed),
                kind="engine", classifier=True, seed=seed,
                params=dict(SMALL, multiple_outputs=multiple),
                data=(X, y, Xt)))
        cases.append(dict(name="default-regression-s%d" % seed,
                          kind="engine", classifier=False, seed=seed,
                          params=dict(SMALL), data=regression_data()))
    X, y, Xt = problems["3c"]
    # paths the others do not take: the regression branch of the agreement
    # heuristic, the generational population and a test set given to fit
    Xr, yr, Xtr = regression_data()
    cases.append(dict(name="regression-orthogonal", kind="engine",
                      classifier=False, seed=0,
                      params=dict(SMALL, orthogonal_selection=True),
                      data=(Xr, yr, Xtr)))
    cases.append(dict(name="generational-3c", kind="engine", classifier=True,
                      seed=0, params=dict(SMALL,
                                          population_class="Generational"),
                      data=(X, y, Xt)))
    cases.append(dict(name="test-set-3c", kind="engine", classifier=True,
                      seed=0, params=dict(SMALL), data=(X, y, Xt),
                      test_set=True))
    cases.append(dict(name="ensemble-classification", kind="ensemble",
                      classifier=True, seed=0,
                      params=dict(SMALL, n_estimators=3), data=(X, y, Xt)))
    cases.append(dict(name="ensemble-regression", kind="ensemble",
                      classifier=False, seed=0,
                      params=dict(SMALL, n_estimators=3),
                      data=regression_data()))
    for name, params in (("default", dict(SMALL)),
                         ("agr-rnd", {**configs.params_for("agr-rnd"),
                                      **SMALL}),
                         ("ads-rnd", {**configs.params_for("ads-rnd"),
                                      **SMALL}),
                         ("prs-fit-all", {**configs.params_for("prs-fit-all"),
                                          **SMALL})):
        X, y, Xt = problems["3c"]
        Xe, Xte = with_extremes(X, Xt)
        cases.append(dict(name="extremes-%s" % name, kind="engine",
                          classifier=True, seed=0, params=params,
                          data=(Xe, y, Xte)))
    Xr, yr, Xtr = regression_data()
    Xe, Xte = with_extremes(Xr, Xtr)
    cases.append(dict(name="extremes-regression", kind="engine",
                      classifier=False, seed=0, params=dict(SMALL),
                      data=(Xe, yr, Xte)))
    return cases


def stack(df):
    "The decision function as one 2-d array (one row per output)"
    if isinstance(df, (list, tuple)):
        return np.array([np.asarray(x, dtype=np.float64) for x in df])
    return np.asarray(df, dtype=np.float64)


def observe_model(model, X, Xt):
    "Everything observable of a fitted ``Model``"
    out = {"df_train": stack(model.decision_function(X)),
           "df_test": stack(model.decision_function(Xt)),
           "pred_train": np.asarray(model.predict(X)),
           "pred_test": np.asarray(model.predict(Xt)),
           "size": np.array(model.size), "height": np.array(model.height),
           "fitness_vs": np.array(model.fitness_vs, dtype=np.float64),
           "pickle_size": np.array(len(pickle.dumps(model)))}
    return out


def history_digest(engine):
    """SHA-256 (as a ``uint8`` array) of every individual created: its class,
    the positions of its arguments, its weights and its fitness. It checks
    the whole evolution, not only the individual that is kept."""
    digest = hashlib.sha256()
    for node in engine.population.hist:
        digest.update(pickle.dumps((
            type(node).__name__, node.variable, node.weight, node.fitness,
            node.fitness_vs)))
    return np.frombuffer(digest.digest(), dtype=np.uint8)


def run_case(case):
    "Run one case; returns a dictionary of NumPy values"
    from EvoDAG.base import EvoDAG as Engine
    from EvoDAG.model import EvoDAGE
    X, y, Xt = case["data"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            if case["kind"] == "ensemble":
                params = dict(case["params"])
                ensemble = EvoDAGE(seed=case["seed"],
                                   classifier=case["classifier"],
                                   **params).fit(X, y)
                out = {"df_train": stack(ensemble.decision_function(X)),
                       "df_test": stack(ensemble.decision_function(Xt)),
                       "pred_train": np.asarray(ensemble.predict(X)),
                       "pred_test": np.asarray(ensemble.predict(Xt)),
                       "sizes": np.array([m.size for m in ensemble.models]),
                       "heights": np.array([m.height
                                            for m in ensemble.models]),
                       "pickle_size": np.array(
                           [len(pickle.dumps(m)) for m in ensemble.models])}
                return out
            engine = Engine.init(classifier=case["classifier"],
                                 seed=case["seed"], **case["params"])
            engine.fit(X, y, test_set=Xt if case.get("test_set") else None)
            out = observe_model(engine.model(), X, Xt)
            if case.get("test_set"):
                # what the engine evaluated on the test set along the way
                out["df_fit_test"] = stack(engine.decision_function())
                out["hy_test"] = stack(engine.population.estopping.hy_test)
            out["n_individuals"] = np.array(len(engine.population.hist))
            out["history"] = history_digest(engine)
    return out


# -- record and check ------------------------------------------------------

def record(output, log=print):
    "Run every case and store its values in ``output``; returns the names"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = []
    for case in make_cases():
        t0 = time.perf_counter()
        values = run_case(case)
        np.savez(output / (case["name"] + ".npz"), **values)
        names.append(case["name"])
        log("%-36s %5.1f s" % (case["name"], time.perf_counter() - t0))
    with open(output / "index.json", "w") as fpt:
        json.dump(names, fpt, indent=1)
    return names


def same(a, b):
    "Bit-exact equality of two arrays (NaN in the same place is equal)"
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def check(reference, log=print):
    """Run the cases again and compare with ``reference``. Returns the list
    of ``(case, what differs)``; empty when everything is identical."""
    reference = Path(reference)
    with open(reference / "index.json") as fpt:
        recorded = json.load(fpt)
    cases = {case["name"]: case for case in make_cases()}
    differences = []
    for name in recorded:
        if name not in cases:
            differences.append((name, "the case no longer exists"))
            continue
        values = run_case(cases[name])
        with np.load(reference / (name + ".npz")) as stored:
            keys = set(stored.files)
            bad = sorted(k for k in keys & set(values)
                         if not same(stored[k], values[k]))
            bad += sorted(keys ^ set(values))
        if bad:
            differences.append((name, ", ".join(bad)))
            log("DIFFERENT %s: %s" % (name, ", ".join(bad)))
    extra = sorted(set(cases) - set(recorded))
    differences += [(name, "not in the reference") for name in extra]
    return differences


def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Bit-level parity check of the engine.")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="store the reference outputs")
    rec.add_argument("--output", required=True, help="directory")
    chk = sub.add_parser("check", help="compare with a reference")
    chk.add_argument("--reference", required=True, help="directory")
    args = parser.parse_args(argv)
    if args.command == "record":
        names = record(args.output)
        print("recorded %d cases in %s" % (len(names), args.output))
        return 0
    differences = check(args.reference)
    if differences:
        print("%d cases differ:" % len(differences))
        for name, what in differences:
            print("  %s: %s" % (name, what))
        return 1
    print("identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
