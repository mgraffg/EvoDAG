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
"""Reproduce the first table of the Performance page with this release.

    python benchmarks/raetsch.py --n-jobs 8
    python benchmarks/raetsch.py --summary

EvoDAG, as ``EnsembleClassifier(EvoDAGClassifier())`` with every parameter at
its default, and sixteen classifiers of scikit-learn with their default
parameters are trained on the splits of Gunnar Raetsch's benchmark problems
and evaluated on the test part with the balanced error rate (BER, in
percent). One line per ``(dataset, split, algorithm)`` is appended to a JSONL
file (``--output``, ``benchmarks/raetsch-results.jsonl`` by default, which is
not committed), and a triple that is already in the file is not computed
again, so an interrupted run resumes where it stopped.
``--summary`` turns the file into the Markdown table of the page.

The data (``benchmarks.mat``, 13 problems with 100 splits each, 20 for image
and splice) is downloaded once into ``~/.cache/evodag`` from a pinned commit
of https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets and its
sha256 is verified; ``--data PATH`` uses a local file instead. Nothing is ever
written inside the repository except the results file.

**The full run is very long**: one default fit of the 30-member ensemble on
one split of heart took more than 40 minutes on 12 cores, and the whole table
is hundreds of CPU hours. ``--quick`` replaces the EvoDAG parameters by tiny
ones (``popsize=50, early_stopping_rounds=50, n_estimators=3``) to test the
pipeline; its lines are marked ``"quick": true`` and are left out of the
summary unless ``--include-quick`` is given. They are not results.

Needs scipy (to read the ``.mat``) and scikit-learn; neither is a dependency
of the package.
"""
import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = ("https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets/raw/"
       "66aa4e766b60bf031dfd27b9cbe97307cdf7242e/benchmarks.mat")
SHA256 = "47c19e4bc4716edc4077cfa5ea61edf4d02af4ec51a0ecfe035626ae8b561c75"
CACHE = Path.home() / ".cache" / "evodag" / "benchmarks.mat"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "raetsch-results.jsonl"

# the nine problems of the table, spelled as in the .mat file
DATASETS = ["banana", "thyroid", "diabetis", "heart", "ringnorm", "twonorm",
            "german", "image", "waveform"]

# (name, module, class) of the sixteen classifiers of the page
SKLEARN = [
    ("LogisticRegression", "sklearn.linear_model", "LogisticRegression"),
    ("SGD", "sklearn.linear_model", "SGDClassifier"),
    ("Perceptron", "sklearn.linear_model", "Perceptron"),
    ("PassiveAggressive", "sklearn.linear_model", "PassiveAggressiveClassifier"),
    ("SVC", "sklearn.svm", "SVC"),
    ("LinearSVC", "sklearn.svm", "LinearSVC"),
    ("KNeighbors", "sklearn.neighbors", "KNeighborsClassifier"),
    ("NearestCentroid", "sklearn.neighbors", "NearestCentroid"),
    ("GaussianNB", "sklearn.naive_bayes", "GaussianNB"),
    ("BernoulliNB", "sklearn.naive_bayes", "BernoulliNB"),
    ("DecisionTree", "sklearn.tree", "DecisionTreeClassifier"),
    ("RandomForest", "sklearn.ensemble", "RandomForestClassifier"),
    ("ExtraTrees", "sklearn.ensemble", "ExtraTreesClassifier"),
    ("AdaBoost", "sklearn.ensemble", "AdaBoostClassifier"),
    ("GradientBoosting", "sklearn.ensemble", "GradientBoostingClassifier"),
    ("MLP", "sklearn.neural_network", "MLPClassifier"),
]
ALGORITHMS = ["evodag"] + [name for name, _, _ in SKLEARN]

# The table of the page as published (EvoDAG, old version, default parameters)
PUBLISHED = {"rank": 2.4,
             "ber": {"banana": 12.0, "thyroid": 7.7, "diabetis": 25.0,
                     "heart": 16.1, "ringnorm": 1.5, "twonorm": 2.3,
                     "german": 28.3, "image": 3.6, "waveform": 10.8}}


# -- data ------------------------------------------------------------------

def sha256_of(path):
    "Hexadecimal sha256 of a file"
    digest = hashlib.sha256()
    with open(path, "rb") as fpt:
        for chunk in iter(lambda: fpt.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected=SHA256):
    "Raise ``ValueError`` if the file is not the one with the expected hash"
    found = sha256_of(path)
    if found != expected:
        raise ValueError("%s has sha256 %s but %s was expected; refusing to "
                         "use it (delete it to download it again)"
                         % (path, found, expected))


def ensure_data(path=CACHE, url=URL, sha256=SHA256):
    """Path of the verified ``benchmarks.mat``, downloaded the first time."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        print("downloading %s" % url, file=sys.stderr)
        with urllib.request.urlopen(url) as response, open(part, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(part, path)
    verify_sha256(path, sha256)
    return path


def load_dataset(path, name):
    """One problem of the ``.mat``: ``X``, ``y`` (labels -1 and 1) and the
    ``train`` and ``test`` indices, one row per split, converted from the
    1-based indices of the file to 0-based ones."""
    from scipy.io import loadmat
    record = loadmat(str(path))[name][0, 0]
    X = np.asarray(record["x"], dtype=float)
    y = np.asarray(record["t"]).ravel().astype(int)
    train = np.asarray(record["train"]).astype(int) - 1
    test = np.asarray(record["test"]).astype(int) - 1
    return X, y, train, test


def parse_splits(text, available):
    """Split numbers of ``text`` (``all``, ``0-4``, ``0,3,5-7``), checked
    against the ``available`` number of splits."""
    if text == "all":
        return list(range(available))
    chosen = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            chosen.extend(range(int(low), int(high) + 1))
        else:
            chosen.append(int(part))
    bad = [s for s in chosen if not 0 <= s < available]
    if bad:
        raise ValueError("splits %s are out of range: there are %d splits"
                         % (bad, available))
    return chosen


# -- measure and algorithms ------------------------------------------------

def ber(y, yh):
    """Balanced error rate in percent: the mean over the classes of ``y`` of
    the percentage of their examples that are misclassified."""
    y = np.asarray(y)
    yh = np.asarray(yh)
    errors = [np.mean(yh[y == c] != c) for c in np.unique(y)]
    return float(np.mean(errors) * 100.)


def make_algorithm(name, n_jobs=1, quick=False):
    """A new, unfitted estimator, or ``None`` if the installed scikit-learn
    does not have it (the reason is then in ``missing_reason``)."""
    if name == "evodag":
        from EvoDAG import EnsembleClassifier, EvoDAGClassifier
        if quick:
            return EnsembleClassifier(
                EvoDAGClassifier(popsize=50, early_stopping_rounds=50),
                n_estimators=3, random_state=0, n_jobs=n_jobs)
        return EnsembleClassifier(EvoDAGClassifier(), n_estimators=30,
                                  random_state=0, n_jobs=n_jobs)
    for label, module, cls in SKLEARN:
        if label == name:
            try:
                klass = getattr(importlib.import_module(module), cls)
            except (ImportError, AttributeError):
                return None
            # defaults, except a fixed seed so that a run can be repeated
            estimator = klass()
            if "random_state" in estimator.get_params():
                estimator.set_params(random_state=0)
            return estimator
    raise ValueError("unknown algorithm %r" % name)


def environment():
    "Versions and machine, stored in every line of the results"
    import sklearn
    import EvoDAG
    return {"evodag": EvoDAG.__version__, "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.platform(), "cpus": os.cpu_count()}


# -- results ---------------------------------------------------------------

def read_results(path):
    "The lines (dictionaries) of a JSONL file; a missing file has none"
    path = Path(path)
    if not path.exists():
        return []
    lines = []
    with open(path, encoding="utf8") as fpt:
        for line in fpt:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except ValueError:
                # the last line of an interrupted run may be cut
                continue
    return lines


def key_of(line):
    return (line["dataset"], int(line["split"]), line["algorithm"],
            bool(line.get("quick", False)))


def run(data, output, datasets=None, splits="all", algorithms=None,
        n_jobs=1, quick=False, log=print):
    """Compute the missing ``(dataset, split, algorithm)`` triples and append
    them to ``output``; returns the number of lines written."""
    datasets = list(datasets or DATASETS)
    algorithms = list(algorithms or ALGORITHMS)
    unknown = sorted(set(algorithms) - set(ALGORITHMS))
    if unknown:
        raise ValueError("unknown algorithms %s; choose among %s"
                         % (unknown, ALGORITHMS))
    done = {key_of(x) for x in read_results(output) if "split" in x}
    known_missing = {x["algorithm"] for x in read_results(output)
                     if x.get("missing")}
    env = environment()
    written = 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "a", encoding="utf8") as fpt:
        def write(record):
            fpt.write(json.dumps(record) + "\n")
            fpt.flush()

        usable = []
        for name in algorithms:
            if make_algorithm(name, n_jobs, quick) is None:
                log("note: %s does not exist in scikit-learn %s; skipped"
                    % (name, env["scikit-learn"]))
                if name not in known_missing:
                    write({"algorithm": name, "missing": True,
                           "reason": "not in scikit-learn %s"
                           % env["scikit-learn"]})
                    written += 1
            else:
                usable.append(name)
        for dataset in datasets:
            X, y, train, test = load_dataset(data, dataset)
            for split in parse_splits(splits, train.shape[0]):
                tr, ts = train[split], test[split]
                for name in usable:
                    if (dataset, split, name, quick) in done:
                        continue
                    estimator = make_algorithm(name, n_jobs, quick)
                    t0 = time.time()
                    estimator.fit(X[tr], y[tr])
                    t1 = time.time()
                    hy = estimator.predict(X[ts])
                    t2 = time.time()
                    record = {"dataset": dataset, "split": split,
                              "algorithm": name, "ber": ber(y[ts], hy),
                              "fit_seconds": t1 - t0,
                              "predict_seconds": t2 - t1, "quick": quick,
                              "environment": env}
                    write(record)
                    written += 1
                    log("%s split %d %s: BER %.2f (fit %.1fs)"
                        % (dataset, split, name, record["ber"],
                           record["fit_seconds"]))
    return written


# -- summary ---------------------------------------------------------------

def average_ranks(means):
    """Average rank of every algorithm. ``means`` maps dataset to a
    dictionary algorithm -> mean BER. Ranks (1 is best, ties share the mean
    of their positions) are computed per dataset and averaged."""
    ranks = {}
    for scores in means.values():
        ordered = sorted(scores.values())
        for name, value in scores.items():
            below = sum(1 for v in ordered if v < value)
            ties = sum(1 for v in ordered if v == value)
            ranks.setdefault(name, []).append(below + (ties + 1) / 2.)
    return {name: float(np.mean(r)) for name, r in ranks.items()}


def summary(path, include_quick=False):
    "The Markdown table of the page, built from the results file."
    lines = read_results(path)
    missing = sorted({x["algorithm"] for x in lines if x.get("missing")})
    lines = [x for x in lines if "split" in x
             and (include_quick or not x.get("quick", False))]
    if not lines:
        return "No results in %s." % path
    bers = {}
    for x in lines:
        bers.setdefault(x["dataset"], {}).setdefault(
            x["algorithm"], {})[int(x["split"])] = x["ber"]
    datasets = [d for d in DATASETS if d in bers]
    datasets += sorted(d for d in bers if d not in DATASETS)
    means = {d: {a: float(np.mean(list(s.values()))) for a, s in algs.items()}
             for d, algs in bers.items()}
    counts = {d: min(len(s) for s in algs.values())
              for d, algs in bers.items()}
    ranks = average_ranks(means)
    best = {d: min(means[d].values()) for d in datasets}

    def cell(dataset, algorithm):
        if algorithm not in means[dataset]:
            return ""
        text = "%.1f" % means[dataset][algorithm]
        if means[dataset][algorithm] == best[dataset]:
            text = "**%s**" % text
        return text

    out = ["Mean BER (%) over the splits of each problem, "
           "and average rank (1 is best)."]
    if include_quick:
        out.append("Includes lines of `--quick` runs, which are not results.")
    out.append("")
    out.append("Splits per problem: " + ", ".join(
        "%s %d" % (d, counts[d]) for d in datasets) + ".")
    if missing:
        out.append("Not available in this scikit-learn: %s."
                   % ", ".join(missing))
    out.append("")
    out.append("|Classifier|Average rank|" + "|".join(datasets) + "|")
    out.append("|-------|------:|" + "|".join("---:" for _ in datasets) + "|")
    published = ["%.1f" % PUBLISHED["ber"][d] if d in PUBLISHED["ber"] else ""
                 for d in datasets]
    out.append("|EvoDAG (published)|%.1f|%s|" % (PUBLISHED["rank"],
                                                 "|".join(published)))
    for name in sorted(ranks, key=lambda a: (ranks[a], a)):
        label = "EvoDAG" if name == "evodag" else name
        out.append("|%s|%.1f|%s|" % (label, ranks[name], "|".join(
            cell(d, name) for d in datasets)))
    return "\n".join(out)


# -- command line ----------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the first table of the Performance page.")
    parser.add_argument("--data", help="local benchmarks.mat (default: "
                        "download into ~/.cache/evodag and verify the hash)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="problems (default: the nine of the table)")
    parser.add_argument("--splits", default="all",
                        help="all, 0-4, 0,3,5-7 ... (default: all)")
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS,
                        help="default: evodag and the sixteen classifiers "
                        "of scikit-learn")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="processes for the members of the EvoDAG "
                        "ensemble; it only changes the speed")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="JSONL file of results (appended, resumable)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny EvoDAG parameters, to test the pipeline")
    parser.add_argument("--summary", action="store_true",
                        help="print the Markdown table of the results file")
    parser.add_argument("--include-quick", action="store_true",
                        help="with --summary, also use the --quick lines")
    args = parser.parse_args(argv)
    if args.summary:
        print(summary(args.output, include_quick=args.include_quick))
        return 0
    data = Path(args.data) if args.data else ensure_data()
    written = run(data, args.output, datasets=args.datasets,
                  splits=args.splits, algorithms=args.algorithms,
                  n_jobs=args.n_jobs, quick=args.quick)
    print("%d lines written to %s" % (written, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
