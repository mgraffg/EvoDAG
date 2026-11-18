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
"""Run one unit of the selection heuristics experiments.

    python benchmarks/ecj2022/run.py --dataset iris --config agr-rnd
    python benchmarks/ecj2022/run.py --dataset iris --config agr-rnd --workers 4 --quick
    python benchmarks/ecj2022/run.py --dataset iris --sklearn

A unit is a dataset with a configuration of ``configs.py`` (an ensemble of 30
EvoDAG models, one per seed ``0 .. 29``, each trained on its own random half
of the training set) or a dataset with the sixteen scikit-learn classifiers.
The prepared data are those of ``datasets.py``. Results go to
``<root>/results[-quick]/<dataset>/<config>/``: ``member-<i>.model.gz`` and
``member-<i>.json`` for each member (the JSON is written last, and its
presence means the member is done, so an interrupted unit resumes at the
member that was missing) and, when every member is done, ``result.json``
(it counts as the finished unit only if it has the number of members that
was asked for) and ``predictions.npz`` (the labels of the test set, the
predictions of the majority vote and of the median, and the median decision
function). The files of the scikit-learn unit are
``<dataset>/sklearn/<Name>.json`` and ``<Name>.npz`` (``y_test`` and ``hy``);
``--classifiers`` or ``--classifier-index`` select some of the classifiers.
Each member is a process of its own (``fork`` when the platform has it); a
member killed by a signal is recorded in ``member-<i>.error.txt``.

Every fit uses one core (the BLAS and OpenMP thread variables are set to 1
before NumPy is imported) and no test set is given to ``fit``: the
predictions are computed afterwards from the stored models.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import io  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402

ENV_ROOT = "EVODAG_EXPERIMENTS"
SKLEARN = configs.SKLEARN
DEFAULT_SKLEARN_TIMEOUT_HOURS = 24.0


def resolve_root(root=None, environ=None):
    """The experiments directory: ``root``, else ``EVODAG_EXPERIMENTS``, else
    ``experiments`` at the root of the repository."""
    environ = os.environ if environ is None else environ
    if root:
        return Path(root)
    if environ.get(ENV_ROOT):
        return Path(environ[ENV_ROOT])
    return Path(__file__).resolve().parents[2] / "experiments"


def results_dir(root, quick=False):
    "Directory of the results (``results-quick`` in quick mode)"
    return Path(root) / ("results-quick" if quick else "results")


def unit_dir(root, dataset, name, quick=False):
    "Directory of the files of one unit"
    return results_dir(root, quick) / dataset / name


# -- data and measures -----------------------------------------------------

def load_data(root, dataset):
    "``X_train, y_train, X_test, y_test`` of a prepared dataset"
    path = Path(root) / "data" / (dataset + ".npz")
    if not path.exists():
        raise FileNotFoundError(
            "%s does not exist: prepare the data with "
            "benchmarks/ecj2022/datasets.py" % path)
    with np.load(path) as data:
        return (data["X_train"], data["y_train"], data["X_test"],
                data["y_test"])


def macro_f1(y, hy):
    """Macro-F1: the unweighted mean, over the classes that are in ``y`` or in
    ``hy``, of ``2 tp / (2 tp + fp + fn)`` (0 when the denominator is 0)."""
    y = np.asarray(y)
    hy = np.asarray(hy)
    scores = []
    for c in np.union1d(y, hy):
        tp = float(np.sum((y == c) & (hy == c)))
        fp = float(np.sum((y != c) & (hy == c)))
        fn = float(np.sum((y == c) & (hy != c)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def recall_macro(y, hy):
    """Macro-recall: the mean, over the classes that are in ``y``, of the
    proportion of the examples of the class that ``hy`` predicts as that
    class. The balanced error rate (BER) is ``1 - recall_macro``."""
    y = np.asarray(y)
    hy = np.asarray(hy)
    recalls = [float(np.sum((y == c) & (hy == c))) / float(np.sum(y == c))
               for c in np.unique(y)]
    return float(np.mean(recalls))


def accuracy(y, hy):
    "Proportion of correct predictions"
    return float(np.mean(np.asarray(y) == np.asarray(hy)))


def git_dirty():
    """Whether ``git status --porcelain`` prints anything (the tree has
    changes not committed); ``None`` if git fails."""
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=60)
        if done.returncode != 0:
            return None
        return bool(done.stdout.decode().strip())
    except (OSError, subprocess.SubprocessError):
        return None


def git_commit():
    "``git rev-parse HEAD`` of the repository, ``unknown`` if it fails"
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
        text = done.stdout.decode().strip()
        return text if done.returncode == 0 and text else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment(sklearn=False):
    "Versions and machine, stored in every result"
    import EvoDAG
    env = {"evodag": EvoDAG.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "host": socket.gethostname(),
           "git_commit": git_commit(), "git_dirty": git_dirty()}
    if sklearn:
        import sklearn as sk
        env["scikit-learn"] = sk.__version__
    return env


def write_atomic(path, data):
    "Write bytes to a temporary name in the same directory and rename it"
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fpt:
        fpt.write(data)
    os.replace(tmp, path)


def write_json(path, content):
    "Write a JSON file atomically"
    write_atomic(path, (json.dumps(content, indent=2) + "\n").encode("utf8"))


def write_npz(path, **arrays):
    "Write a compressed ``.npz`` atomically"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_atomic(path, buffer.getvalue())


def mp_context():
    "``fork`` when available (the children share the loaded modules)"
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else None)


# -- the EvoDAG unit -------------------------------------------------------

def member_paths(directory, i):
    "Files of member ``i``: the model, its JSON and its error file"
    directory = Path(directory)
    return (directory / ("member-%d.model.gz" % i),
            directory / ("member-%d.json" % i),
            directory / ("member-%d.error.txt" % i))


def fit_member(task):
    """Fit one member and write its files; the argument is a tuple
    ``(root, dataset, config, quick, i)``. Returns ``(i, error or None)``."""
    root, dataset, config, quick, i = task
    directory = unit_dir(root, dataset, config, quick)
    model_path, json_path, error_path = member_paths(directory, i)
    try:
        from EvoDAG.model import EvoDAG as SingleEvoDAG
        X, y, _, _ = load_data(root, dataset)
        params = configs.params_for(config, quick)
        t0 = time.perf_counter()
        # no test set: it would double the memory of the population
        member = SingleEvoDAG(seed=i, **params).fit(X, y)
        seconds = time.perf_counter() - t0
        model = member.model
        write_atomic(model_path, gzip.compress(pickle.dumps(model)))
        info = {"seed": i, "fit_seconds": seconds,
                "host": socket.gethostname(),
                "size": int(model.size), "height": int(model.height),
                "max_rss_kb": int(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss)}
        try:
            info["fitness_vs"] = float(model.fitness_vs)
        except Exception:
            pass
        if error_path.exists():
            error_path.unlink()
        write_json(json_path, info)  # last: it says that the member is done
        return i, None
    except Exception:
        text = traceback.format_exc()
        write_atomic(error_path, text.encode("utf8"))
        return i, text


def done_members(directory, members):
    "Members that have their JSON file"
    return [i for i in range(members)
            if member_paths(directory, i)[1].exists()]


def load_members(directory, members):
    "``(models, infos)`` of the members ``0 .. members - 1``, in that order"
    models, infos = [], []
    for i in range(members):
        model_path, json_path, _ = member_paths(directory, i)
        with gzip.open(model_path) as fpt:
            models.append(pickle.load(fpt))
        with open(json_path) as fpt:
            infos.append(json.load(fpt))
    return models, infos


def build_result(root, dataset, config, quick, members, params):
    "Assemble the ensemble from the member files and measure it"
    from EvoDAG.model import Ensemble
    directory = unit_dir(root, dataset, config, quick)
    X_train, y_train, X_test, y_test = load_data(root, dataset)
    models, infos = load_members(directory, members)
    ensemble = Ensemble(models)
    t0 = time.perf_counter()
    hy = np.asarray(ensemble.predict(X_test))
    predict_seconds = time.perf_counter() - t0
    scores = np.asarray(ensemble.decision_function(X_test))
    hy_median = np.argmax(scores, axis=1)
    labels = getattr(ensemble, "_labels", None)
    if labels is not None:
        hy_median = np.asarray(labels)[hy_median]
    fit_seconds = float(sum(x["fit_seconds"] for x in infos))
    y_test = np.asarray(y_test).astype(np.int64)
    hy = np.asarray(hy).astype(np.int64)
    hy_median = np.asarray(hy_median).astype(np.int64)
    # before result.json: the unit is finished when result.json exists
    write_npz(directory / "predictions.npz", y_test=y_test, hy=hy,
              hy_median=hy_median,
              decision_function=scores.astype(np.float32))
    recall, recall_median = recall_macro(y_test, hy), recall_macro(
        y_test, hy_median)
    return {
        "dataset": dataset, "config": config, "quick": bool(quick),
        "members": members,
        "macro_f1": macro_f1(y_test, hy),
        "macro_f1_median": macro_f1(y_test, hy_median),
        "macro_recall": recall, "ber": 1.0 - recall,
        "macro_recall_median": recall_median, "ber_median": 1.0 - recall_median,
        "accuracy": accuracy(y_test, hy),
        "fit_seconds": fit_seconds,
        "fit_seconds_per_sample": fit_seconds / len(y_train),
        "predict_seconds": predict_seconds,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_classes": int(len(np.unique(np.concatenate([y_train, y_test])))),
        "max_rss_kb": max(x["max_rss_kb"] for x in infos),
        "params": params, "environment": environment(),
    }


def run_members(tasks, directory, workers, dataset, config, log=print):
    """Fit the members of ``tasks`` (``fit_member`` arguments), at most
    ``workers`` processes at a time, and return the sorted indices of those
    that have no JSON file afterwards.

    One fresh process per member (the memory is returned and the peak memory
    is per member). The processes are not a ``multiprocessing.Pool`` because a
    Pool never returns when a worker is killed by a signal (the out-of-memory
    killer, a crash of a kernel) and the unit would hold its cores for ever:
    here a process that ends without its JSON file is a failed member."""
    ctx = mp_context()
    pending = list(tasks)
    running = {}
    failed = []
    workers = max(1, min(workers, len(pending)))
    try:
        while pending or running:
            while pending and len(running) < workers:
                task = pending.pop(0)
                i = task[4]
                error_path = member_paths(directory, i)[2]
                if error_path.exists():  # the error of an earlier attempt
                    error_path.unlink()
                proc = ctx.Process(target=fit_member, args=(task,))
                proc.start()
                running[i] = proc
            for i, proc in list(running.items()):
                proc.join(timeout=0.2)
                if proc.is_alive():
                    continue
                del running[i]
                _, json_path, error_path = member_paths(directory, i)
                if proc.exitcode != 0 and not json_path.exists():
                    # killed by a signal or crashed outside Python
                    if not error_path.exists():
                        write_atomic(error_path, (
                            "process died with exit code %s\n"
                            % proc.exitcode).encode("utf8"))
                if json_path.exists():
                    log("%s %s: member %d done" % (dataset, config, i))
                else:
                    failed.append(i)
                    text = (error_path.read_text() if error_path.exists()
                            else "no result file")
                    log("%s %s: member %d FAILED\n%s"
                        % (dataset, config, i, text))
    finally:
        for proc in running.values():  # an interruption of this process
            proc.terminate()
            proc.join()
    return sorted(failed)


def result_has_members(path, members):
    """Whether ``path`` is a ``result.json`` of an ensemble of ``members``
    members (a missing or unreadable file is not)."""
    try:
        with open(path) as fpt:
            return json.load(fpt).get("members") == members
    except (OSError, ValueError, AttributeError):
        return False


def run_evodag_unit(root, dataset, config, workers=1, members=None,
                    quick=False, log=print):
    """Run the missing members of a unit and, when all are done, write
    ``result.json``. Returns the exit status (1 if a member failed)."""
    root = Path(root)
    members = configs.n_members(quick) if members is None else members
    directory = unit_dir(root, dataset, config, quick)
    if result_has_members(directory / "result.json", members):
        log("%s %s: already done" % (dataset, config))
        return 0
    load_data(root, dataset)  # fail early if the dataset is not prepared
    directory.mkdir(parents=True, exist_ok=True)
    done = set(done_members(directory, members))
    pending = [i for i in range(members) if i not in done]
    failed = []
    if pending:
        tasks = [(str(root), dataset, config, quick, i) for i in pending]
        failed = run_members(tasks, directory, workers, dataset, config, log)
    if failed:
        log("%s %s: members %s failed" % (dataset, config, sorted(failed)))
        return 1
    result = build_result(root, dataset, config, quick, members,
                          configs.params_for(config, quick))
    write_json(directory / "result.json", result)
    log("%s %s: macro-F1 %.4f (median %.4f), %.1f s of training"
        % (dataset, config, result["macro_f1"], result["macro_f1_median"],
           result["fit_seconds"]))
    return 0


# -- the scikit-learn unit -------------------------------------------------

def make_classifier(name):
    """A new classifier with default parameters (and ``random_state=0`` when
    it has one), or ``None`` if the installed scikit-learn lacks it."""
    for label, module, cls in SKLEARN:
        if label == name:
            try:
                klass = getattr(importlib.import_module(module), cls)
            except (ImportError, AttributeError):
                return None
            estimator = klass()
            if "random_state" in estimator.get_params():
                estimator.set_params(random_state=0)
            return estimator
    raise ValueError("unknown classifier %r" % name)


def _fit_classifier(name, root, dataset, quick, conn):
    "Child process: fit, measure and send a dictionary through ``conn``"
    try:
        X_train, y_train, X_test, y_test = load_data(root, dataset)
        estimator = make_classifier(name)
        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        t1 = time.perf_counter()
        hy = np.asarray(estimator.predict(X_test))
        t2 = time.perf_counter()
        recall = recall_macro(y_test, hy)
        result = {"macro_f1": macro_f1(y_test, hy),
                  "macro_recall": recall, "ber": 1.0 - recall,
                  "accuracy": accuracy(y_test, hy),
                  "fit_seconds": t1 - t0,
                  "fit_seconds_per_sample": (t1 - t0) / len(y_train),
                  "predict_seconds": t2 - t1,
                  "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                  "n_classes": int(len(np.unique(
                      np.concatenate([y_train, y_test])))),
                  "max_rss_kb": int(resource.getrusage(
                      resource.RUSAGE_SELF).ru_maxrss),
                  # the predictions cross the pipe as a list; the parent
                  # stores them in <Name>.npz, not in the JSON
                  "hy": hy.astype(np.int64).tolist()}
    except BaseException as error:
        result = {"error": "%s: %s" % (type(error).__name__, error)}
    conn.send(result)
    conn.close()


def run_classifier(name, root, dataset, quick, timeout_seconds):
    "Fit one classifier in a child process; errors and timeouts are results"
    import sklearn
    if make_classifier(name) is None:
        return {"error": "not in scikit-learn %s" % sklearn.__version__}
    ctx = mp_context()
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_fit_classifier,
                          args=(name, str(root), dataset, quick, child))
    process.start()
    child.close()
    if parent.poll(timeout_seconds):
        try:
            result = parent.recv()
        except EOFError:
            result = {"error": "the process died (exit code %s)"
                      % process.exitcode}
        process.join()
        return result
    process.terminate()
    process.join()
    return {"error": "timeout"}


def run_sklearn_unit(root, dataset, quick=False,
                     timeout_hours=DEFAULT_SKLEARN_TIMEOUT_HOURS, names=None,
                     log=print):
    """Fit the classifiers that have no result file (all sixteen by default,
    or ``names``); returns the exit status 0. Each result is ``<Name>.json``
    and, when the fit succeeded, ``<Name>.npz`` with ``y_test`` and ``hy``,
    written first."""
    directory = unit_dir(root, dataset, "sklearn", quick)
    directory.mkdir(parents=True, exist_ok=True)
    _, _, _, y_test = load_data(root, dataset)
    for name in names or [n for n, _, _ in SKLEARN]:
        path = directory / (name + ".json")
        if path.exists():
            continue
        result = run_classifier(name, root, dataset, quick,
                                timeout_hours * 3600.0)
        hy = result.pop("hy", None)
        if hy is not None:
            write_npz(directory / (name + ".npz"),
                      y_test=np.asarray(y_test).astype(np.int64),
                      hy=np.asarray(hy, dtype=np.int64))
        record = {"dataset": dataset, "classifier": name,
                  "quick": bool(quick), **result,
                  "environment": environment(sklearn=True)}
        write_json(path, record)
        log("%s %s: %s" % (dataset, name, "error: " + result["error"]
                           if "error" in result else
                           "macro-F1 %.4f" % result["macro_f1"]))
    return 0


# -- command line ----------------------------------------------------------

def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Run one unit of the selection heuristics experiments.")
    parser.add_argument("--dataset", required=True,
                        help="name of a prepared dataset")
    parser.add_argument("--config", help="a configuration of configs.py")
    parser.add_argument("--sklearn", action="store_true",
                        help="run the scikit-learn classifiers instead")
    parser.add_argument("--classifiers", nargs="+", metavar="NAME",
                        help="with --sklearn, only these classifiers "
                             "(default: all sixteen)")
    parser.add_argument("--classifier-index", type=int, metavar="N",
                        help="with --sklearn, only the N-th classifier of "
                             "the table (0 to 15), for a job array "
                             "($SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--task-file", help="JSON list of configurations "
                        "written by submit.py")
    parser.add_argument("--task-index", type=int,
                        help="position in --task-file (default: "
                             "$SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--workers", type=int, default=1,
                        help="members trained at the same time")
    parser.add_argument("--members", type=int,
                        help="members of the ensemble (default 30, 3 with "
                             "--quick)")
    parser.add_argument("--quick", action="store_true",
                        help="tiny parameters and 3 members, to test")
    parser.add_argument("--sklearn-timeout", type=float,
                        default=DEFAULT_SKLEARN_TIMEOUT_HOURS,
                        help="hours allowed to each scikit-learn classifier")
    parser.add_argument("--root", help="experiments directory (default: "
                        "$%s or <repository>/experiments)" % ENV_ROOT)
    args = parser.parse_args(argv)
    root = resolve_root(args.root)
    if args.sklearn:
        names = args.classifiers
        unknown = [n for n in names or [] if n not in configs.SKLEARN_NAMES]
        if unknown:
            parser.error("unknown classifiers: %s; choose among %s"
                         % (", ".join(unknown),
                            ", ".join(configs.SKLEARN_NAMES)))
        if args.classifier_index is not None:
            if not 0 <= args.classifier_index < len(configs.SKLEARN_NAMES):
                parser.error("--classifier-index must be between 0 and %d"
                             % (len(configs.SKLEARN_NAMES) - 1))
            names = [configs.SKLEARN_NAMES[args.classifier_index]]
        return run_sklearn_unit(root, args.dataset, args.quick,
                                args.sklearn_timeout, names)
    config = args.config
    if args.task_file:
        index = args.task_index
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        with open(args.task_file) as fpt:
            config = json.load(fpt)[index]
    if config is None:
        parser.error("give --config, --task-file or --sklearn")
    if config not in configs.CONFIGS:
        parser.error("unknown configuration %r; choose among %s"
                     % (config, ", ".join(sorted(configs.CONFIGS))))
    return run_evodag_unit(root, args.dataset, config, args.workers,
                           args.members, args.quick)


if __name__ == "__main__":
    sys.exit(main())
