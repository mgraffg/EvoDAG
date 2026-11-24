"""Tests of the scripts of ``benchmarks/ecj2022/`` that run, submit and
summarise the experiments. No network and no Slurm: everything is written in
``tmp_path`` with ``--root`` pointing there, on a tiny synthetic dataset."""
import json
import multiprocessing
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarks" / "ecj2022"
sys.path.insert(0, str(SCRIPTS))

import configs  # noqa: E402
import run as runner  # noqa: E402
import submit  # noqa: E402
import summary  # noqa: E402

# the scripts run on a Slurm cluster: the members are forked processes and the
# tests kill one with SIGKILL and write shell stand-ins for sbatch and squeue
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the experiment scripts need a POSIX platform")


def make_root(tmp_path, names=("toy",), n=120, n_train=84):
    "An experiments directory with synthetic prepared datasets"
    rng = np.random.RandomState(0)
    manifest = []
    for name in names:
        y = rng.randint(0, 3, n)
        X = rng.randn(n, 4)
        X[:, 0] += y
        X[:, 1] -= y
        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        np.savez(data / (name + ".npz"), X_train=X[:n_train],
                 X_test=X[n_train:], y_train=y[:n_train].astype(np.int64),
                 y_test=y[n_train:].astype(np.int64),
                 classes=np.array(["a", "b", "c"]))
        manifest.append({"name": name, "n_train": n_train, "n_test": n - n_train,
                         "n_variables": 4, "n_classes": 3})
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


# -- configurations --------------------------------------------------------

def test_configs_are_the_21_of_the_article():
    assert len(configs.CONFIGS) == 21
    assert "default" not in configs.CONFIGS
    # every configuration has the fitness of the article
    assert all(configs.params_for(c)["fitness_function"] == "macro-F1"
               for c in configs.CONFIGS)
    assert configs.CONFIGS["agr-rnd"] == dict(
        selection="agreement", negative_selection=False,
        first_selection="random", heuristic_functions="tailored")
    assert configs.CONFIGS["ads-fit--fit-all"] == dict(
        selection="angle", negative_selection=True,
        first_selection="fitness", heuristic_functions="all")
    assert configs.CONFIGS["nvs-rnd"] == dict(
        selection="novelty", negative_selection=False,
        heuristic_functions="tailored")
    assert configs.CONFIGS["fit-fit"] == dict(
        selection="fitness", negative_selection=True)
    assert configs.CONFIGS["rnd-rnd"] == dict(
        selection="random", negative_selection=False)
    assert configs.params_for("fit-fit") == dict(
        fitness_function="macro-F1", random_generations=0, time_limit=None,
        selection="fitness", negative_selection=True)
    assert configs.params_for("fit-fit", quick=True)["popsize"] == 50
    assert configs.n_members() == 30 and configs.n_members(True) == 3
    with pytest.raises(ValueError):
        configs.params_for("nothing")


def test_the_tables_resolve():
    assert len(configs.TABLE_HEURISTICS) == 14
    assert len(configs.TABLE_SCHEMES) == 15
    assert len(configs.SKLEARN_NAMES) == 16
    allowed = set(configs.CONFIGS) | set(configs.SKLEARN_NAMES)
    for names in configs.TABLES.values():
        assert set(names) <= allowed and len(set(names)) == len(names)
    assert len(configs.TABLE_CLASSIFIERS) == 21
    assert set(configs.ARTICLE_NAMES) == set(configs.CONFIGS)
    assert set(configs.SKLEARN_ARTICLE_NAMES) == set(configs.SKLEARN_NAMES)


def test_every_configuration_is_accepted_by_the_engine():
    from EvoDAG.model import EvoDAG as SingleEvoDAG
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0)
    for name in configs.CONFIGS:
        member = SingleEvoDAG(seed=0, **configs.params_for(name, True))
        assert member.fit(X, y).predict(X).shape == (60,), name


# -- measures --------------------------------------------------------------

def test_macro_f1_equals_scikit_learn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(1)
    for _ in range(5):
        y = rng.randint(0, 4, 50)
        hy = rng.randint(0, 3, 50)  # class 3 is never predicted
        assert runner.macro_f1(y, hy) == pytest.approx(
            metrics.f1_score(y, hy, average="macro", zero_division=0))
    assert runner.macro_f1([0, 1], [0, 1]) == 1.0
    assert runner.macro_f1([0, 0], [1, 1]) == 0.0


def test_macro_recall_and_ber():
    # class 2 is never predicted; class 5 of hy is not in y
    y = [0, 0, 1, 1, 2, 2]
    hy = [0, 1, 1, 1, 0, 0]
    assert runner.recall_macro(y, hy) == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert runner.recall_macro([0, 0, 1, 1], [0, 5, 1, 1]) == pytest.approx(
        0.75)
    assert runner.recall_macro([3, 4], [3, 4]) == 1.0
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.RandomState(2)
    yy, hh = rng.randint(0, 4, 60), rng.randint(0, 4, 60)
    assert runner.recall_macro(yy, hh) == pytest.approx(
        metrics.recall_score(yy, hh, average="macro", zero_division=0))


def test_the_fork_start_method_is_used_on_linux():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork on this platform")
    assert runner.mp_context().get_start_method() == "fork"


# -- the EvoDAG unit -------------------------------------------------------

def test_thread_variables_are_set_to_one():
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS")}
    code = ("import os, sys; sys.path.insert(0, %r); import run; "
            "print([os.environ[k] for k in ('OMP_NUM_THREADS', "
            "'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')])" % str(SCRIPTS))
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True,
                         stdout=subprocess.PIPE).stdout.decode()
    assert out.strip() == "['1', '1', '1']"


def test_no_test_set_is_given_to_fit(tmp_path, monkeypatch):
    from EvoDAG.model import EvoDAG as SingleEvoDAG
    root = make_root(tmp_path)
    calls = []
    original = SingleEvoDAG.fit

    def fit(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SingleEvoDAG, "fit", fit)
    runner.unit_dir(root, "toy", "agr-rnd", True).mkdir(parents=True)
    assert runner.fit_member((str(root), "toy", "agr-rnd", True, 0)) == (0, None)
    (args, kwargs), = calls
    assert len(args) == 2 and not kwargs


def test_quick_unit_resumes_and_matches_evodage(tmp_path):
    from EvoDAG import EvoDAGE
    root = make_root(tmp_path)
    status = runner.run_evodag_unit(root, "toy", "agr-rnd", workers=2,
                                    quick=True, log=lambda *a: None)
    assert status == 0
    directory = runner.unit_dir(root, "toy", "agr-rnd", quick=True)
    assert directory == tmp_path / "results-quick" / "toy" / "agr-rnd"
    for i in range(3):
        assert (directory / ("member-%d.model.gz" % i)).exists()
        info = json.loads((directory / ("member-%d.json" % i)).read_text())
        assert info["seed"] == i and info["fit_seconds"] > 0
        assert info["max_rss_kb"] > 0 and info["size"] > 0
    result = json.loads((directory / "result.json").read_text())
    assert result["quick"] is True and result["members"] == 3
    assert result["n_train"] == 84 and result["n_test"] == 36
    assert result["n_classes"] == 3
    assert result["fit_seconds_per_sample"] == pytest.approx(
        result["fit_seconds"] / 84)
    for key in ("macro_f1", "macro_f1_median", "accuracy", "predict_seconds",
                "max_rss_kb", "params", "environment"):
        assert key in result
    assert set(result["environment"]) >= {"evodag", "numpy", "python",
                                          "host", "git_commit"}
    # the ensemble of the files predicts what EvoDAGE predicts
    X, y, Xt, yt = runner.load_data(root, "toy")
    params = configs.params_for("agr-rnd", quick=True)
    expected = EvoDAGE(n_estimators=3, seed=0, **params).fit(X, y).predict(Xt)
    models, _ = runner.load_members(directory, 3)
    from EvoDAG.model import Ensemble
    got = Ensemble(models).predict(Xt)
    assert got.tolist() == np.asarray(expected).tolist()
    assert set(np.unique(got)) <= set(np.unique(y))
    assert result["macro_f1"] == runner.macro_f1(yt, expected)
    # a second run does nothing
    stamps = {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
    assert runner.main(["--dataset", "toy", "--config", "agr-rnd", "--quick",
                        "--root", str(root)]) == 0
    assert {p.name: p.stat().st_mtime_ns
            for p in directory.iterdir()} == stamps
    # removing a member and the result refits only that member
    (directory / "member-1.json").unlink()
    (directory / "result.json").unlink()
    assert runner.main(["--dataset", "toy", "--config", "agr-rnd", "--quick",
                        "--root", str(root), "--workers", "2"]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
    assert after["member-0.json"] == stamps["member-0.json"]
    assert after["member-2.json"] == stamps["member-2.json"]
    assert after["member-1.json"] != stamps.get("member-1.json")
    again = json.loads((directory / "result.json").read_text())
    assert again["macro_f1"] == result["macro_f1"]
    assert again["macro_f1_median"] == result["macro_f1_median"]


def test_task_file_picks_the_configuration(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    task_file = tmp_path / "toy.tasks.json"
    task_file.write_text(json.dumps(["fit-fit", "rnd-rnd"]))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    assert runner.main(["--dataset", "toy", "--task-file", str(task_file),
                        "--quick", "--root", str(root)]) == 0
    assert (tmp_path / "results-quick" / "toy" / "rnd-rnd"
            / "result.json").exists()
    assert not (tmp_path / "results-quick" / "toy" / "fit-fit").exists()
    assert runner.main(["--dataset", "toy", "--task-file", str(task_file),
                        "--task-index", "0", "--quick",
                        "--root", str(root)]) == 0
    assert (tmp_path / "results-quick" / "toy" / "fit-fit"
            / "result.json").exists()


def test_a_failing_member_leaves_an_error_file(tmp_path, monkeypatch):
    from EvoDAG.model import EvoDAG as SingleEvoDAG
    root = make_root(tmp_path)
    original = SingleEvoDAG.fit

    def fit(self, *args, **kwargs):
        if self._m._seed == 1:
            raise RuntimeError("boom")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SingleEvoDAG, "fit", fit)
    status = runner.main(["--dataset", "toy", "--config", "fit-fit",
                          "--quick", "--root", str(root), "--workers", "2"])
    assert status == 1
    directory = tmp_path / "results-quick" / "toy" / "fit-fit"
    assert "boom" in (directory / "member-1.error.txt").read_text()
    assert (directory / "member-0.json").exists()
    assert (directory / "member-2.json").exists()
    assert not (directory / "member-1.json").exists()
    assert not (directory / "result.json").exists()
    # the unit resumes: the member is fitted next time and the error is gone
    monkeypatch.setattr(SingleEvoDAG, "fit", original)
    assert runner.main(["--dataset", "toy", "--config", "fit-fit", "--quick",
                        "--root", str(root)]) == 0
    assert not (directory / "member-1.error.txt").exists()
    assert (directory / "result.json").exists()


def run_with_timeout(function, seconds=120):
    "Call ``function`` in a thread; the test fails instead of hanging"
    box = []
    thread = threading.Thread(target=lambda: box.append(function()),
                              daemon=True)
    thread.start()
    thread.join(seconds)
    assert box, "the unit did not return in %d s (a hang)" % seconds
    return box[0]


def test_a_member_killed_by_a_signal_is_a_failure_not_a_hang(
        tmp_path, monkeypatch):
    from EvoDAG.model import EvoDAG as SingleEvoDAG
    root = make_root(tmp_path)
    original = SingleEvoDAG.fit

    def fit(self, *args, **kwargs):
        if self._m._seed == 1:
            os.kill(os.getpid(), signal.SIGKILL)  # like the OOM killer
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SingleEvoDAG, "fit", fit)
    messages = []
    status = run_with_timeout(lambda: runner.run_evodag_unit(
        root, "toy", "fit-fit", workers=2, members=3, quick=True,
        log=messages.append))
    assert status == 1
    directory = tmp_path / "results-quick" / "toy" / "fit-fit"
    assert "exit code" in (directory / "member-1.error.txt").read_text()
    assert (directory / "member-0.json").exists()
    assert (directory / "member-2.json").exists()
    assert not (directory / "member-1.json").exists()
    assert not (directory / "result.json").exists()
    assert any("member 1 FAILED" in m for m in messages)
    # one worker works too
    (directory / "member-0.json").unlink()
    status = run_with_timeout(lambda: runner.run_evodag_unit(
        root, "toy", "fit-fit", workers=1, members=3, quick=True,
        log=lambda *a: None))
    assert status == 1 and (directory / "member-0.json").exists()
    # the unit resumes
    monkeypatch.setattr(SingleEvoDAG, "fit", original)
    assert run_with_timeout(lambda: runner.run_evodag_unit(
        root, "toy", "fit-fit", workers=2, quick=True,
        log=lambda *a: None)) == 0
    assert not (directory / "member-1.error.txt").exists()
    assert (directory / "result.json").exists()


def test_predictions_and_provenance_are_stored(tmp_path):
    root = make_root(tmp_path)
    assert runner.run_evodag_unit(root, "toy", "fit-fit", workers=2,
                                  quick=True, log=lambda *a: None) == 0
    directory = tmp_path / "results-quick" / "toy" / "fit-fit"
    result = json.loads((directory / "result.json").read_text())
    with np.load(directory / "predictions.npz") as stored:
        assert set(stored.files) == {"y_test", "hy", "hy_median",
                                     "decision_function"}
        assert stored["y_test"].dtype == np.int64
        assert stored["hy"].dtype == np.int64
        assert stored["hy_median"].dtype == np.int64
        assert stored["decision_function"].dtype == np.float32
        assert stored["y_test"].shape == stored["hy"].shape == (36,)
        assert stored["hy_median"].shape == (36,)
        assert stored["decision_function"].shape == (36, 3)
        y, hy, hy_median = (stored["y_test"], stored["hy"],
                            stored["hy_median"])
    assert result["macro_recall"] == pytest.approx(runner.recall_macro(y, hy))
    assert result["macro_recall_median"] == pytest.approx(
        runner.recall_macro(y, hy_median))
    assert "ber" not in result and "ber_median" not in result
    assert result["macro_f1"] == pytest.approx(runner.macro_f1(y, hy))
    assert not list(directory.glob("*.tmp"))
    # provenance
    assert result["environment"]["host"]
    assert result["environment"]["git_dirty"] in (True, False, None)
    info = json.loads((directory / "member-0.json").read_text())
    assert info["host"] == result["environment"]["host"]


def test_git_dirty(monkeypatch):
    def fake(output, code=0):
        return lambda argv, **kw: subprocess.CompletedProcess(
            argv, code, stdout=output, stderr=b"")
    monkeypatch.setattr(runner.subprocess, "run", fake(b" M run.py\n"))
    assert runner.git_dirty() is True
    monkeypatch.setattr(runner.subprocess, "run", fake(b""))
    assert runner.git_dirty() is False
    monkeypatch.setattr(runner.subprocess, "run", fake(b"", 128))
    assert runner.git_dirty() is None

    def missing(argv, **kw):
        raise OSError("no git")
    monkeypatch.setattr(runner.subprocess, "run", missing)
    assert runner.git_dirty() is None


def test_a_result_with_fewer_members_is_not_the_finished_unit(tmp_path):
    root = make_root(tmp_path)
    directory = tmp_path / "results-quick" / "toy" / "fit-fit"
    assert runner.main(["--dataset", "toy", "--config", "fit-fit", "--quick",
                        "--members", "2", "--root", str(root)]) == 0
    assert json.loads((directory / "result.json").read_text())[
        "members"] == 2
    assert not submit.unit_done(root, "toy", "fit-fit", quick=True)
    assert summary.read_result(root, "toy", "fit-fit") is None
    # the summary reads the quick results of the expected size only
    assert summary.read_result(root, "toy", "fit-fit", quick=True) is None
    stamps = {p.name: p.stat().st_mtime_ns for p in directory.iterdir()}
    assert runner.main(["--dataset", "toy", "--config", "fit-fit", "--quick",
                        "--root", str(root)]) == 0
    assert json.loads((directory / "result.json").read_text())[
        "members"] == 3
    # the two members were reused, the third was fitted
    for name in ("member-0.json", "member-1.json"):
        assert (directory / name).stat().st_mtime_ns == stamps[name]
    assert (directory / "member-2.json").exists()
    assert submit.unit_done(root, "toy", "fit-fit", quick=True)
    assert summary.read_result(root, "toy", "fit-fit", quick=True)["members"] \
        == 3


def test_a_missing_dataset_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="datasets.py"):
        runner.run_evodag_unit(tmp_path, "nothing", "fit-fit", quick=True)


# -- the scikit-learn unit -------------------------------------------------

def test_sklearn_unit(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    root = make_root(tmp_path)
    assert runner.run_sklearn_unit(
        root, "toy", quick=True, names=["DecisionTree", "GaussianNB"],
        log=lambda *a: None) == 0
    directory = tmp_path / "results-quick" / "toy" / "sklearn"
    for name in ("DecisionTree", "GaussianNB"):
        record = json.loads((directory / (name + ".json")).read_text())
        assert 0 <= record["macro_f1"] <= 1 and record["fit_seconds"] >= 0
        assert record["n_train"] == 84 and record["n_test"] == 36
        assert "scikit-learn" in record["environment"]
        assert record["fit_seconds_per_sample"] == pytest.approx(
            record["fit_seconds"] / 84)
        assert "hy" not in record
        with np.load(directory / (name + ".npz")) as stored:
            assert set(stored.files) == {"y_test", "hy"}
            assert stored["y_test"].shape == stored["hy"].shape == (36,)
            assert stored["hy"].dtype == np.int64
            assert record["macro_recall"] == pytest.approx(
                runner.recall_macro(stored["y_test"], stored["hy"]))
            assert "ber" not in record
    stamp = (directory / "GaussianNB.json").stat().st_mtime_ns
    runner.run_sklearn_unit(root, "toy", quick=True, names=["GaussianNB"],
                            log=lambda *a: None)
    assert (directory / "GaussianNB.json").stat().st_mtime_ns == stamp


def test_sklearn_classifier_selection(tmp_path, capsys):
    pytest.importorskip("sklearn")
    root = make_root(tmp_path)
    directory = tmp_path / "results-quick" / "toy" / "sklearn"
    # the third classifier of the table only
    assert runner.main(["--dataset", "toy", "--sklearn", "--quick",
                        "--classifier-index", "2", "--root", str(root)]) == 0
    assert [p.name for p in sorted(directory.glob("*.json"))] == [
        configs.SKLEARN_NAMES[2] + ".json"]
    assert runner.main(["--dataset", "toy", "--sklearn", "--quick",
                        "--classifiers", "GaussianNB", "DecisionTree",
                        "--root", str(root)]) == 0
    assert sorted(p.stem for p in directory.glob("*.json")) == sorted([
        configs.SKLEARN_NAMES[2], "GaussianNB", "DecisionTree"])
    with pytest.raises(SystemExit):
        runner.main(["--dataset", "toy", "--sklearn", "--quick",
                     "--classifiers", "Nothing", "--root", str(root)])
    with pytest.raises(SystemExit):
        runner.main(["--dataset", "toy", "--sklearn", "--quick",
                     "--classifier-index", "16", "--root", str(root)])
    capsys.readouterr()


def test_sklearn_errors_are_results(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    root = make_root(tmp_path)

    class Sleeper:
        def fit(self, X, y):
            time.sleep(30)

    class Broken:
        def fit(self, X, y):
            raise ValueError("cannot fit this")

    def fake(name):
        return {"SVC": Sleeper(), "MLP": Broken(), "SGD": None}[name]
    monkeypatch.setattr(runner, "make_classifier", fake)
    started = time.time()
    runner.run_sklearn_unit(root, "toy", quick=True,
                            timeout_hours=2 / 3600.,
                            names=["SVC", "MLP", "SGD"], log=lambda *a: None)
    assert time.time() - started < 20
    directory = tmp_path / "results-quick" / "toy" / "sklearn"
    read = lambda n: json.loads((directory / (n + ".json")).read_text())
    assert read("SVC")["error"] == "timeout"
    assert "cannot fit this" in read("MLP")["error"]
    assert read("SGD")["error"].startswith("not in scikit-learn ")
    assert not list(directory.glob("*.npz"))  # no predictions of errors


# -- resources and submission ----------------------------------------------

def test_the_memory_factor_is_the_one_of_the_pilot():
    # 1.43 measured, times 1.25 and rounded up; never below 1.5, never above 3
    assert submit.MEMORY_FACTOR == 1.8
    assert 1.5 <= submit.MEMORY_FACTOR <= 3.0


def test_resources():
    assert submit.member_mb(3, 84) == 1515
    assert submit.member_mb(2, 199523) == 24486
    assert submit.member_mb(26, 14000) == 22467
    assert submit.member_mb(3, 84, popsize=50) == 1501
    # the cap of workers
    assert submit.resources(3, 84) == (8, 8 * 1515)
    # the cap of memory
    assert submit.resources(2, 199523) == (4, 4 * 24486)
    assert submit.resources(26, 14000) == (5, 5 * 22467)
    # a member that does not fit still gets one worker
    assert submit.resources(2, 199523, max_mem_mb=1000) == (1, 24486)
    assert submit.resources(3, 84, max_workers=2) == (2, 2 * 1515)
    assert submit.resources(3, 84, quick=True)[1] == 8 * 1501


def fake_sbatch(tmp_path, monkeypatch):
    "An ``sbatch`` that records its arguments and prints a job id"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "sbatch"
    script.write_text("#!/bin/sh\necho \"$@\" >> %s\necho 4242\n"
                      % (tmp_path / "sbatch.log"))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", "%s:%s" % (bindir, os.environ["PATH"]))
    return tmp_path / "sbatch.log"


def snapshot(path):
    "Sorted list of the files and directories under ``path`` with sizes"
    return sorted((str(p.relative_to(path)), p.stat().st_size
                   if p.is_file() else -1) for p in Path(path).rglob("*"))


def done_result(directory, members=30):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"members": members}))


def test_dry_run_prints_the_commands_and_submits_nothing(
        tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path, names=("alpha", "beta"))
    # the result of one configuration of alpha exists and is left out
    done_result(runner.unit_dir(root, "alpha", "fit-fit"))
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    before = snapshot(root)
    assert submit.main(["--dry-run", "--root", str(root)]) == 0
    # nothing was written: no logs directory, no task file
    assert snapshot(root) == before
    assert not (root / "logs").exists()
    out = capsys.readouterr().out.splitlines()
    # an array and a scikit-learn array per dataset, each with its list
    commands = [x for x in out if x.startswith("sbatch")]
    notes = [x for x in out if x.startswith("#")]
    assert len(commands) == 4 and len(notes) == 4
    alpha, alpha_sk, beta, beta_sk = commands
    assert alpha.startswith("sbatch --parsable --job-name=ecj-alpha ")
    assert "--array=0-19 " in alpha and "--array=0-20 " in beta
    assert "--cpus-per-task=8 --mem=12120M" in alpha
    assert "--job-name=ecj-alpha-sklearn" in alpha_sk
    assert "--array=0-15 " in alpha_sk
    assert "--cpus-per-task=1 --mem=16000M" in alpha_sk
    assert "--sklearn " in alpha_sk and "--task-file" in alpha
    assert alpha_sk.endswith("--classifier-index $SLURM_ARRAY_TASK_ID'")
    assert "--workers 8" in alpha and "--quick" not in alpha
    assert str(root / "logs" / "%x-%A_%a.out") in alpha
    assert "fit-fit" not in notes[0] and "rnd-rnd" in notes[0]
    assert len(notes[0].split(": ")[-1].split()) == 20
    assert "task file %s" % (root / "logs") in notes[0]
    assert ".tasks.json" in notes[0] and ".tasks.json" in alpha
    assert "LogisticRegression" in notes[1]
    # options: restricted configurations, quick, throttle
    assert submit.main(["--dry-run", "--root", str(root), "--datasets",
                        "beta", "--configs", "agr-rnd", "fit-fit", "--quick",
                        "--max-parallel", "3", "--no-sklearn"]) == 0
    line, note = capsys.readouterr().out.splitlines()
    assert "--array=0-1%3 " in line and "--quick" in line
    assert "--mem=%dM" % (8 * 1501) in line
    assert note.endswith(": agr-rnd fit-fit")
    assert snapshot(root) == before


def test_the_task_file_has_a_name_per_submission(tmp_path, monkeypatch,
                                                 capsys):
    root = make_root(tmp_path, names=("alpha", "beta"))
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    monkeypatch.setattr(submit, "timestamp", lambda: "20300101-000000")
    log = fake_sbatch(tmp_path, monkeypatch)
    assert submit.main(["--root", str(root), "--configs", "fit-fit",
                        "--no-sklearn"]) == 0
    first = sorted(p.name for p in (root / "logs").glob("*.tasks.json"))
    assert first == ["alpha.20300101-000000.tasks.json",
                     "beta.20300101-000000.tasks.json"]
    assert json.loads((root / "logs" / first[0]).read_text()) == ["fit-fit"]
    # a second submission does not overwrite the file the running array reads
    monkeypatch.setattr(submit, "timestamp", lambda: "20300101-000100")
    assert submit.main(["--root", str(root), "--configs", "fit-fit",
                        "rnd-rnd", "--no-sklearn"]) == 0
    assert len(list((root / "logs").glob("*.tasks.json"))) == 4
    assert json.loads((root / "logs" / first[0]).read_text()) == ["fit-fit"]
    assert "alpha.20300101-000100.tasks.json" in log.read_text()
    # and run.py reads it by the path it is given
    task_file = root / "logs" / "alpha.20300101-000100.tasks.json"
    assert runner.main(["--dataset", "alpha", "--task-file", str(task_file),
                        "--task-index", "1", "--quick",
                        "--root", str(root)]) == 0
    assert (root / "results-quick" / "alpha" / "rnd-rnd"
            / "result.json").exists()
    capsys.readouterr()


def test_the_heaviest_dataset_is_submitted_first(tmp_path, monkeypatch,
                                                 capsys):
    root = make_root(tmp_path, names=("light", "heavy", "middle"))
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    for entry, (n_train, classes) in zip(
            manifest, ((100, 2), (5000, 3), (500, 20))):
        entry["n_train"], entry["n_classes"] = n_train, classes
    (root / "data" / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    assert submit.main(["--dry-run", "--root", str(root), "--no-sklearn"]) == 0
    names = [x.split("--job-name=")[1].split()[0]
             for x in capsys.readouterr().out.splitlines()
             if x.startswith("sbatch")]
    # 5000 * 3 > 500 * 20 > 100 * 2
    assert names == ["ecj-heavy", "ecj-middle", "ecj-light"]
    # explicit datasets follow the same rule
    assert submit.main(["--dry-run", "--root", str(root), "--no-sklearn",
                        "--datasets", "light", "heavy"]) == 0
    names = [x.split("--job-name=")[1].split()[0]
             for x in capsys.readouterr().out.splitlines()
             if x.startswith("sbatch")]
    assert names == ["ecj-heavy", "ecj-light"]


def test_the_sklearn_array_has_the_pending_classifiers_only(
        tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    directory = runner.unit_dir(root, "toy", "sklearn")
    directory.mkdir(parents=True)
    for i in (1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15):
        (directory / (configs.SKLEARN_NAMES[i] + ".json")).write_text("{}")
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    assert submit.main(["--dry-run", "--root", str(root), "--configs",
                        "fit-fit", "--sklearn-timeout", "2.5"]) == 0
    out = capsys.readouterr().out.splitlines()
    sk = [x for x in out if "ecj-toy-sklearn" in x][0]
    assert "--array=0,3,7 " in sk
    assert "--sklearn-timeout 2.5" in sk
    note = [x for x in out if x.startswith("#")][-1]
    assert note.endswith(": %s" % " ".join(
        configs.SKLEARN_NAMES[i] for i in (0, 3, 7)))
    assert submit.array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert submit.array_spec(range(16)) == "0-15"
    assert submit.array_spec([4]) == "4"


def test_a_unit_that_is_in_the_queue_is_not_submitted(tmp_path, monkeypatch,
                                                      capsys):
    root = make_root(tmp_path, names=("busy", "free"))
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: ["ecj-busy"])
    assert submit.main(["--dry-run", "--root", str(root), "--no-sklearn"]) == 0
    out = capsys.readouterr().out
    assert "ecj-free" in out
    assert "ecj-busy" not in out and "busy array" not in out
    # without anything else to do
    monkeypatch.setattr(submit, "running_jobs",
                        lambda **kw: ["ecj-busy", "ecj-free"])
    assert submit.main(["--dry-run", "--root", str(root),
                        "--no-sklearn"]) == 0
    assert capsys.readouterr().out.strip() == "nothing to submit"


class FakeRun:
    "A ``subprocess.run`` for ``squeue`` that fails ``failures`` times"

    def __init__(self, failures, stdout=b"ecj-toy RUNNING\nother RUNNING\n"):
        self.failures, self.calls, self.stdout = failures, 0, stdout

    def __call__(self, argv, **kwargs):
        self.calls += 1
        code = 1 if self.calls <= self.failures else 0
        return subprocess.CompletedProcess(
            argv, code, stdout=self.stdout if code == 0 else b"",
            stderr=b"slurm_load_jobs error: Socket timed out")


def test_squeue_failures(monkeypatch):
    monkeypatch.setattr(submit.shutil, "which", lambda name: "/bin/squeue")
    sleeps = []
    monkeypatch.setattr(submit.time, "sleep", sleeps.append)
    # it always fails: retried twice, then the program stops
    fake = FakeRun(failures=10)
    monkeypatch.setattr(submit.subprocess, "run", fake)
    with pytest.raises(SystemExit, match="squeue failed.*nothing was "
                                         "submitted"):
        submit.running_jobs()
    assert fake.calls == 3 and sleeps == [5, 5]
    # a transient failure is survived
    fake = FakeRun(failures=2)
    monkeypatch.setattr(submit.subprocess, "run", fake)
    assert submit.running_jobs() == ["ecj-toy"]
    # a time out is a failure too
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 60)
    monkeypatch.setattr(submit.subprocess, "run", timeout)
    with pytest.raises(SystemExit, match="squeue failed"):
        submit.running_jobs(retries=0)
    # without squeue (no Slurm) the queue is empty
    monkeypatch.setattr(submit.shutil, "which", lambda name: None)
    assert submit.running_jobs() == []


def test_a_squeue_failure_submits_nothing(tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    monkeypatch.setattr(submit.shutil, "which", lambda name: "/bin/x")
    monkeypatch.setattr(submit.time, "sleep", lambda s: None)
    monkeypatch.setattr(submit.subprocess, "run", FakeRun(failures=10))
    with pytest.raises(SystemExit, match="squeue failed"):
        submit.main(["--dry-run", "--root", str(root)])
    assert not (root / "logs").exists()


def test_wait(tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    done_result(runner.unit_dir(root, "toy", "fit-fit"))
    answers = [["ecj-toy"], SystemExit("squeue failed"), ["ecj-toy"], []]
    sleeps = []

    def queue(**kwargs):
        answer = answers.pop(0)
        if isinstance(answer, SystemExit):
            raise answer
        return answer
    monkeypatch.setattr(submit, "running_jobs", queue)
    monkeypatch.setattr(submit.time, "sleep", sleeps.append)
    before = snapshot(root)
    assert submit.main(["--wait", "3600", "--root", str(root)]) == 0
    line, = capsys.readouterr().out.splitlines()
    assert line == "queue: 0 jobs left; units finished 1/22, failed 0"
    assert sleeps == [60, 60, 60] and not answers
    assert snapshot(root) == before  # it submits and writes nothing
    # the time runs out: one line and a zero exit status
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: ["ecj-toy"] * 3)
    assert submit.main(["--wait", "0", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == (
        "queue: 3 jobs left; units finished 1/22, failed 0")
    # a queue that cannot be read is unknown
    def broken(**kwargs):
        raise SystemExit("squeue failed")
    monkeypatch.setattr(submit, "running_jobs", broken)
    assert submit.main(["--wait", "0", "--root", str(root)]) == 0
    assert capsys.readouterr().out.startswith("queue: unknown jobs left")


def test_a_dataset_that_is_done_is_not_submitted(tmp_path, monkeypatch,
                                                 capsys):
    root = make_root(tmp_path)
    for name in configs.CONFIGS:
        done_result(runner.unit_dir(root, "toy", name))
    sk = runner.unit_dir(root, "toy", "sklearn")
    sk.mkdir(parents=True)
    for name in configs.SKLEARN_NAMES:
        (sk / (name + ".json")).write_text("{}")
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    assert submit.main(["--dry-run", "--root", str(root)]) == 0
    assert capsys.readouterr().out.strip() == "nothing to submit"
    # a dataset with jobs in the queue is skipped
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: ["ecj-toy"])
    (sk / "SVC.json").unlink()
    submit.main(["--dry-run", "--root", str(root)])
    assert "ecj-toy-sklearn" in capsys.readouterr().out
    # a result with another number of members is not a finished unit
    done_result(runner.unit_dir(root, "toy", "fit-fit"), members=2)
    assert not submit.unit_done(root, "toy", "fit-fit")
    assert submit.unit_done(root, "toy", "rnd-rnd")
    assert submit.pending_configs(root, "toy", list(configs.CONFIGS)) == [
        "fit-fit"]
    (runner.unit_dir(root, "toy", "rnd-rnd") / "result.json").write_text("{")
    assert not submit.unit_done(root, "toy", "rnd-rnd")  # not parsable


def test_real_submission_and_the_missing_sbatch(tmp_path, monkeypatch,
                                               capsys):
    root = make_root(tmp_path)
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    monkeypatch.setattr(submit.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="sbatch"):
        submit.main(["--root", str(root)])
    assert not (root / "logs").exists()
    monkeypatch.undo()
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: [])
    log = fake_sbatch(tmp_path, monkeypatch)
    assert submit.main(["--root", str(root), "--configs", "fit-fit"]) == 0
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and "--job-name=ecj-toy " in calls[0]
    assert "--array=0-15 " in calls[1]
    lines = [json.loads(x) for x in
             (root / "logs" / "submissions.jsonl").read_text().splitlines()]
    assert [x["job_id"] for x in lines] == ["4242", "4242"]
    assert lines[0]["dataset"] == "toy" and lines[0]["tasks"] == ["fit-fit"]
    assert lines[0]["cpus"] == 8 and lines[1]["kind"] == "sklearn"
    capsys.readouterr()


def test_missing_manifest_and_dataset(tmp_path):
    with pytest.raises(SystemExit, match="datasets.py"):
        submit.main(["--dry-run", "--root", str(tmp_path)])
    root = make_root(tmp_path)
    with pytest.raises(SystemExit, match="datasets.py"):
        submit.main(["--dry-run", "--root", str(root), "--datasets",
                     "iris"])


def test_smoke_selects_the_five_datasets():
    assert submit.SMOKE_DATASETS == ["fertility", "iris", "tae", "wine",
                                     "parkinsons"]


def test_status(tmp_path, monkeypatch, capsys):
    root = make_root(tmp_path)
    monkeypatch.setattr(submit, "running_jobs", lambda **kw: ["ecj-toy"])
    done_result(runner.unit_dir(root, "toy", "fit-fit"))
    part = runner.unit_dir(root, "toy", "rnd-rnd")
    part.mkdir(parents=True)
    (part / "member-0.json").write_text("{}")
    (part / "member-1.json").write_text("{}")
    bad = runner.unit_dir(root, "toy", "agr-rnd")
    bad.mkdir(parents=True)
    (bad / "member-0.error.txt").write_text("trace")
    sk = runner.unit_dir(root, "toy", "sklearn")
    sk.mkdir(parents=True)
    (sk / "SVC.json").write_text("{}")
    assert submit.main(["--status", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "toy" in out and "1/22 finished, 2 partial, 1 failed" in out
    assert "rnd-rnd: 2/30 members" in out
    assert "agr-rnd: 1 members with errors" in out
    assert "sklearn: 1/16 classifiers" in out
    assert "in the queue: ecj-toy" in out
    assert "total: 1/22 units finished" in out


# -- the summary -----------------------------------------------------------

def write_result(root, dataset, system, f1=None, seconds=None, error=None,
                 members=None):
    if system in configs.CONFIGS:
        directory = runner.unit_dir(root, dataset, system)
        path = directory / "result.json"
    else:
        directory = runner.unit_dir(root, dataset, "sklearn")
        path = directory / (system + ".json")
    directory.mkdir(parents=True, exist_ok=True)
    if error:
        content = {"error": error}
    else:
        content = {"macro_f1": f1, "macro_f1_median": f1 - 0.1,
                   "fit_seconds_per_sample": seconds,
                   "members": members or configs.n_members()}
        if system not in configs.CONFIGS:
            del content["members"]
    path.write_text(json.dumps(content))


def test_rank_rule():
    values = {"a": 0.9, "b": 0.9, "c": 0.8, "d": 0.95, "e": None}
    assert summary.rank_row(values) == {"d": 1, "a": 2, "b": 2, "c": 4}
    assert summary.rank_row({"a": 3.0, "b": 1.0, "c": 1.0},
                            higher_is_better=False) == {"b": 1, "c": 1,
                                                        "a": 3}
    table = {"d1": {"a": 0.9, "b": 0.8, "c": 0.7},
             "d2": {"a": 0.5, "b": 0.5, "c": 0.9},
             "d3": {"a": 0.9, "b": None, "c": 0.1}}
    averages, complete = summary.average_ranks(table, ["a", "b", "c"])
    assert complete == ["d1", "d2"]
    assert averages == {"a": 1.5, "b": 2.0, "c": 2.0}


def test_summary_tables(tmp_path):
    root = make_root(tmp_path, names=("d1", "d2", "d3"))
    systems = configs.TABLE_HEURISTICS
    for d, dataset in enumerate(("d1", "d2", "d3")):
        for k, system in enumerate(configs.TABLE_CLASSIFIERS):
            if dataset == "d3" and system == "LinearSVC":
                write_result(root, dataset, system, error="timeout")
            elif dataset == "d3" and system == "SVC":
                continue
            else:
                # agr-rnd and rnd-rnd tie on d1
                f1 = 0.5 + 0.01 * ((k * (d + 1)) % 7)
                if dataset == "d1" and system in ("agr-rnd", "rnd-rnd"):
                    f1 = 0.9
                write_result(root, dataset, system, f1=f1,
                             seconds=0.001 * (k + 1))
    for system in systems:
        if system not in configs.TABLE_CLASSIFIERS:
            for dataset in ("d1", "d2", "d3"):
                write_result(root, dataset, system, f1=0.6, seconds=0.1)
    text = summary.summarize(root)
    assert "Combinations of parent and negative selection" in text
    assert "Selection schemes" in text
    assert "Comparison with classifiers" in text
    assert "Friedman test p-value" in text
    assert "Training time per training example" in text
    # the table of heuristics is complete on the 3 datasets
    assert "uses the 3 datasets where every system has a result." in text
    # in the classifiers table d3 is incomplete and listed
    assert "uses the 2 datasets where every system has a result; " \
           "incomplete: d3." in text
    lines = {line.split("|")[1].strip(): line for line in text.splitlines()
             if line.startswith("| d")}
    assert "-" in lines["d3"]
    # a tie shares the best rank: 0.90 (1) twice
    classifiers = text[text.index("### Comparison with classifiers"):]
    d1 = [x for x in classifiers.splitlines() if x.startswith("| d1 ")][0]
    assert d1.count("0.90 (1)") == 2
    assert "**Average rank**" in text
    # the time table is in scientific notation, the fastest has rank 1
    timing = text[text.index("### Training time"):]
    d1 = [x for x in timing.splitlines() if x.startswith("| d1 ")][0]
    assert "1.00e-03 (1)" in d1
    # columns are sorted by average rank
    header = [x for x in text.splitlines() if x.startswith("| dataset")][0]
    ranks = [x for x in text.splitlines()
             if x.startswith("| **Average rank**")][0]
    values = [float(v) for v in ranks.split("|")[2:-1]]
    assert values == sorted(values) and len(values) == 14
    assert header.count("|") == ranks.count("|")


def test_summary_lists_the_datasets_that_were_not_started(tmp_path):
    root = make_root(tmp_path, names=("d1", "d2", "d3"))
    for system in configs.TABLE_SCHEMES:
        write_result(root, "d1", system, f1=0.7, seconds=0.1)
    assert not (runner.results_dir(root) / "d2").exists()
    text = summary.summarize(root)
    assert "Released defaults" not in text
    schemes = text[text.index("### Selection schemes"):
                   text.index("### Comparison with classifiers")]
    assert "incomplete: d2, d3." in schemes
    line = [x for x in schemes.splitlines() if x.startswith("| d3 ")][0]
    assert set(line.replace("|", " ").split()) == {"d3", "-"}


def test_summary_ignores_a_result_with_the_wrong_number_of_members(tmp_path):
    root = make_root(tmp_path)
    write_result(root, "toy", "agr-rnd", f1=0.8, seconds=1.0, members=2)
    write_result(root, "toy", "fit-fit", f1=0.7, seconds=1.0)
    f1, _ = summary.collect(root, ["toy"], ["agr-rnd", "fit-fit"])
    assert f1 == {"toy": {"agr-rnd": None, "fit-fit": 0.7}}


def test_summary_median_and_empty(tmp_path):
    assert "No results" in summary.summarize(tmp_path)
    root = make_root(tmp_path)
    write_result(root, "toy", "agr-rnd", f1=0.8, seconds=1.0)
    f1, _ = summary.collect(root, ["toy"], ["agr-rnd", "fit-fit"])
    assert f1 == {"toy": {"agr-rnd": 0.8, "fit-fit": None}}
    f1, _ = summary.collect(root, ["toy"], ["agr-rnd"], median=True)
    assert f1["toy"]["agr-rnd"] == pytest.approx(0.7)
    out = tmp_path / "out.md"
    assert summary.main(["--root", str(root), "--output", str(out)]) == 0
    assert "Selection schemes" in out.read_text()


PAPER = r"""
\begin{table}
\begin{tabular}{|l||c|c|c|}
\hline
 & acc-rnd & agr-fit & tpot \\ \hline \hline
d1&0.94(1) &0.93(3) &0.5(2) \\ \hline
d2&0.55(2) &-(23) &0.6(1) \\ \hline  \hline
Average rank &1.5&13.0&1.5\\ \hline
\end{tabular}
\label{tab:res_apx_heuristics}
\end{table}
"""


def test_paper_tables_are_parsed(tmp_path):
    values, averages = summary.parse_paper_table(
        PAPER, "tab:res_apx_heuristics")
    assert values == {"d1": {"acc-rnd": 0.94, "agr-fit": 0.93, "tpot": 0.5},
                      "d2": {"acc-rnd": 0.55, "agr-fit": None, "tpot": 0.6}}
    assert averages == {"acc-rnd": 1.5, "agr-fit": 13.0, "tpot": 1.5}
    values, averages = summary.paper_table(PAPER, ["tab:res_apx_heuristics"])
    assert values["d1"] == {"agr-rnd": 0.94, "agr-fit": 0.93}
    assert values["d2"]["agr-fit"] is None
    assert averages == {"agr-rnd": 1.5, "agr-fit": 13.0}
    with pytest.raises(ValueError):
        summary.parse_paper_table(PAPER, "tab:nothing")
    # the comparison tables
    f1 = {"d1": {"agr-rnd": 0.90, "agr-fit": 0.95},
          "d2": {"agr-rnd": 0.50, "agr-fit": 0.7}}
    text = summary.paper_comparison(
        f1, {"agr-rnd": 1.0, "agr-fit": 2.0}, ["agr-rnd", "agr-fit"],
        ["d1", "d2"], PAPER, ["tab:res_apx_heuristics"])
    assert "| agr-rnd | 1.0 | 1.5 |" in text
    assert "| agr-fit | 2.0 | 13.0 |" in text
    assert "| d1 | -0.04 | +0.02 |" in text
    assert "| d2 | -0.05 | - |" in text


def test_summary_with_paper_tex_option(tmp_path):
    root = make_root(tmp_path, names=("d1", "d2"))
    for dataset in ("d1", "d2"):
        for system in configs.TABLE_CLASSIFIERS:
            write_result(root, dataset, system, f1=0.7, seconds=0.1)
        for system in configs.TABLE_SCHEMES + configs.TABLE_HEURISTICS:
            write_result(root, dataset, system, f1=0.7, seconds=0.1)
    labels = [x for part in configs.PAPER_LABELS.values() for x in part]
    tex = tmp_path / "paper.tex"
    tex.write_text("".join(PAPER.replace("tab:res_apx_heuristics", label)
                           for label in labels))
    out = tmp_path / "out.md"
    assert summary.main(["--root", str(root), "--paper-tex", str(tex),
                         "--output", str(out)]) == 0
    text = out.read_text()
    assert text.count("#### Against the article") == 3
    assert "| d1 | -0.24 |" in text
    # the tables of time are not in this fake: one sentence, no failure
    assert "The comparison with the article could not be made" in text
    # only macro-F1 is reported: the option of another measure is gone
    with pytest.raises(SystemExit):
        summary.main(["--root", str(root), "--measure", "ber"])
    # an article that cannot be read
    assert summary.main(["--root", str(root), "--paper-tex",
                         str(tmp_path / "missing.tex"),
                         "--output", str(out)]) == 0
    assert "could not be read" in out.read_text()
    # without the option there is no comparison
    assert summary.main(["--root", str(root), "--output", str(out)]) == 0
    assert "Against the article" not in out.read_text()


TIME_TABLE = r"""
\begin{table}
\begin{tabular}{|l||c|c|c|}
\hline
 & EvoDAG agr-rnd & EvoDAG fit-fit & LR \\ \hline \hline
d1&2.0e-01(2) &1.0e-01(1) &1.0e-03(3) \\ \hline
d2&4.0e-01(2) &-(23) &3.0e-03(3) \\ \hline  \hline
Average time per sample &0.3&0.1&0.002\\ \hline
Average rank &1.5&13.0&1.5\\ \hline
\end{tabular}
\label{%s}
\end{table}
"""


def test_the_time_of_the_article_is_compared(tmp_path):
    tex = "".join(TIME_TABLE % label for label in configs.PAPER_TIME_LABELS)
    # the three parts hold the same columns here, which is enough to merge
    values, _ = summary.paper_table(tex, configs.PAPER_TIME_LABELS)
    assert values["d1"] == {"agr-rnd": 0.2, "fit-fit": 0.1,
                            "LogisticRegression": 0.001}
    seconds = {"d1": {"agr-rnd": 0.1, "fit-fit": 0.3,
                      "LogisticRegression": 0.0005},
               "d2": {"agr-rnd": 0.5, "fit-fit": 0.1,
                      "LogisticRegression": None}}
    text = summary.paper_time_comparison(
        seconds, ["agr-rnd", "fit-fit", "LogisticRegression"], ["d1", "d2"],
        tex)
    assert "ours divided by the article's" in text
    assert "| d1 | 0.5 | 3 | 0.5 |" in text
    assert "| d2 | 1.2 | - | - |" in text  # 0.5 / 0.4 = 1.25 -> 1.2
    # a source without the tables: a sentence
    assert "could not be made" in summary.attempt(
        summary.paper_time_comparison, seconds, ["agr-rnd"], ["d1"], "")
    # in the whole summary
    root = make_root(tmp_path, names=("d1", "d2"))
    for dataset in ("d1", "d2"):
        for system in configs.TABLE_CLASSIFIERS:
            write_result(root, dataset, system, f1=0.7, seconds=0.2)
    (tmp_path / "paper.tex").write_text(tex)
    out = tmp_path / "out.md"
    assert summary.main(["--root", str(root), "--paper-tex",
                         str(tmp_path / "paper.tex"),
                         "--output", str(out)]) == 0
    full = out.read_text()
    assert "#### Training time against the article" in full
    section = full[full.index("#### Training time against the article"):]
    lines = section.splitlines()
    header = [x.strip() for x in [y for y in lines
                                  if y.startswith("| dataset")][0].split("|")]
    row = [x.strip() for x in [y for y in lines
                               if y.startswith("| d1 ")][0].split("|")]
    cells = dict(zip(header, row))
    assert cells["agr-rnd"] == "1"  # 0.2 / 0.2
    assert cells["fit-fit"] == "2"  # 0.2 / 0.1
    assert cells["rnd-rnd"] == "-"  # not in the article's file here


def test_the_readme_names_every_option_of_the_scripts():
    import re
    readme = (SCRIPTS / "README.md").read_text()
    for script in ("profile_fit.py", "parity.py"):  # the Profile section
        assert "`%s`" % script in readme, script
    for script in ("datasets.py", "run.py", "submit.py", "summary.py"):
        assert "`%s`" % script in readme, script
        out = subprocess.run([sys.executable, str(SCRIPTS / script),
                              "--help"], stdout=subprocess.PIPE,
                             check=True).stdout.decode()
        for option in set(re.findall(r"--[a-z][a-z-]*", out)):
            if option != "--help":
                assert option in readme, (script, option)


# -- the files of the repository -------------------------------------------

def test_no_absolute_path_in_the_scripts():
    for path in SCRIPTS.glob("*"):
        if path.suffix in (".py", ".md"):
            assert "/" + "home" + "/" not in path.read_text(), path
