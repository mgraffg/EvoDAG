"""Tests of ``benchmarks/raetsch.py``. Nothing is downloaded: a tiny ``.mat``
with the structure of the real one is built in ``tmp_path``."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from scipy.io import savemat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "raetsch_script", ROOT / "benchmarks" / "raetsch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


raetsch = load_script()


def fake_mat(path, names=("heart", "thyroid"), n=60, splits=3):
    "A ``.mat`` with the structure of benchmarks.mat: x, t (+-1), 1-based train/test"
    rng = np.random.RandomState(0)
    content = {}
    for k, name in enumerate(names):
        X = rng.randn(n, 3)
        t = np.where(X[:, 0] + 0.3 * rng.randn(n) > 0, 1, -1)
        train = np.array([rng.permutation(n)[:40] + 1 for _ in range(splits)])
        test = np.array([np.setdiff1d(np.arange(1, n + 1), row)
                         for row in train])
        record = np.zeros((1, 1), dtype=[("x", "O"), ("t", "O"),
                                         ("train", "O"), ("test", "O")])
        record[0, 0]["x"] = X
        record[0, 0]["t"] = t.reshape(-1, 1)
        record[0, 0]["train"] = train
        record[0, 0]["test"] = test
        content[name] = record
    savemat(str(path), content)
    return path


def test_split_indices_are_one_based(tmp_path):
    mat = fake_mat(tmp_path / "fake.mat", names=("heart",))
    X, y, train, test = raetsch.load_dataset(mat, "heart")
    assert X.shape == (60, 3) and set(np.unique(y)) <= {-1, 1}
    assert train.shape == (3, 40) and test.shape == (3, 20)
    assert train.min() >= 0 and train.max() <= 59
    # train and test of a split partition the examples
    for tr, ts in zip(train, test):
        assert sorted(np.concatenate([tr, ts]).tolist()) == list(range(60))
    from scipy.io import loadmat
    raw = loadmat(str(mat))["heart"][0, 0]["train"]
    np.testing.assert_array_equal(train, raw - 1)
    assert raetsch.parse_splits("0-1", 3) == [0, 1]
    assert raetsch.parse_splits("all", 3) == [0, 1, 2]
    assert raetsch.parse_splits("0,2", 3) == [0, 2]
    with pytest.raises(ValueError):
        raetsch.parse_splits("0-3", 3)


def test_ber():
    from EvoDAG.utils import BER
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2])
    yh = np.array([0, 0, 0, 1, 1, 0, 2, 1, 0])
    assert raetsch.ber(y, yh) == pytest.approx(float(BER(y, yh)))
    y = np.array([-1, -1, 1, 1, 1])
    assert raetsch.ber(y, np.array([-1, 1, 1, 1, -1])) == pytest.approx(
        (50 + 100 / 3.) / 2)


def test_run_and_resume(tmp_path):
    mat = fake_mat(tmp_path / "fake.mat")
    output = tmp_path / "results.jsonl"
    kw = dict(datasets=["heart", "thyroid"], splits="0-1",
              algorithms=["GaussianNB", "evodag"], quick=True,
              log=lambda *a: None)
    assert raetsch.run(mat, output, **kw) == 2 * 2 * 2
    lines = raetsch.read_results(output)
    assert {(x["dataset"], x["split"], x["algorithm"]) for x in lines} == {
        (d, s, a) for d in ("heart", "thyroid") for s in (0, 1)
        for a in ("GaussianNB", "evodag")}
    for x in lines:
        assert x["quick"] is True and 0 <= x["ber"] <= 100
        assert x["fit_seconds"] >= 0 and x["predict_seconds"] >= 0
        assert {"evodag", "scikit-learn", "numpy", "python",
                "machine"} <= set(x["environment"])
    # an interrupted run (here: the same call again) computes nothing again
    assert raetsch.run(mat, output, **kw) == 0
    assert len(raetsch.read_results(output)) == 8
    # more splits only add the missing lines
    kw["splits"] = "0-2"
    assert raetsch.run(mat, output, **kw) == 2 * 2
    # a cut last line is ignored
    with open(output, "a") as fpt:
        fpt.write('{"dataset": "heart", "spl')
    assert len(raetsch.read_results(output)) == 12


def test_evodag_algorithm_uses_the_defaults():
    from EvoDAG import EnsembleClassifier, EvoDAGClassifier
    m = raetsch.make_algorithm("evodag", n_jobs=3)
    assert isinstance(m, EnsembleClassifier)
    assert m.n_estimators == 30 and m.random_state == 0 and m.n_jobs == 3
    assert m.estimator.get_params() == EvoDAGClassifier().get_params()
    q = raetsch.make_algorithm("evodag", quick=True)
    assert q.n_estimators == 3 and q.estimator.popsize == 50
    assert len(raetsch.SKLEARN) == 16


def write_lines(path, rows):
    with open(path, "w") as fpt:
        for dataset, split, algorithm, value, quick in rows:
            fpt.write(json.dumps({"dataset": dataset, "split": split,
                                  "algorithm": algorithm, "ber": value,
                                  "quick": quick}) + "\n")


def test_summary_table(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = []
    # banana: evodag 10, 20 -> mean 15; svc 12, 14 -> 13; nb 30, 30 -> 30
    # heart: evodag 20; svc 20; nb 10 (one split only)
    for split, e, s, n in ((0, 10, 12, 30), (1, 20, 14, 30)):
        rows += [("banana", split, "evodag", e, False),
                 ("banana", split, "SVC", s, False),
                 ("banana", split, "GaussianNB", n, False)]
    rows += [("heart", 0, "evodag", 20, False), ("heart", 0, "SVC", 20, False),
             ("heart", 0, "GaussianNB", 10, False),
             ("heart", 1, "evodag", 0, True)]  # a quick line: ignored
    write_lines(path, rows)
    text = raetsch.summary(path)
    assert "Splits per problem: banana 2, heart 1." in text
    assert "--quick" not in text
    table = {line.split("|")[1]: line.split("|")[2:-1]
             for line in text.splitlines() if line.startswith("|")}
    # banana ranks: SVC 1, evodag 2, nb 3; heart: nb 1, evodag/svc 2.5
    assert table["SVC"] == ["1.8", "**13.0**", "20.0"]
    assert table["EvoDAG"] == ["2.2", "15.0", "20.0"]
    assert table["GaussianNB"] == ["2.0", "30.0", "**10.0**"]
    assert table["EvoDAG (published)"][0] == "2.4"
    assert table["EvoDAG (published)"][1:] == ["12.0", "16.1"]
    # best average rank first (after the published row)
    names = [line.split("|")[1] for line in text.splitlines()
             if line.startswith("|")][3:]
    assert names == ["SVC", "GaussianNB", "EvoDAG"]
    # with the quick lines the heart split 1 of evodag counts (mean 10, which
    # ties with GaussianNB for the best); splits are counted per algorithm
    # and the smallest number is reported
    with_quick = raetsch.summary(path, include_quick=True)
    assert "Splits per problem: banana 2, heart 1." in with_quick
    assert "`--quick`" in with_quick
    table = {line.split("|")[1]: line.split("|")[2:-1]
             for line in with_quick.splitlines() if line.startswith("|")}
    assert table["EvoDAG"][2] == "**10.0**"
    # only quick lines: nothing without the flag
    write_lines(path, [("heart", 0, "evodag", 1.0, True)])
    assert raetsch.summary(path).startswith("No results")


def test_average_ranks_share_ties():
    ranks = raetsch.average_ranks({"d": {"a": 1.0, "b": 1.0, "c": 3.0}})
    assert ranks == {"a": 1.5, "b": 1.5, "c": 3.0}


def test_hash_mismatch_is_refused(tmp_path):
    bad = tmp_path / "benchmarks.mat"
    bad.write_bytes(b"not the data")
    with pytest.raises(ValueError, match="sha256"):
        raetsch.ensure_data(path=bad, url="http://invalid.invalid/none")
    assert bad.exists()  # it is not deleted, nor replaced
    good = tmp_path / "good.mat"
    good.write_bytes(b"abc")
    digest = raetsch.sha256_of(good)
    assert raetsch.ensure_data(path=good, sha256=digest) == good


def test_main_summary_and_missing_classifier(tmp_path, capsys, monkeypatch):
    mat = fake_mat(tmp_path / "fake.mat", names=("heart",))
    output = tmp_path / "out.jsonl"
    # a classifier that the installed scikit-learn lacks is skipped and noted
    monkeypatch.setattr(raetsch, "SKLEARN", raetsch.SKLEARN + [
        ("Gone", "sklearn.neighbors", "NoSuchClassifier")])
    monkeypatch.setattr(raetsch, "ALGORITHMS", raetsch.ALGORITHMS + ["Gone"])
    code = raetsch.main(["--data", str(mat), "--datasets", "heart",
                         "--splits", "0", "--algorithms", "GaussianNB",
                         "Gone", "--quick", "--output", str(output)])
    assert code == 0
    assert "Gone does not exist" in capsys.readouterr().out
    assert any(x.get("missing") for x in raetsch.read_results(output))
    assert raetsch.main(["--summary", "--include-quick",
                         "--output", str(output)]) == 0
    out = capsys.readouterr().out
    assert "Not available" in out and "Gone" in out and "GaussianNB" in out
