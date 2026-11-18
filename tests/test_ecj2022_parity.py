"""Tests of ``benchmarks/ecj2022/parity.py`` and ``profile_fit.py``. The case
list is reduced to a few tiny cases so that the tests take seconds."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarks" / "ecj2022"
sys.path.insert(0, str(SCRIPTS))

import configs  # noqa: E402
import parity  # noqa: E402
import profile_fit  # noqa: E402
import run as runner  # noqa: E402

SMALL = dict(popsize=30, early_stopping_rounds=30, time_limit=None)


def tiny_cases():
    X, y, Xt = parity.classification_data(3)
    Xr, yr, Xtr = parity.regression_data()
    return [
        dict(name="agr-rnd", kind="engine", classifier=True, seed=0,
             params={**configs.params_for("agr-rnd"), **SMALL},
             data=(X[:100], y[:100], Xt[:20])),
        dict(name="default-regression", kind="engine", classifier=False,
             seed=1, params=dict(SMALL), data=(Xr[:100], yr[:100], Xtr[:20])),
        dict(name="ensemble", kind="ensemble", classifier=True, seed=0,
             params=dict(SMALL, n_estimators=2),
             data=(X[:100], y[:100], Xt[:20])),
    ]


def test_the_case_list_covers_what_the_issue_asks():
    cases = parity.make_cases()
    names = [c["name"] for c in cases]
    assert len(set(names)) == len(names)
    for config in configs.CONFIGS:
        for label in ("3c", "2c"):
            for seed in (0, 1):
                assert "%s-%s-s%d" % (config, label, seed) in names
    assert "default-regression-s0" in names
    assert any(n.startswith("ensemble-") for n in names)
    assert any(n.startswith("extremes-") for n in names)
    assert len(names) == 106
    for extra in ("regression-orthogonal", "generational-3c", "test-set-3c"):
        assert extra in names
    assert all(c["params"]["popsize"] == 200 for c in cases)


def test_record_then_check_is_identical(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(parity, "make_cases", tiny_cases)
    names = parity.record(tmp_path / "ref", log=lambda *a: None)
    assert names == ["agr-rnd", "default-regression", "ensemble"]
    with np.load(tmp_path / "ref" / "agr-rnd.npz") as stored:
        assert {"df_train", "df_test", "pred_train", "pred_test", "size",
                "height", "fitness_vs", "n_individuals", "history",
                "pickle_size"} <= set(stored.files)
    assert parity.check(tmp_path / "ref", log=lambda *a: None) == []
    assert parity.main(["check", "--reference", str(tmp_path / "ref")]) == 0
    assert "identical" in capsys.readouterr().out


def test_a_test_set_given_to_fit_is_recorded():
    case = dict(tiny_cases()[0], test_set=True)
    out = parity.run_case(case)
    assert out["hy_test"].shape == (3, 20)
    assert out["df_fit_test"].shape == (3, 20)
    assert "hy_test" not in parity.run_case(tiny_cases()[0])


def test_check_detects_a_change(tmp_path, monkeypatch, capsys):
    from EvoDAG.population import BasePopulation
    monkeypatch.setattr(parity, "make_cases", tiny_cases)
    parity.record(tmp_path / "ref", log=lambda *a: None)
    monkeypatch.setattr(BasePopulation, "random_selection",
                        lambda self, negative=False: 0)
    differences = parity.check(tmp_path / "ref", log=lambda *a: None)
    assert differences and differences[0][0] in ("agr-rnd",
                                                 "default-regression",
                                                 "ensemble")
    assert parity.main(["check", "--reference", str(tmp_path / "ref")]) == 1
    out = capsys.readouterr().out
    assert "cases differ" in out


def test_same_is_bit_exact():
    a = np.array([1.0, np.nan])
    assert parity.same(a, a.copy())
    assert not parity.same(a, np.array([1.0 + 2e-16, np.nan]))
    assert not parity.same(a, a.astype(np.float32))
    assert not parity.same(a, a[:1])


def write_dataset(root, name="toy"):
    rng = np.random.RandomState(0)
    X = rng.randn(90, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64) + (X[:, 2] > 1)
    (root / "data").mkdir(parents=True)
    np.savez(root / "data" / (name + ".npz"), X_train=X[:60], X_test=X[60:],
             y_train=y[:60].astype(np.int64), y_test=y[60:].astype(np.int64),
             classes=np.array(["a", "b", "c"]))


def test_profile_fit_prints_the_time_and_a_table(tmp_path, capsys):
    write_dataset(tmp_path)
    output = tmp_path / "out" / "toy.pstats"
    status = profile_fit.main(["--dataset", "toy", "--config", "agr-rnd",
                               "--root", str(tmp_path), "--top", "5",
                               "--max-seconds", "60",
                               "--output", str(output)])
    assert status == 0
    out = capsys.readouterr().out
    assert "60 examples, 4 variables, 3 classes" in out
    assert "fit time:" in out and "individuals created:" in out
    assert "Top 5 functions by own time" in out
    assert "Top 5 functions by cumulative time" in out
    assert "TRUNCATED" not in out
    assert output.exists()


def test_profile_fit_options(tmp_path):
    write_dataset(tmp_path)
    X, y, _, _ = runner.load_data(tmp_path, "toy")
    result = profile_fit.profile_member(X, y, "default", max_seconds=60)
    assert result["truncated"] is False and result["individuals"] > 0
    assert result["params"]["time_limit"] == 60
    with pytest.raises(SystemExit):
        profile_fit.main(["--dataset", "toy", "--config", "nothing",
                          "--root", str(tmp_path)])
