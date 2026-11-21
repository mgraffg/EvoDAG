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
"""The defaults of classification are the configuration ``agr-rnd`` of the
selection paper: agreement for the parents with a random reference, applied
to the functions it was designed for, a random choice of the individual that
leaves and macro-F1 as fitness. Regression and the engine's own keyword
defaults keep the tournaments on fitness."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from EvoDAG import EvoDAG, RGP
from EvoDAG.model import EvoDAG as SingleModel, EvoDAGE

CONF = Path(__file__).resolve().parents[1] / "EvoDAG" / "conf"
# the keys of the classification file that make it ``agr-rnd``
AGR_RND = dict(selection="agreement", first_selection="random",
               heuristic_functions="tailored", negative_selection=False,
               fitness_function="macro-F1", random_generations=0)
SELECTION_KEYS = ("selection", "first_selection", "heuristic_functions",
                  "orthogonal_selection", "negative_selection")
# (selection, negative_selection) of an engine
PAPER = ("agreement", False)
CLASSIC = (None, True)


def load(name):
    with open(CONF / name) as fpt:
        return json.load(fpt)


def selection(engine):
    return engine._selection, engine._negative_selection


def problem(n=60):
    rng = np.random.RandomState(0)
    X = rng.randn(n, 3)
    return X, (X[:, 0] > 0).astype(int)


@pytest.fixture
def constructed(monkeypatch):
    "Selection of every engine that is built, in order"
    seen = []
    original = EvoDAG.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # the helpers that only read the function set build an engine with no
        # arguments at all; the engines that are fitted are built by ``init``
        if kwargs:
            seen.append(selection(self))
    monkeypatch.setattr(EvoDAG, "__init__", record)
    return seen


def test_default_files():
    classification = load("default_parameters.json")
    for key, value in AGR_RND.items():
        assert classification[key] == value, key
    assert "orthogonal_selection" not in classification
    regression = load("default_parameters_r.json")
    for key in SELECTION_KEYS:
        assert key not in regression, key
    assert regression["fitness_function"] == "ER"
    space = load("parameter_values.json")
    for key in ("selection", "first_selection", "heuristic_functions",
                "negative_selection", "fitness_function"):
        assert space[key] == [AGR_RND[key]], key
    assert "orthogonal_selection" not in space


def test_init_reads_every_key_of_agr_rnd():
    engine = RGP.init()
    assert engine._selection == "agreement"
    assert engine._first_selection == "random"
    assert engine._heuristic_functions == "tailored"
    assert engine._negative_selection is False
    assert engine._fitness_function == "macro-F1"
    assert engine._random_generations == 0
    assert engine._orthogonal_selection is False


def test_the_defaults_fit_like_agr_rnd_written_out():
    X, y = problem(120)
    kw = dict(popsize=30, early_stopping_rounds=30, time_limit=None, seed=0)
    a = SingleModel(**kw).fit(X, y).decision_function(X)
    b = SingleModel(**kw, **AGR_RND).fit(X, y).decision_function(X)
    assert np.array_equal(np.array(a), np.array(b))


def test_init_classifier_and_regressor():
    assert selection(RGP.init()) == PAPER
    assert selection(RGP.init(classifier=False)) == CLASSIC


def test_single_model_classifier_and_regressor():
    assert selection(SingleModel()._m) == PAPER
    assert selection(SingleModel(classifier=False)._m) == CLASSIC


def test_constructor_defaults():
    # the constructor keeps the classical selection; its fitness is the
    # macro-F1 of the classification file
    engine = EvoDAG()
    assert selection(engine) == CLASSIC
    assert engine._orthogonal_selection is False
    assert engine._fitness_function == "macro-F1"
    assert engine._random_generations == 0


def test_keywords_override_the_file():
    engine = RGP.init(selection="fitness", negative_selection=True)
    assert selection(engine) == ("fitness", True)
    # ``selection=None`` gives the legacy booleans back
    engine = RGP.init(selection=None, orthogonal_selection=True)
    assert engine._selection is None and engine._orthogonal_selection is True
    engine = RGP.init(classifier=False, selection="agreement",
                      negative_selection=False)
    assert selection(engine) == PAPER


def test_a_parameters_file_replaces_the_defaults(tmp_path):
    fname = tmp_path / "params.json"
    fname.write_text(json.dumps({"popsize": 20}))
    engine = RGP.init(params_fname=str(fname))
    assert selection(engine) == CLASSIC
    assert engine._fitness_function == "macro-F1"


def test_ensemble_members_use_the_defaults(constructed):
    X, y = problem()
    EvoDAGE(n_estimators=2, popsize=20, early_stopping_rounds=20).fit(X, y)
    assert constructed == [PAPER] * 2
    del constructed[:]
    EvoDAGE(n_estimators=2, popsize=20, early_stopping_rounds=20,
            selection="fitness", negative_selection=True).fit(X, y)
    assert constructed == [("fitness", True)] * 2


def test_ensemble_regressor_is_unaffected(constructed):
    X, y = problem()
    EvoDAGE(n_estimators=2, classifier=False, popsize=20,
            early_stopping_rounds=20).fit(X, X[:, 0] + X[:, 1])
    assert constructed == [CLASSIC] * 2


def write_csv(path, X, y):
    with open(path, "w") as fpt:
        for x, v in zip(X, y):
            fpt.write(",".join(map(str, x.tolist() + [v])) + "\n")


@pytest.mark.parametrize("task, expected", [("-C", PAPER), ("-R", CLASSIC)])
def test_command_line_train(task, expected, tmp_path, monkeypatch,
                            constructed):
    from EvoDAG.command_line import train
    X, y = problem()
    fname = tmp_path / "train.csv"
    write_csv(fname, X, y)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "EvoDAG-train", task, "-n", "2", "-m", "model.gz", "--kw",
        '{"popsize": 20, "early_stopping_rounds": 20}', str(fname)])
    train()
    assert constructed == [expected] * 2


@pytest.mark.parametrize("task, present", [("-C", True), ("-R", False)])
def test_parameter_search_samples(task, present, tmp_path, monkeypatch):
    from EvoDAG.command_line import params
    X, y = problem()
    fname = tmp_path / "train.csv"
    write_csv(fname, X, y)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "EvoDAG-params", task, "-r", "2", "-p", "20", "-e", "20",
        "--only-paramsfiles", "-P", "configs", str(fname)])
    params()
    files = sorted((tmp_path / "configs").glob("*_params.json"))
    assert files
    for path in files:
        configuration = json.loads(path.read_text())
        assert "orthogonal_selection" not in configuration
        for key in ("selection", "first_selection", "heuristic_functions",
                    "negative_selection"):
            assert (key in configuration) is present, key
            if present:
                assert configuration[key] == AGR_RND[key], key
