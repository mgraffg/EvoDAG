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
"""The default selection: agreement for the parents and random negative
selection in classification (the best-ranked configuration of the selection
paper), the ordinary selections in regression and in the engine's own keyword
defaults. Added by review fix-1 of the run that wrote the website."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from EvoDAG import EvoDAG, RGP
from EvoDAG.model import EvoDAG as SingleModel, EvoDAGE

CONF = Path(__file__).resolve().parents[1] / "EvoDAG" / "conf"


def load(name):
    with open(CONF / name) as fpt:
        return json.load(fpt)


def selection(engine):
    return engine._orthogonal_selection, engine._negative_selection


def problem(n=60):
    rng = np.random.RandomState(0)
    X = rng.randn(n, 3)
    return X, (X[:, 0] > 0).astype(int)


@pytest.fixture
def constructed(monkeypatch):
    "Selection flags of every engine that is built, in order"
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
    assert classification["orthogonal_selection"] is True
    assert classification["negative_selection"] is False
    regression = load("default_parameters_r.json")
    assert "orthogonal_selection" not in regression
    assert "negative_selection" not in regression
    space = load("parameter_values.json")
    assert space["orthogonal_selection"] == [True]
    assert space["negative_selection"] == [False]


def test_fitness_function_stays_ber():
    assert load("default_parameters.json")["fitness_function"] == "BER"


def test_init_classifier_and_regressor():
    assert selection(RGP.init()) == (True, False)
    assert selection(RGP.init(classifier=False)) == (False, True)


def test_single_model_classifier_and_regressor():
    assert selection(SingleModel()._m) == (True, False)
    assert selection(SingleModel(classifier=False)._m) == (False, True)


def test_constructor_defaults_are_unchanged():
    assert selection(EvoDAG()) == (False, True)


def test_keywords_override_the_file():
    engine = RGP.init(orthogonal_selection=False, negative_selection=True)
    assert selection(engine) == (False, True)
    engine = RGP.init(classifier=False, orthogonal_selection=True,
                      negative_selection=False)
    assert selection(engine) == (True, False)


def test_a_parameters_file_replaces_the_defaults(tmp_path):
    fname = tmp_path / "params.json"
    fname.write_text(json.dumps({"popsize": 20}))
    assert selection(RGP.init(params_fname=str(fname))) == (False, True)


def test_ensemble_members_use_the_defaults(constructed):
    X, y = problem()
    EvoDAGE(n_estimators=2, popsize=20, early_stopping_rounds=20).fit(X, y)
    assert constructed == [(True, False)] * 2
    del constructed[:]
    EvoDAGE(n_estimators=2, popsize=20, early_stopping_rounds=20,
            orthogonal_selection=False, negative_selection=True).fit(X, y)
    assert constructed == [(False, True)] * 2


def test_ensemble_regressor_is_unaffected(constructed):
    X, y = problem()
    EvoDAGE(n_estimators=2, classifier=False, popsize=20,
            early_stopping_rounds=20).fit(X, X[:, 0] + X[:, 1])
    assert constructed == [(False, True)] * 2


def write_csv(path, X, y):
    with open(path, "w") as fpt:
        for x, v in zip(X, y):
            fpt.write(",".join(map(str, x.tolist() + [v])) + "\n")


@pytest.mark.parametrize("task, expected", [("-C", (True, False)),
                                            ("-R", (False, True))])
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
        assert ("orthogonal_selection" in configuration) is present
        assert ("negative_selection" in configuration) is present
        if present:
            assert configuration["orthogonal_selection"] is True
            assert configuration["negative_selection"] is False
