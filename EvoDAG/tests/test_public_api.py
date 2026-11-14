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
import subprocess
import sys

import numpy as np
import pytest

import EvoDAG
import EvoDAG.base
import EvoDAG.model


def test_user_facing_classes_are_the_model_module_ones():
    from EvoDAG import EvoDAGE, Model, Ensemble
    assert EvoDAGE is EvoDAG.model.EvoDAGE
    assert Model is EvoDAG.model.Model
    assert Ensemble is EvoDAG.model.Ensemble


def test_engine_names_unchanged():
    assert EvoDAG.EvoDAG is EvoDAG.base.EvoDAG
    assert EvoDAG.RGP is EvoDAG.RootGP is EvoDAG.EvoDAG


def test_model_module_imports_keep_working():
    from EvoDAG.model import EvoDAG as model_class, EvoDAGE
    assert model_class is not EvoDAG.EvoDAG
    assert issubclass(model_class, EvoDAGE)


def test_all():
    assert sorted(EvoDAG.__all__) == sorted(
        ["EvoDAG", "RGP", "RootGP", "EvoDAGE", "Model", "Ensemble"])
    for name in EvoDAG.__all__:
        assert hasattr(EvoDAG, name)
    namespace = {}
    exec("from EvoDAG import *", namespace)
    for name in EvoDAG.__all__:
        assert name in namespace


@pytest.mark.parametrize("first", ["EvoDAG", "EvoDAG.model", "EvoDAG.base"])
def test_fresh_interpreter_import(first):
    code = "import %s; import EvoDAG; from EvoDAG import EvoDAGE, Model, Ensemble" % first
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_single_model_wrapper_reaches_the_engine():
    m = EvoDAG.model.EvoDAG(classifier=True, popsize=20, early_stopping_rounds=20,
                            time_limit=None)
    assert isinstance(m._m, EvoDAG.EvoDAG)


def test_ensemble_end_to_end():
    from EvoDAG import EvoDAGE
    rs = np.random.RandomState(0)
    X = rs.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = EvoDAGE(n_estimators=2, n_jobs=1, popsize=50,
                early_stopping_rounds=50).fit(X, y)
    hy = m.predict(X)
    assert hy.shape == (100,)
    assert set(np.unique(hy)) <= {0, 1}
