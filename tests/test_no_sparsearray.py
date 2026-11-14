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

CODE = """
import sys
sys.modules["SparseArray"] = None  # `import SparseArray` now fails
import numpy as np
from EvoDAG import EvoDAG
from EvoDAG.model import EvoDAGE

rs = np.random.RandomState(0)
X = rs.randn(150, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
gp = EvoDAG(seed=0, classifier=True, popsize=50, early_stopping_rounds=50,
            time_limit=None, multiple_outputs=True).fit(X, y)
m = gp.model()
df = m.decision_function(X)
assert isinstance(df, list) and all(isinstance(x, np.ndarray) for x in df)
hy = m.predict(X)
assert isinstance(hy, np.ndarray) and hy.shape == (150,)
assert (hy == y).mean() > 0.5
reg = EvoDAG.init(classifier=False, seed=0, popsize=50,
                  early_stopping_rounds=50, time_limit=None)
reg.fit(X, X[:, 0] * 2 - X[:, 1])
assert isinstance(reg.model().predict(X), np.ndarray)
ens = EvoDAGE(n_estimators=2, n_jobs=1, popsize=30, early_stopping_rounds=30,
              time_limit=None).fit(X, y)
assert isinstance(ens.predict(X), (np.ndarray, list))
assert isinstance(ens.decision_function(X), np.ndarray)
"""


def test_no_sparsearray():
    "EvoDAG works with SparseArray made unimportable"
    r = subprocess.run([sys.executable, '-c', CODE], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]
