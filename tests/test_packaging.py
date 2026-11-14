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
import importlib.util
try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "check_package_versions.py"
    spec = importlib.util.spec_from_file_location("check_package_versions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_is_dynamic():
    with open(ROOT / "pyproject.toml", "rb") as fpt:
        pyproject = tomllib.load(fpt)
    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"]
    version = pyproject["tool"]["setuptools"]["dynamic"]["version"]
    assert version["attr"] == "EvoDAG.__version__"


def test_local_version_matches_package():
    import EvoDAG
    script = load_script()
    assert script.local_version(ROOT) == EvoDAG.__version__


def test_next_version():
    from packaging.version import Version
    script = load_script()
    next_patch, next_minor = script.next_version(Version("0.17.2"))
    assert next_patch == Version("0.17.3")
    assert next_minor == Version("0.18.0")
