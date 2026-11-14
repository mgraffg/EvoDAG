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


def test_project_urls_point_to_the_site():
    with open(ROOT / "pyproject.toml", "rb") as fpt:
        urls = tomllib.load(fpt)["project"]["urls"]
    site = "https://mgraffg.github.io/EvoDAG/"
    assert urls["Homepage"] == site
    assert urls["Documentation"] == site
    assert urls["Developer reference"] == site + "dev/"
    assert urls["Source"] == "https://github.com/mgraffg/EvoDAG"
    assert urls["Issues"] == "https://github.com/mgraffg/EvoDAG/issues"


def test_citation_file_prefers_the_evco_article():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf8")
    assert "include CITATION.cff" in (ROOT / "MANIFEST.in").read_text()
    assert "\nversion:" not in text
    preferred, references = text.split("\nreferences:")
    assert "preferred-citation:" in preferred
    assert "10.1162/evco_a_00297" in preferred
    assert "Evolutionary Computation" in preferred
    ropec = references.index("10.1109/ROPEC.2016.7830633")
    neo = references.index("10.1007/978-3-319-44003-3_2")
    assert ropec < neo


def test_citation_order_in_readme_and_site():
    evco = "10.1162/evco_a_00297"
    ropec = "10.1109/ROPEC.2016.7830633"
    neo = "10.1007/978-3-319-44003-3_2"
    for name in ("README.md", "website/pages/citing.qmd"):
        text = (ROOT / name).read_text(encoding="utf8")
        assert "sanchez_selection_2022" in text
        assert text.index(evco) < text.index(ropec) < text.index(neo)
        assert "30(2)" in text and "253" in text


def test_readme_has_only_absolute_links_and_no_dead_ones():
    import re
    text = (ROOT / "README.md").read_text(encoding="utf8")
    assert "coveralls" not in text
    assert "ws.ingeotec.mx" not in text
    targets = re.findall(r"\]\(([^)]+)\)", text)
    assert targets
    relative = [t for t in targets if not t.startswith("https://")]
    assert not relative, relative
