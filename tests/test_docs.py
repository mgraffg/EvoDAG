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
"""Tests that keep the website (``website/``) honest: its examples run, its
command line and parameter pages name everything the code has, and its
navigation and figures point at files that exist."""
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
PAGES = WEBSITE / "pages"


def python_blocks(path):
    "The fenced ``python`` blocks of a Quarto file"
    text = path.read_text(encoding="utf8")
    return re.findall(r"^```python[ \t]*\n(.*?)^```[ \t]*$", text,
                      flags=re.MULTILINE | re.DOTALL)


def run_page_examples(*paths):
    for path in paths:
        blocks = python_blocks(path)
        assert blocks, "%s has no python example" % path.name
        code = "\n".join(blocks)
        namespace = {"__name__": "docs_example"}
        exec(compile(code, str(path), "exec"), namespace)


def test_python_page_examples_run():
    run_page_examples(PAGES / "python.qmd", WEBSITE / "index.qmd")


def test_selection_page_examples_run():
    run_page_examples(PAGES / "selection.qmd")


def test_cli_page_names_every_option():
    from EvoDAG import command_line
    page = (PAGES / "cli.qmd").read_text(encoding="utf8")
    parsers = [command_line.CommandLineParams(),
               command_line.CommandLineTrain(),
               command_line.CommandLinePredict(),
               command_line.CommandLineUtils()]
    missing = set()
    for command in parsers:
        for action in command.parser._actions:
            names = list(action.option_strings)
            if not names:
                names = [action.dest]
            for name in names:
                if name not in page:
                    missing.add(name)
    assert not missing, "cli.qmd does not mention %s" % sorted(missing)


def test_parameters_page_names_every_parameter():
    page = (PAGES / "parameters.qmd").read_text(encoding="utf8")
    for fname in ("default_parameters.json", "default_parameters_r.json",
                  "parameter_values.json"):
        with open(ROOT / "EvoDAG" / "conf" / fname) as fpt:
            keys = json.load(fpt).keys()
        missing = [k for k in keys if "`%s`" % k not in page]
        assert not missing, "parameters.qmd does not mention %s" % missing


def navbar_targets():
    text = (WEBSITE / "_quarto.yml").read_text(encoding="utf8")
    hrefs = re.findall(r"^\s*-?\s*href:\s*(\S+)", text, flags=re.MULTILINE)
    return [h.strip("\"'") for h in hrefs]


def test_navbar_pages_exist():
    local = [h for h in navbar_targets()
             if not re.match(r"(https?:|mailto:|#)", h)]
    assert local
    for href in local:
        assert (WEBSITE / href).is_file(), href


def test_every_page_is_in_the_navbar():
    pages = {p.relative_to(WEBSITE).as_posix()
             for p in WEBSITE.rglob("*.qmd")
             if "_site" not in p.parts and ".quarto" not in p.parts}
    assert pages == set(navbar_targets()) & pages
    assert {"index.qmd", "pages/install.qmd", "pages/python.qmd",
            "pages/cli.qmd", "pages/how-it-works.qmd",
            "pages/selection.qmd", "pages/parameters.qmd",
            "pages/performance.qmd", "pages/citing.qmd"} <= pages


def test_page_links_and_figures_exist():
    for page in list(PAGES.glob("*.qmd")) + [WEBSITE / "index.qmd"]:
        text = page.read_text(encoding="utf8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"(https?:|mailto:|#)", target):
                continue
            path = target.split("#")[0]
            assert (page.parent / path).is_file(), (page.name, target)


def test_figure_script_builds_a_model():
    path = WEBSITE / "figures" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.small_model()
    assert model.size > 1
    for name in ("dag", "evolution", "ensemble"):
        assert (WEBSITE / "figures" / ("%s.svg" % name)).is_file()


def text_files():
    "The files that are in the repository: the tracked ones, or all of them"
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout
        files = [ROOT / f for f in out.decode("utf8").split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        skip = {".venv", "build", "dist", "__pycache__", "_site", ".quarto",
                ".pytest_cache"}
        files = [p for p in ROOT.rglob("*")
                 if p.is_file() and not skip & set(p.relative_to(ROOT).parts)
                 and not any(x.endswith(".egg-info")
                             for x in p.relative_to(ROOT).parts)]
    binary = {".npz", ".so", ".c", ".pyc", ".png", ".gz", ".pyd"}
    return [p for p in files if p.is_file() and p.suffix not in binary]


def test_no_local_paths():
    # written in two pieces so that this file does not contain it itself
    needle = "/" + "home" + "/"
    found = []
    for path in text_files():
        try:
            text = path.read_text(encoding="utf8")
        except UnicodeDecodeError:
            continue
        if needle in text:
            found.append(str(path.relative_to(ROOT)))
    assert not found, "absolute local paths in %s" % found
