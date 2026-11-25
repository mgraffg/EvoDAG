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
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
PAGES = WEBSITE / "pages"
DOCS = ROOT / "docs" / "source"
PACKAGE = ROOT / "EvoDAG"


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
    # the pages use the scikit-learn classes, an optional dependency
    pytest.importorskip("sklearn")
    run_page_examples(PAGES / "python.qmd", WEBSITE / "index.qmd",
                      PAGES / "native-api.qmd")


def test_selection_and_parameters_page_examples_run():
    pytest.importorskip("sklearn")  # the Parameters page uses the estimators
    run_page_examples(PAGES / "selection.qmd", PAGES / "parameters.qmd")


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
            "pages/native-api.qmd", "pages/cli.qmd", "pages/how-it-works.qmd",
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


def package_modules():
    "Names of the modules of the package, Python and Cython, but ``__init__``"
    names = {p.stem for p in PACKAGE.glob("*.py")}
    names |= {p.stem for p in PACKAGE.glob("*.pyx")}
    names.discard("__init__")
    return sorted("EvoDAG.%s" % n for n in names)


def test_every_module_is_documented_exactly_once():
    modules = package_modules()
    assert "EvoDAG.base" in modules and "EvoDAG.cython_utils" in modules
    text = "\n".join(p.read_text(encoding="utf8")
                     for p in DOCS.rglob("*.rst"))
    documented = re.findall(r"^\.\.\s+automodule::\s+(\S+)\s*$", text,
                            flags=re.MULTILINE)
    for name in modules:
        assert documented.count(name) == 1, (name, documented.count(name))
    assert set(documented) <= set(modules)


def public_members(module):
    """The public classes and functions defined in ``module``, and the public
    methods and properties defined in those classes: ``(name, object)``"""
    seen = set()
    for name, obj in vars(module).items():
        if name.startswith("_") or id(obj) in seen:
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isclass(obj):
            seen.add(id(obj))
            yield name, obj
            for mname, member in vars(obj).items():
                if mname.startswith("_"):
                    continue
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if isinstance(member, property):
                    member = member.fget
                if callable(member):
                    yield "%s.%s" % (name, mname), member
        elif callable(obj):
            seen.add(id(obj))
            yield name, obj


def test_public_names_have_docstrings():
    missing = []
    total = 0
    for modname in package_modules():
        module = importlib.import_module(modname)
        for name, obj in public_members(module):
            total += 1
            if not (obj.__doc__ or "").strip():
                missing.append("%s.%s" % (modname, name))
    assert total > 300
    assert not missing, "no docstring: %s" % missing


def test_engine_parameters_are_documented():
    from EvoDAG import EvoDAG
    doc = inspect.getdoc(EvoDAG)
    parameters = inspect.signature(EvoDAG.__init__).parameters
    missing = []
    for name, parameter in parameters.items():
        if name == "self":
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            pattern = r"^\*\*%s$" % name
        else:
            pattern = r"^%s : " % re.escape(name)
        if not re.search(pattern, doc, flags=re.MULTILINE):
            missing.append(name)
    assert not missing, "not described in the docstring: %s" % missing


def sphinx_available():
    try:
        import sphinx  # noqa: F401
        import furo  # noqa: F401
    except ImportError:
        return False
    return True


needs_sphinx = pytest.mark.skipif(
    not sphinx_available(), reason="Sphinx or furo is not installed")


def run_sphinx(builder, output, extra=()):
    command = [sys.executable, "-m", "sphinx", *extra, "-b", builder,
               str(DOCS), str(output)]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)


@needs_sphinx
def test_sphinx_strict_build(tmp_path):
    result = run_sphinx("html", tmp_path / "html",
                        extra=("-n", "-W", "--keep-going"))
    assert result.returncode == 0, result.stdout[-4000:]


@needs_sphinx
def test_sphinx_doctest(tmp_path):
    result = run_sphinx("doctest", tmp_path / "doctest")
    assert result.returncode == 0, result.stdout[-4000:]
    assert " 0 failures in tests" in result.stdout


# Backticked tokens of CLAUDE.md that look like a path but are not one in the
# repository: the virtual environment is local and not tracked.
CLAUDE_NOT_PATHS = {".venv"}
PATH_SUFFIXES = (".py", ".pyx", ".pxd", ".json", ".yml", ".yaml", ".md",
                 ".toml", ".rst", ".qmd", ".scss", ".svg", ".txt", ".in",
                 ".npz")


def claude_md_paths():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf8")
    # the commands are in fenced blocks and are not paths of the tree
    text = re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)
    paths = []
    for token in re.findall(r"`([^`\n]+)`", text):
        if token in CLAUDE_NOT_PATHS or re.search(r"[\s*<>=]", token):
            continue
        if re.match(r"(https?:|/|-)", token):
            continue
        if "/" in token or token.endswith(PATH_SUFFIXES):
            paths.append(token)
    return paths


def test_claude_md_paths_exist():
    paths = claude_md_paths()
    assert len(paths) > 30
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, "CLAUDE.md names paths that do not exist: %s" % missing


def test_claude_md_names_the_test_command():
    with open(ROOT / ".issue-pilot.json") as fpt:
        command = json.load(fpt)["test_command"]
    assert command in (ROOT / "CLAUDE.md").read_text(encoding="utf8")


def test_parameters_page_gives_the_selection_defaults():
    "The page says, for the selection keys, the value that the files have"
    page = (PAGES / "parameters.qmd").read_text(encoding="utf8")
    conf = ROOT / "EvoDAG" / "conf"
    with open(conf / "default_parameters.json") as fpt:
        classification = json.load(fpt)
    with open(conf / "default_parameters_r.json") as fpt:
        regression = json.load(fpt)
    with open(conf / "parameter_values.json") as fpt:
        space = json.load(fpt)
    assert "orthogonal_selection" not in classification
    for key in ("selection", "first_selection", "heuristic_functions",
                "negative_selection"):
        assert key in classification and key not in regression
        value = classification[key]
        # a string is written `"agreement"` and a boolean true or false
        if isinstance(value, str):
            shown = '`"%s"`' % value
        else:
            shown = str(value).lower()
        # the row of the table of the parameters of the evolution
        row = re.search(r"^\| `%s` \| (\S+) " % key, page, flags=re.MULTILINE)
        assert row, key
        assert row.group(1) == shown, (key, row.group(1), shown)
        # and the row of the table of the search space
        row = re.search(r"^\| `%s` \| (\S+) \|$" % key, page,
                        flags=re.MULTILINE)
        assert row, key
        assert row.group(1) == shown, (key, row.group(1), shown)
        assert space[key] == [value], key


def test_readme_quick_start_uses_sklearn_classes():
    text = (ROOT / "README.md").read_text(encoding="utf8")
    assert "EnsembleClassifier(EvoDAGClassifier(" in text
    assert "EvoDAG[sklearn]" in text


def test_navbar_has_native_api_right_after_python():
    targets = navbar_targets()
    assert targets.index("pages/native-api.qmd") == \
        targets.index("pages/python.qmd") + 1


def test_python_page_points_to_the_native_api():
    "The first example of the pages is scikit-learn, the native classes live apart"
    for path in (WEBSITE / "index.qmd", PAGES / "python.qmd"):
        assert "EnsembleClassifier(tiny" in python_blocks(path)[0]
    page = (PAGES / "python.qmd").read_text(encoding="utf8")
    assert "native-api.qmd" in page
    code = "\n".join(python_blocks(PAGES / "python.qmd"))
    assert "EvoDAGE" not in code and "RGP" not in code
    assert "RGP" not in page


def test_selection_page_has_one_footnote_about_the_first_argument():
    page = (PAGES / "selection.qmd").read_text(encoding="utf8")
    footnotes = re.findall(r"^\[\^[^\]]+\]:.*$", page, flags=re.MULTILINE)
    assert len(footnotes) == 1
    assert "0.18" in footnotes[0]
    for title in ("The defaults against the article", "What the paper reports",
                  "What the released code has", "The legacy booleans"):
        assert title not in page, title
    # the Performance page points at the section the footnote belongs to
    assert "### The first argument is the reference" in page
    assert "selection.qmd#the-first-argument-is-the-reference" in \
        (PAGES / "performance.qmd").read_text(encoding="utf8")


def site_texts():
    "The files of the user site and the README"
    return [WEBSITE / "index.qmd", *sorted(PAGES.glob("*.qmd")),
            ROOT / "README.md"]


def test_site_does_not_speak_of_the_repository_or_of_earlier_versions():
    for path in site_texts():
        text = path.read_text(encoding="utf8")
        for number, line in enumerate(text.splitlines(), 1):
            where = "%s:%d" % (path.name, number)
            for word in ("benchmarks/", "tests/test_", "ecj2022", "raetsch"):
                assert word not in line, (where, word)
            # a table cell may be 0.18; prose may not, but for the footnote
            if line.startswith("|") or line.startswith("[^first-argument]:"):
                continue
            assert not re.search(r"\b0\.18\b", line), where
    page = (PAGES / "parameters.qmd").read_text(encoding="utf8")
    assert "What still differs" not in page
    page = (PAGES / "how-it-works.qmd").read_text(encoding="utf8")
    assert "Where the papers and the code differ" not in page


DATASETS = """ad adult agaricus-lepiota aps-failure banknote bank biodeg car
census-income cmc dota2 drug-consumption fertility IndianLiverPatient iris
krkopt letter-recognition magic04 ml-prove musk1 musk2 optdigits page-blocks
parkinsons pendigits segmentation sensorless tae wine yeast""".split()
CLASSIFIERS = """LogisticRegression SGD Perceptron PassiveAggressive SVC LinearSVC
KNeighbors NearestCentroid GaussianNB BernoulliNB DecisionTree RandomForest
ExtraTrees AdaBoost GradientBoosting MLP""".split()
EVODAG_SYSTEMS = """agr-rnd agr-fit rnd-rnd sim-fit prs-fit-all prs-rnd
prs-rnd-all prs-fit sim-fit-all sim-rnd sim-rnd-all fit-fit agr-rnd-all
agr-fit-all agr-fit--fit agr-rnd--fit ads-rnd ads-fit--fit ads-rnd-all
ads-fit--fit-all nvs-rnd""".split()


def test_performance_page_has_the_results_of_the_selection_experiments():
    page = (PAGES / "performance.qmd").read_text(encoding="utf8")
    assert len(DATASETS) == 30 and len(CLASSIFIERS) == 16
    for name in DATASETS:
        assert re.search(r"^\|%s\|" % re.escape(name), page,
                         flags=re.MULTILINE), name
    for name in CLASSIFIERS:
        assert "`%s`" % name in page, name
    # the rows of the tables of average ranks
    for row in (r"`agr-rnd`\|EvoDAG\|11\.0",
                r"`ExtraTrees`\|scikit-learn\|10\.2",
                r"`agr-rnd`\|5\.6", r"`agr-fit`\|4\.9", r"`agr-fit`\|4\.6",
                r"`fit-fit`\|7\.3"):
        assert re.search(r"^\|%s\|$" % row, page, flags=re.MULTILINE), row
    for pvalue in ("1.91e-05", "0.000228", "8.41e-68", "3.56e-195"):
        assert "p = " + pvalue in page, pvalue
    # the macro-F1 table and the table of time have the 37 systems
    assert len(EVODAG_SYSTEMS) == 21 and len(set(EVODAG_SYSTEMS)) == 21
    for name in EVODAG_SYSTEMS:
        rows = re.findall(r"^\|`%s`\|EvoDAG\|\d+\.\d\|$" % re.escape(name),
                          page, flags=re.MULTILINE)
        assert len(rows) >= 2, name
    for name in CLASSIFIERS:
        rows = re.findall(r"^\|`%s`\|scikit-learn\|\d+\.\d\|$" % name,
                          page, flags=re.MULTILINE)
        assert len(rows) >= 2, name
    # honest about where the default stands, and no result of old versions
    assert "fifth" in page and "fourth" in page
    assert "0.10." not in page and "PSMS" not in page


def test_the_package_and_the_reference_mention_version_0_18_once():
    hits = []
    for pattern in ("*.py", "*.pyx"):
        for path in sorted(PACKAGE.glob(pattern)):
            for number, line in enumerate(
                    path.read_text(encoding="utf8").splitlines(), 1):
                if "0.18" in line:
                    hits.append("%s:%d" % (path.name, number))
    assert len(hits) <= 1, hits
    for path in sorted(DOCS.glob("*.rst")):
        assert "0.18" not in path.read_text(encoding="utf8"), path.name


def test_the_experiments_readme_gives_only_the_current_results():
    text = (ROOT / "benchmarks" / "ecj2022" / "README.md").read_text(
        encoding="utf8")
    for phrase in ("results-before-fix", "before the correction",
                   "After the correction"):
        assert phrase not in text, phrase
    assert "`agr-rnd` 11.0" in text and "`ExtraTrees` 10.2" in text
    for pvalue in ("1.91e-05", "0.000228", "8.41e-68"):
        assert "Friedman p-value " + pvalue in text, pvalue
