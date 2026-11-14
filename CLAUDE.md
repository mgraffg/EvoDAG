# CLAUDE.md

This file guides Claude Code (claude.ai/code) in this repository. It says what the tree **is**, in
the present tense; it describes conventions that are already followed and names nothing that is
not in the tree. `tests/test_docs.py` checks that the paths it names exist and that it quotes the
test command.

## What EvoDAG is

EvoDAG (Evolving Directed Acyclic Graph) is a steady-state semantic genetic programming system
for classification and regression, in Python with Cython kernels. Its genetic operator acts at
the root: an offspring is a function of existing individuals whose coefficients are fitted by
ordinary least squares against the target. It is described in two papers: *EvoDAG: A Semantic
Genetic Programming Python Library* (ROPEC 2016, the library) and *Selection Heuristics on
Semantic Genetic Programming for Classification Problems* (Evolutionary Computation, the parent
selection heuristics). The package is `EvoDAG`; the repository is the package (there is no src layout).

## Architecture

Modules in the order the data flows; each is documented in the developer reference (`docs/source/`).

- **`EvoDAG/base.py`, the engine.** `EvoDAG.base.EvoDAG` (exported as `EvoDAG.EvoDAG`, `RGP` and
  `RootGP`) runs one evolution: `fit`, then `model()`. `EvoDAG.init` reads the default parameters;
  the constructor's own defaults are different. **Two classes are called `EvoDAG`**: the engine
  above and `EvoDAG.model.EvoDAG`, a single model with the `fit`/`predict` interface; every
  docstring says which one it is.
- **`EvoDAG/population.py` and `EvoDAG/function_selection.pyx`.** The population (`SteadyState`,
  `Generational`), tournament and negative selection, the early stopping individual, and the
  tournament that chooses a function. Parent selection by agreement or dot product is in
  `EvoDAG/base.py` (`orthogonal_selection`, `orthogonal_dot_selection`); there is no Pearson heuristic.
  The classification defaults use agreement for the parents and random negative selection
  (`orthogonal_selection` true, `negative_selection` false); regression uses the tournaments on fitness.
- **`EvoDAG/node.py` and `EvoDAG/naive_bayes.py`.** An individual is a node: `Variable` or a
  `Function` subclass. A node stores the positions of its arguments in the history
  (`population.hist`), its `weight`, `height` and fitness. `hy` holds its values on the training
  examples and `hy_test` on the test set (`None` when none was given); with several outputs
  (always in classification, one per class) they are lists of arrays. Weights come from least
  squares (`EvoDAG/linalg_solve.pyx`) except for the Naive Bayes and Centroid nodes, which use the
  class labels.
- **`EvoDAG/bagging_fitness.py`.** The training/validation split (`tr_fraction`), the masks and
  the fitness on each part. A fit is deterministic for a fixed `seed`.
- **`EvoDAG/model.py`.** `Model` (the nodes the early stopping individual depends on, with their
  weights), `Ensemble` and `EvoDAGE` (members with seeds `seed`, `seed + 1`, ...; classes by
  majority vote, median for `decision_function`). `EvoDAGE.fit` returns an `Ensemble`.
- **`EvoDAG/estimators.py`.** `EvoDAGClassifier` and `EvoDAGRegressor`, scikit-learn estimators that
  fit **one** model (`model_`, a `Model`); `random_state` is the engine `seed`, and the predictions
  equal those of `EvoDAG.model.EvoDAG` with the same parameters and seed. scikit-learn is the optional
  extra `EvoDAG[sklearn]`: the module is imported lazily by `EvoDAG/__init__.py` (the two names are not
  in `__all__`) and nothing else in the package imports it. `EnsembleClassifier` and
  `EnsembleRegressor` in the same module clone any estimator `n_estimators` times with
  `random_state + i` and combine the members as `Ensemble` does (majority vote, median), so
  `EnsembleClassifier(EvoDAGClassifier())` gives the predictions of `EvoDAGE`.
- **`EvoDAG/utils.py` and `EvoDAG/command_line.py`.** The CSV/JSON readers, the error measures
  and `RandomParameterSearch`; the four console commands `EvoDAG-params`, `EvoDAG-train`,
  `EvoDAG-predict` and `EvoDAG-utils`. Models are stored as gzip pickles.
- **Cython kernels**: `EvoDAG/array_utils.pyx` (elementwise operations and reductions),
  `EvoDAG/cython_utils.pyx` (Naive Bayes, centroids, classification scores),
  `EvoDAG/linalg_solve.pyx` and `EvoDAG/function_selection.pyx`. They are built by `setup.py`;
  the generated `.c` and `.so` files are not committed.
- **Array type.** Vectors are one-dimensional NumPy `float64` arrays (there is no SparseArray
  dependency); a zero in a product stays zero, even against infinity (`array_utils.mul`), and
  the masks are read-only arrays. The density mechanism (`min_density`, `density_safe`) stays.
- **Parameters** are JSON in `EvoDAG/conf/`: `EvoDAG/conf/default_parameters.json` (classification),
  `EvoDAG/conf/default_parameters_r.json` (regression) and `EvoDAG/conf/parameter_values.json`
  (the space of the random search). A key named like a function (`Add`, `Mul`, ...) switches it on (boolean) or sets its
  arity (integer, 0 removes it). `EvoDAG/gp.py` is legacy code that the engine does not use.
  There are two layers of defaults: these files (read by `EvoDAGE`, `EvoDAG.model.EvoDAG`, `init`
  and the command line) and the keyword defaults of the engine constructor in `EvoDAG/base.py`
  (used only when the class is called directly); the Parameters page of the user site lists both.

## Layout

- `EvoDAG/`: the package. `tests/`: the pytest suite, with the golden files in `tests/golden/`
  and their recorder `tests/golden/record.py`.
- `docs/`: the Sphinx developer reference (`docs/source/`). `website/`: the Quarto user site
  (`website/pages/`, `website/figures/`).
- `benchmarks/`: `benchmarks/bench.py` and its recorded results. `scripts/`: the PyPI version check.
- `.github/workflows/`: `.github/workflows/test.yml`, `.github/workflows/website.yml`,
  `.github/workflows/publish-pypi.yml`. `.issue-pilot.json`: the
  test command and the base branch. `CITATION.cff`: the citation metadata (the Evolutionary Computation
  article is the preferred citation; `website/pages/citing.qmd` and the README keep the same order).

## Commands

Run them from the repository root. The suite needs the extensions built in place.

```bash
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python setup.py build_ext --inplace && .venv/bin/python -m pytest -q tests
.venv/bin/python -m pytest -q tests/test_docs.py::test_cli_page_names_every_option
.venv/bin/python -m sphinx -n -W --keep-going -b html docs/source docs/build/html
.venv/bin/python -m sphinx -b doctest docs/source docs/build/doctest
quarto render website
.venv/bin/python website/figures/make_figures.py
.venv/bin/python tests/golden/record.py --output /tmp/golden-check
.venv/bin/python benchmarks/bench.py --output /tmp/bench.json
.venv/bin/python scripts/check_package_versions.py
```

The second line is the test command of `.issue-pilot.json` (about 1 to 2 minutes). The strict
build and the doctest take seconds (the build downloads the intersphinx inventories, so it needs
network). `website/figures/make_figures.py` needs Graphviz (`dot`); `scripts/check_package_versions.py`
queries PyPI.
The benchmark takes about a minute; run it alone, and repeat it before trusting a difference
under 5 to 10 percent. Times only compare on the same machine. The development environment is a
`.venv` with the package installed in editable mode; do not reinstall packages in it casually.

## Rules

- **Reproducibility.** For a fixed `seed` results must be identical. Do not change the order in
  which random numbers are drawn, the defaults of the JSON files or the model format without a
  deliberate decision.
- **Golden files** (`tests/golden/`) are the reference outputs of the library; `tests/test_golden.py`
  compares against them (numbers within 1e-6, plus 1e-9 relative for huge values). Never edit a
  golden file, loosen the tolerance, skip a case or delete a test to get green. If behaviour has
  to change on purpose, say so and re-record with `tests/golden/record.py` in the commit that
  explains why. The groups that are a whole evolutionary run are reproducible only on a platform
  whose floating point arithmetic matches the recording machine (`tests/golden/platform.json`
  holds its fingerprint); elsewhere, CI included, those groups are skipped and the node, kernel,
  mask and reader groups still run.
- **Performance claims need the benchmark**: run `benchmarks/bench.py` and compare it with the
  recorded results (`benchmarks/results-numpy.json`, `benchmarks/results-optimized.json`).
- **Public names need a docstring and a verified example**, as `tests/test_docs.py` enforces:
  every public name has a docstring, every engine parameter is described, the Sphinx build is
  strict and its doctests run. Documentation changes must not change behaviour.
- **No new runtime dependency** without discussion: the only one is NumPy (`pyproject.toml`).
  scikit-learn is allowed only as the optional `sklearn` extra, imported by `EvoDAG/estimators.py` alone.
  Build requirements are Cython and NumPy.
- Keep the Cython extensions and the density mechanism. Scratch files go in `tmp_path`, never in
  the repository root; do not commit build output (the generated C files and shared libraries,
  the build and dist directories, the Sphinx build under docs and the rendered site under website).

## Documentation

Two halves of one GitHub Pages site, built by `.github/workflows/website.yml` (pull requests build
it, pushes to `master` deploy it; Pages has to be enabled once with the source "GitHub Actions").

- **User site** (Quarto, `website/`, served at the root): installing, the Python API, the command
  line, how the method works, the selection heuristics, the parameters, performance, citing. Code
  is not executed when it renders; `tests/test_docs.py` runs the Python blocks of
  `website/index.qmd`, `website/pages/python.qmd` and `website/pages/selection.qmd`, checks that
  `website/pages/cli.qmd` names every option of the four commands and that
  `website/pages/parameters.qmd` names every key of the JSON files, and that the navbar and
  figures exist.
- **Developer reference** (Sphinx, `docs/`, served under the path dev of the site): one `automodule` per module of
  the package, NumPy-style docstrings, doctests. The user site says how to use EvoDAG; the
  reference says what each name does.
- The figures in `website/figures/` are generated from the code and committed, so the build
  needs neither Graphviz nor a fit. No file of the papers is committed, and no tracked file
  contains an absolute local path (a test checks it).

## Releases

`version` is dynamic: it is the string literal `__version__` in `EvoDAG/__init__.py` (setuptools
reads it without importing the package; keep it a plain literal). `scripts/check_package_versions.py`,
run by the `check-package-versions` job of `.github/workflows/test.yml`, requires the version to be the next patch or
the next minor after the one on PyPI. A release is a tag `v<version>`: `.github/workflows/publish-pypi.yml` checks
that the tag matches the version, builds wheels with cibuildwheel (Python 3.10 to 3.13, Linux,
macOS, Windows; its test command is `pytest tests`) and the sdist, runs `twine check` and uploads
to PyPI with the repository secret `TWINE`. Do not tag, upload or change repository settings
unless asked. The conda-forge feedstock is a separate repository.

## Conventions

- English in code, comments, docstrings, documentation, commit messages and issues.
- Commit subjects are one imperative sentence, with the issue number at the end when there is one:
  `Add a Quarto user site for EvoDAG and a Pages workflow (issue #67)`; the body explains why.
- The base branch is `master` (`.issue-pilot.json`). Work happens on a branch and arrives through a
  pull request; do not push to `master`, force-push or rewrite history. Tests must be green before
  a commit.
- Tests live in `tests/`, mostly one file per module (for instance `tests/test_node.py` and `tests/test_model.py`), plus
  `tests/test_docs.py`,
  `tests/test_packaging.py`, `tests/test_public_api.py` and `tests/test_golden.py`.
- The package supports Python 3.10 or newer (`requires-python` in `pyproject.toml`).
