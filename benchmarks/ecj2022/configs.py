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
"""The named configurations of the selection heuristics article.

A configuration is a dictionary of keyword arguments of the engine that
differ from the common ones (:data:`COMMON`). Its name is
``<parent>-<negative>[--<first>][-all]``: the parent selection (``fit``,
``rnd``, ``agr``, ``sim``, ``prs``, ``ads`` or ``nvs``), the negative
selection (``fit`` or ``rnd``), the selection of the first parent when it is
not the default ``rnd``, and ``-all`` when the heuristic is applied to every
function (the ``*`` of the article).

>>> CONFIGS["agr-fit--fit"]["selection"], CONFIGS["agr-fit--fit"]["first_selection"]
('agreement', 'fitness')
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent

# keyword arguments of every configuration
COMMON = dict(fitness_function="macro-F1", random_generations=0,
              time_limit=None)
# replace the population, the early stopping and the members in --quick mode
QUICK = dict(popsize=50, early_stopping_rounds=50)
QUICK_MEMBERS = 3
MEMBERS = 30

PARENT = {"fit": "fitness", "rnd": "random", "agr": "agreement",
          "sim": "cosine", "prs": "pearson", "ads": "angle",
          "nvs": "novelty"}
FIRST = {"rnd": "random", "fit": "fitness"}

# (name, names in the article, parent, negative, first, functions); None is
# a dash of the table: the keyword is not given
_ROWS = [
    ("fit-fit", "fit-fit", "fit", "fit", None, None),
    ("rnd-rnd", "rnd-rnd", "rnd", "rnd", None, None),
    ("agr-rnd", "agr-rnd, acc-rnd, agr-rnd--rnd", "agr", "rnd", "rnd",
     "tailored"),
    ("agr-fit", "agr-fit, agr-fit--rnd", "agr", "fit", "rnd", "tailored"),
    ("sim-rnd", "sim-rnd, sim-rnd--rnd", "sim", "rnd", "rnd", "tailored"),
    ("sim-fit", "sim-fit, sim-fit--rnd", "sim", "fit", "rnd", "tailored"),
    ("prs-rnd", "prs-rnd, prs-rnd--rnd", "prs", "rnd", "rnd", "tailored"),
    ("prs-fit", "prs-fit, prs-fit--rnd", "prs", "fit", "rnd", "tailored"),
    ("agr-rnd-all", "agr-rnd*", "agr", "rnd", "rnd", "all"),
    ("agr-fit-all", "agr-fit*", "agr", "fit", "rnd", "all"),
    ("sim-rnd-all", "sim-rnd*", "sim", "rnd", "rnd", "all"),
    ("sim-fit-all", "sim-fit*", "sim", "fit", "rnd", "all"),
    ("prs-rnd-all", "prs-rnd*", "prs", "rnd", "rnd", "all"),
    ("prs-fit-all", "prs-fit*", "prs", "fit", "rnd", "all"),
    ("agr-fit--fit", "agr-fit--fit", "agr", "fit", "fit", "tailored"),
    ("agr-rnd--fit", "agr-rnd--fit", "agr", "rnd", "fit", "tailored"),
    ("ads-rnd", "ads-rnd--rnd, ads-rnd", "ads", "rnd", "rnd", "tailored"),
    ("ads-fit--fit", "ads-fit--fit", "ads", "fit", "fit", "tailored"),
    ("ads-rnd-all", "ads-rnd--rnd*", "ads", "rnd", "rnd", "all"),
    ("ads-fit--fit-all", "ads-fit--fit*", "ads", "fit", "fit", "all"),
    ("nvs-rnd", "nvs-rnd", "nvs", "rnd", None, "tailored"),
]


def _config(parent, negative, first, functions):
    kwargs = {"selection": PARENT[parent],
              "negative_selection": negative == "fit"}
    if first is not None:
        kwargs["first_selection"] = FIRST[first]
    if functions is not None:
        kwargs["heuristic_functions"] = functions
    return kwargs


CONFIGS = {name: _config(parent, negative, first, functions)
           for name, _, parent, negative, first, functions in _ROWS}

# column headers the article uses for each configuration
ARTICLE_NAMES = {name: [x.strip() for x in article.split(",")]
                 for name, article, *_ in _ROWS}

TABLE_HEURISTICS = [
    "agr-rnd", "agr-fit", "rnd-rnd", "sim-fit", "prs-fit-all", "prs-rnd",
    "prs-rnd-all", "prs-fit", "sim-fit-all", "sim-rnd", "sim-rnd-all",
    "fit-fit", "agr-rnd-all", "agr-fit-all"]
TABLE_SCHEMES = [
    "agr-rnd", "agr-fit--fit", "agr-fit", "rnd-rnd", "sim-fit",
    "agr-rnd--fit", "prs-fit", "prs-rnd", "ads-rnd", "sim-rnd",
    "ads-fit--fit", "fit-fit", "nvs-rnd", "ads-rnd-all", "ads-fit--fit-all"]
# the table of the comparison with the classifiers has every configuration
EVODAG_CLASSIFIERS = list(CONFIGS)

# labels of the tables of the article (several when a table is in parts)
PAPER_LABELS = {
    "heuristics": ["tab:res_apx_heuristics"],
    "schemes": ["tab:res_apx_selectionschemes_part1",
                "tab:res_apx_selectionschemes_part2"],
    "classifiers": ["tab:res_comparison_f1", "tab:res_apx_classifiers_part2"],
}
# the three tables of the article with the training time per example (the
# columns of the EvoDAG systems are named ``EvoDAG agr-rnd``...)
PAPER_TIME_LABELS = ["tab:res_apx_classifiers_time_part1",
                     "tab:res_apx_classifiers_time_part2",
                     "tab:res_apx_classifiers_time_part3"]

# the abbreviations of the article for the scikit-learn classifiers
SKLEARN_ARTICLE_NAMES = {
    "LogisticRegression": "LR", "SGD": "SGD", "Perceptron": "PER",
    "PassiveAggressive": "PA", "SVC": "SVC", "LinearSVC": "LSVC",
    "KNeighbors": "KN", "NearestCentroid": "NC", "GaussianNB": "NB",
    "BernoulliNB": "NBB", "DecisionTree": "DT", "RandomForest": "RF",
    "ExtraTrees": "ET", "AdaBoost": "AB", "GradientBoosting": "GB",
    "MLP": "MLP"}


def load_raetsch():
    "The module ``benchmarks/raetsch.py`` (loaded by path)"
    spec = importlib.util.spec_from_file_location(
        "raetsch_script", HERE.parent / "raetsch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# (name, module, class) of the sixteen classifiers of the article, the list
# of benchmarks/raetsch.py
SKLEARN = load_raetsch().SKLEARN
SKLEARN_NAMES = [name for name, _, _ in SKLEARN]
TABLE_CLASSIFIERS = EVODAG_CLASSIFIERS + SKLEARN_NAMES

TABLES = {"heuristics": TABLE_HEURISTICS, "schemes": TABLE_SCHEMES,
          "classifiers": TABLE_CLASSIFIERS}


def params_for(config, quick=False):
    """Keyword arguments of ``EvoDAG.model.EvoDAG`` (without the seed) for a
    configuration."""
    if config not in CONFIGS:
        raise ValueError("unknown configuration %r; choose among %s"
                         % (config, sorted(CONFIGS)))
    params = {**COMMON, **CONFIGS[config]}
    if quick:
        params.update(QUICK)
    return params


def n_members(quick=False):
    "Members of the ensemble"
    return QUICK_MEMBERS if quick else MEMBERS
