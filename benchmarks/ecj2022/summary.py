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
"""Turn the results of the experiments into the tables of the article.

    python benchmarks/ecj2022/summary.py
    python benchmarks/ecj2022/summary.py --quick --median --output tables.md
    python benchmarks/ecj2022/summary.py --measure ber
    python benchmarks/ecj2022/summary.py --paper-tex path/to/paper.tex

Prints Markdown with, for each of the three tables of the article (the 14
combinations of selection heuristics, the 15 selection schemes and the
comparison with the classifiers of scikit-learn) and a fourth one with the
released defaults, the macro-F1 on the test set of every system on every
dataset with its rank in parentheses, the average rank (computed on the
datasets where every system has a result), and the p-value of the Friedman
test. ``--measure ber`` prints the balanced error rate (``1 - macro-recall``,
lower is better) instead. For the classifiers it also prints the training
time per training example and its ranks (1 is the fastest). The datasets of
the manifest that have no result are listed as incomplete. A result of an
ensemble with a number of members other than the expected one (30, or 3 with
``--quick``) counts as missing.

A rank is 1 plus the number of systems that are strictly better: systems with
the same value share the best rank of the tie and the following ranks are
skipped. A system without a result or with an error is shown as ``-``.

With ``--paper-tex`` the numbers of the tables of the article are read from
its LaTeX source and the average ranks and the macro-F1 differences (ours
minus the article's) are printed after each table, and, after the table of
the classifiers, the training time per example as the ratio of ours to the
article's. The table of the defaults has no counterpart and the BER is not in
the article. If the file or a table cannot be read, one sentence says so. The
file is never copied.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402
import run as runner  # noqa: E402

TITLES = {"heuristics": "Combinations of parent and negative selection",
          "schemes": "Selection schemes",
          "classifiers": "Comparison with classifiers",
          "defaults": "Released defaults"}
MEASURES = {"macro_f1": ("macro_f1", "macro_f1_median", True,
                         "Macro-F1 of the test set"),
            "ber": ("ber", "ber_median", False,
                    "Balanced error rate (BER) on the test set")}


# -- reading the results ---------------------------------------------------

def dataset_order(root):
    "Names of the datasets in the order of the manifest (if there is one)"
    path = Path(root) / "data" / "manifest.json"
    if path.exists():
        with open(path) as fpt:
            return [e["name"] for e in json.load(fpt)]
    return []


def read_result(root, dataset, system, quick=False):
    """The result dictionary of a system (a configuration or a scikit-learn
    name) on a dataset; ``None`` if there is none, ``{"error": ...}`` for an
    error."""
    if system in configs.CONFIGS:
        path = runner.unit_dir(root, dataset, system, quick) / "result.json"
    else:
        path = runner.unit_dir(root, dataset, "sklearn", quick) / (
            system + ".json")
    if not path.exists():
        return None
    with open(path) as fpt:
        result = json.load(fpt)
    # a run with --members N is not the finished unit
    if (system in configs.CONFIGS and "error" not in result
            and result.get("members") != configs.n_members(quick)):
        return None
    return result


def collect(root, datasets, systems, quick=False, median=False,
            measure="macro_f1"):
    """``(f1, time)``: dictionaries dataset -> system -> number (or ``None``
    for a missing result, an error or, for a result written before the
    measure existed, a missing key); ``f1`` holds the ``measure``."""
    key = MEASURES[measure][1 if median else 0]
    f1, seconds = {}, {}
    for dataset in datasets:
        f1[dataset], seconds[dataset] = {}, {}
        for system in systems:
            result = read_result(root, dataset, system, quick)
            usable = result is not None and "error" not in result
            f1[dataset][system] = result.get(key) if usable else None
            seconds[dataset][system] = (
                result.get("fit_seconds_per_sample") if usable else None)
    return f1, seconds


# -- ranks -----------------------------------------------------------------

def rank_row(values, higher_is_better=True):
    """Ranks of one dataset: ``{system: rank}`` for the systems that have a
    number. 1 plus the number of strictly better systems, so a tie shares the
    lowest (best) rank and the next ranks are skipped."""
    numbers = {s: v for s, v in values.items() if v is not None}
    ranks = {}
    for system, value in numbers.items():
        if higher_is_better:
            better = sum(1 for v in numbers.values() if v > value)
        else:
            better = sum(1 for v in numbers.values() if v < value)
        ranks[system] = 1 + better
    return ranks


def complete_datasets(table, systems):
    "Datasets where every system has a number"
    return [d for d, row in table.items()
            if all(row.get(s) is not None for s in systems)]


def average_ranks(table, systems, higher_is_better=True):
    """Average rank of every system over the complete datasets, and those
    datasets."""
    complete = complete_datasets(table, systems)
    if not complete:
        return {s: None for s in systems}, complete
    total = {s: 0.0 for s in systems}
    for dataset in complete:
        for s, r in rank_row({s: table[dataset][s] for s in systems},
                             higher_is_better).items():
            total[s] += r
    return {s: total[s] / len(complete) for s in systems}, complete


def friedman_pvalue(table, systems, datasets):
    "Text with the p-value of the Friedman test over the given datasets"
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return "scipy not installed"
    if len(systems) < 3 or len(datasets) < 2:
        return "not enough systems or datasets"
    columns = [[table[d][s] for d in datasets] for s in systems]
    try:
        _, p = friedmanchisquare(*columns)
    except ValueError as error:
        return "not computed (%s)" % error
    return "%.3g" % p


# -- Markdown --------------------------------------------------------------

def cell(value, rank, fmt):
    "A table cell: the number and its rank, or a dash"
    if value is None:
        return "-"
    return "%s (%d)" % (fmt % value, rank)


def markdown_table(header, rows):
    "A Markdown table from a header list and a list of rows"
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def ranked_table(title, table, systems, datasets, fmt, higher_is_better,
                 unit):
    """Markdown of one measure: the text, the table sorted by average rank,
    and the Friedman p-value. Returns ``(text, average_ranks)``."""
    averages, complete = average_ranks(table, systems, higher_is_better)
    position = {s: i for i, s in enumerate(systems)}
    ordered = sorted(systems, key=lambda s: (
        averages[s] is None, averages[s] if averages[s] is not None else 0,
        position[s]))
    incomplete = [d for d in datasets if d not in complete]
    lines = ["### %s" % title, "",
             "%s. The rank of the system on the dataset is in parentheses "
             "(1 is best). The average rank uses the %d datasets where "
             "every system has a result%s."
             % (unit, len(complete),
                "; incomplete: " + ", ".join(incomplete)
                if incomplete else ""), ""]
    rows = []
    for dataset in datasets:
        ranks = rank_row(table[dataset], higher_is_better)
        rows.append([dataset] + [cell(table[dataset][s], ranks.get(s), fmt)
                                 for s in ordered])
    rows.append(["**Average rank**"] + [
        "-" if averages[s] is None else "%.1f" % averages[s]
        for s in ordered])
    lines.append(markdown_table(["dataset"] + ordered, rows))
    lines += ["", "Friedman test p-value over the %d complete datasets: %s."
              % (len(complete), friedman_pvalue(table, systems, complete))]
    return "\n".join(lines), averages


# -- the article -----------------------------------------------------------

def article_header_map():
    "Header of the article -> name of our system"
    mapping = {}
    for name, headers in configs.ARTICLE_NAMES.items():
        for header in headers:
            mapping[header] = name
    for name, abbreviation in configs.SKLEARN_ARTICLE_NAMES.items():
        mapping[abbreviation] = name
    for name in configs.EVODAG_CLASSIFIERS:  # the tables of time
        mapping["EvoDAG " + name] = name
    return mapping


CELL = re.compile(r"^(-|[0-9.]+(?:[eE][-+]?[0-9]+)?)\((\d+)\)$")


def parse_paper_table(text, label):
    """Parse the ``tabular`` that precedes ``\\label{label}``. Returns
    ``(values, average_rank)``: ``values[dataset][header]`` is a float or
    ``None`` (a ``-`` cell) and ``average_rank[header]`` a float."""
    end = text.find("\\label{%s}" % label)
    if end < 0:
        raise ValueError("the label %s is not in the LaTeX source" % label)
    start = text.rfind("\\begin{tabular}", 0, end)
    if start < 0:
        raise ValueError("no tabular before the label %s" % label)
    body = text[start:end]
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", body)
    body = body.replace("\\end{tabular}", "").replace("\\hline", "")
    header, values, averages = None, {}, {}
    for row in body.split("\\\\"):
        row = row.strip()
        if not row:
            continue
        cells = [c.strip() for c in row.split("&")]
        if header is None:
            if cells[0] == "":
                header = cells[1:]
            continue
        name, rest = cells[0], cells[1:]
        if name == "Average rank":
            averages = {h: float(v) for h, v in zip(header, rest)}
        elif name.startswith("Average"):
            continue
        else:
            row_values = {}
            for h, c in zip(header, rest):
                match = CELL.match(c.replace(" ", ""))
                if match:
                    row_values[h] = (None if match.group(1) == "-"
                                     else float(match.group(1)))
            values[name] = row_values
    return values, averages


def paper_table(text, labels):
    """The article's numbers of a table, in our system names: ``(values,
    averages)`` with ``values[dataset][system]``; the parts of a table that
    is split are merged and unknown systems (TPOT...) are ignored."""
    mapping = article_header_map()
    values, averages = {}, {}
    for label in labels:
        part, part_avg = parse_paper_table(text, label)
        for dataset, row in part.items():
            for header, value in row.items():
                if header in mapping:
                    values.setdefault(dataset, {})[mapping[header]] = value
        for header, value in part_avg.items():
            if header in mapping:
                averages[mapping[header]] = value
    return values, averages


def paper_comparison(f1, averages, systems, datasets, text, labels):
    "Markdown with the average ranks and the macro-F1 differences"
    values, paper_avg = paper_table(text, labels)
    lines = ["#### Against the article", ""]
    rows = [[s, "-" if averages[s] is None else "%.1f" % averages[s],
             "%.1f" % paper_avg[s] if s in paper_avg else "-"]
            for s in systems]
    lines.append(markdown_table(["system", "our average rank",
                                 "article's average rank"], rows))
    lines += ["", "Macro-F1, ours minus the article's, for the cells that "
              "both have. The article's ranks come from all its systems "
              "(TPOT and auto-sklearn included) and ours from the systems "
              "of this table, so the average ranks are only roughly "
              "comparable.", ""]
    rows = []
    for dataset in datasets:
        row = [dataset]
        for s in systems:
            ours = f1[dataset].get(s)
            theirs = values.get(dataset, {}).get(s)
            row.append("-" if ours is None or theirs is None
                       else "%+.2f" % (ours - theirs))
        rows.append(row)
    lines.append(markdown_table(["dataset"] + list(systems), rows))
    return "\n".join(lines)


def paper_time_comparison(seconds, systems, datasets, text):
    """Markdown with the training time per example, ours divided by the
    article's, two significant digits, for the cells that both have."""
    values, _ = paper_table(text, configs.PAPER_TIME_LABELS)
    lines = ["#### Training time against the article", "",
             "Training time per training example, ours divided by the "
             "article's (below 1 is faster than the article, and the "
             "machines are not the same), for the cells that both have.",
             ""]
    rows = []
    for dataset in datasets:
        row = [dataset]
        for s in systems:
            ours = seconds[dataset].get(s)
            theirs = values.get(dataset, {}).get(s)
            row.append("-" if not ours or not theirs
                       else "%.2g" % (ours / theirs))
        rows.append(row)
    lines.append(markdown_table(["dataset"] + list(systems), rows))
    return "\n".join(lines)


def attempt(function, *args):
    """The text of a comparison with the article; if the article's tables do
    not parse, one sentence says so (a comparison never stops the summary)."""
    try:
        return function(*args)
    except (ValueError, KeyError, IndexError) as error:
        return ("The comparison with the article could not be made: %s."
                % error)


# -- command line ----------------------------------------------------------

def summarize(root, quick=False, median=False, paper_text=None,
              measure="macro_f1", paper_note=None):
    """The Markdown of the four tables. ``measure`` is ``macro_f1`` or
    ``ber``; ``paper_note`` is a sentence to print when the article's source
    could not be read."""
    order = dataset_order(root)
    results = runner.results_dir(root, quick)
    found = sorted(p.name for p in results.iterdir()
                   if p.is_dir()) if results.exists() else []
    if not found:
        return "No results in %s." % results
    # every dataset of the manifest, also those that were not started
    datasets = list(order)
    datasets += [d for d in found if d not in datasets]
    _, _, higher_is_better, unit = MEASURES[measure]
    out = ["# Selection heuristics experiments", "",
           "%s, EvoDAG ensembles of %d members with the %s."
           % (unit, configs.n_members(quick),
              "median decision function" if median else "majority vote")]
    if quick:
        out += ["", "These are `--quick` results: tiny parameters, they are "
                "not results of the article."]
    if paper_note:
        out += ["", paper_note]
    if paper_text is not None and measure != "macro_f1":
        out += ["", "The comparison with the article is skipped: the "
                "article reports macro-F1 only."]
    for name, systems in configs.TABLES.items():
        f1, seconds = collect(root, datasets, systems, quick, median,
                              measure)
        text, averages = ranked_table(
            TITLES[name], f1, systems, datasets, "%.2f", higher_is_better,
            unit)
        out += ["", text]
        if (paper_text is not None and measure == "macro_f1"
                and name in configs.PAPER_LABELS):
            out += ["", attempt(paper_comparison, f1, averages, systems,
                                datasets, paper_text,
                                configs.PAPER_LABELS[name])]
        if name == "classifiers":
            text, _ = ranked_table(
                "Training time per training example, comparison with "
                "classifiers", seconds, systems, datasets, "%.2e", False,
                "Seconds of training per training example (the sum over the "
                "members for EvoDAG); the rank 1 is the fastest")
            out += ["", text]
            if paper_text is not None:
                out += ["", attempt(paper_time_comparison, seconds, systems,
                                    datasets, paper_text)]
    return "\n".join(out) + "\n"


def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Tables of the selection heuristics experiments.")
    parser.add_argument("--quick", action="store_true",
                        help="read the results of --quick runs")
    parser.add_argument("--median", action="store_true",
                        help="use the median decision function instead of "
                             "the majority vote")
    parser.add_argument("--measure", choices=sorted(MEASURES),
                        default="macro_f1",
                        help="macro_f1 (default) or ber, the balanced error "
                             "rate (lower is better)")
    parser.add_argument("--paper-tex", metavar="PATH",
                        help="LaTeX source of the article, to compare with "
                             "its numbers (e.g. ../articulos/ECJ2022/paper.tex)")
    parser.add_argument("--output", help="write the Markdown to this file")
    parser.add_argument("--root", help="experiments directory (default: "
                        "$%s or <repository>/experiments)" % runner.ENV_ROOT)
    args = parser.parse_args(argv)
    root = runner.resolve_root(args.root)
    paper_text, note = None, None
    if args.paper_tex:
        try:
            paper_text = Path(args.paper_tex).read_text(encoding="utf8")
        except (OSError, UnicodeDecodeError) as error:
            note = ("The article's source %s could not be read (%s): no "
                    "comparison with the article." % (args.paper_tex,
                                                      type(error).__name__))
    text = summarize(root, args.quick, args.median, paper_text,
                     args.measure, note)
    if args.output:
        Path(args.output).write_text(text, encoding="utf8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
