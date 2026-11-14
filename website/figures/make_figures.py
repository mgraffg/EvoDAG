"""Generate the figures of the website from the code.

Run it from anywhere::

    python website/figures/make_figures.py

It writes ``dag.svg``, ``evolution.svg`` and ``ensemble.svg`` next to this
file. The SVGs are committed, so building the site needs neither Graphviz nor
a fit; only regenerating them needs the ``dot`` program from Graphviz.

``dag.svg`` is produced by the package itself (``Model.graphviz``) from a
small model fitted on synthetic data with a fixed seed. The other two are
schematics written in the dot language below.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from EvoDAG.model import EvoDAG as SingleModel

HERE = os.path.dirname(os.path.abspath(__file__))

EVOLUTION = """digraph evolution {
  rankdir=LR;
  node [shape=box style="rounded,filled" fillcolor="#eef3fa" color="#1c5cab"
        fontname="Helvetica" fontsize=11];
  edge [color="#52514e" fontname="Helvetica" fontsize=10];
  pop [label="population\\n(popsize individuals)"];
  func [label="function f\\nchosen from the function set"];
  args [label="arguments of f\\nchosen from the population"];
  off [label="offspring\\nf(p1, ..., pk) times OLS weights"];
  fit [label="fitness on the training part\\nand on the validation part"];
  neg [label="negative selection\\nchooses who leaves"];
  stop [label="early stopping individual\\n(best on the validation part)" fillcolor="#e8f4ea" color="#2f7d3b"];
  pop -> func [label="draw"];
  func -> args [label="k arguments"];
  args -> off [label="evaluate"];
  off -> fit;
  fit -> neg [label="enters"];
  neg -> pop [label="replaces one"];
  fit -> stop [style=dashed label="keep the best"];
}
"""

ENSEMBLE = """digraph ensemble {
  rankdir=LR;
  node [shape=box style="rounded,filled" fillcolor="#eef3fa" color="#1c5cab"
        fontname="Helvetica" fontsize=11];
  edge [color="#52514e" fontname="Helvetica" fontsize=10];
  data [label="training set\\n(X, y)"];
  s0 [label="split and fit\\nseed = 0"];
  s1 [label="split and fit\\nseed = 1"];
  sd [label="..." shape=plaintext fillcolor=white style=""];
  sn [label="split and fit\\nseed = n_estimators - 1"];
  m0 [label="model 0" fillcolor="#e8f4ea" color="#2f7d3b"];
  m1 [label="model 1" fillcolor="#e8f4ea" color="#2f7d3b"];
  mn [label="model n_estimators - 1" fillcolor="#e8f4ea" color="#2f7d3b"];
  agg [label="aggregate\\nmajority vote of the classes\\nmedian of the decision functions"
       fillcolor="#fdf3e1" color="#b9770e"];
  out [label="prediction"];
  data -> s0; data -> s1; data -> sd [style=dotted]; data -> sn;
  s0 -> m0; s1 -> m1; sn -> mn;
  m0 -> agg; m1 -> agg; mn -> agg;
  sd -> agg [style=dotted];
  agg -> out;
}
"""


def dot_to_svg(source, output):
    "Render a dot-language string into the SVG file ``output``"
    dot = shutil.which("dot")
    if dot is None:
        sys.exit("Graphviz (the 'dot' program) is needed to regenerate the figures")
    result = subprocess.run([dot, "-Tsvg"], input=source.encode("utf8"),
                            stdout=subprocess.PIPE, check=True)
    with open(output, "wb") as fpt:
        fpt.write(result.stdout)


def small_model():
    "A small model fitted on synthetic data (fixed seeds, so it is repeatable)"
    rng = np.random.RandomState(1)
    X = rng.randn(200, 4)
    y = (np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 - 1
         + 0.3 * rng.randn(200) > 0).astype(int)
    fitted = SingleModel(popsize=20, early_stopping_rounds=10, seed=1,
                         all_inputs=False).fit(X, y)
    return fitted.model


def main():
    model = small_model()
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "dag.dot")
        model.graphviz(fname)
        with open(fname, encoding="utf8") as fpt:
            dot_to_svg(fpt.read(), os.path.join(HERE, "dag.svg"))
    dot_to_svg(EVOLUTION, os.path.join(HERE, "evolution.svg"))
    dot_to_svg(ENSEMBLE, os.path.join(HERE, "ensemble.svg"))
    print("model size %s, height %s" % (model.size, model.height))


if __name__ == "__main__":
    main()
