[![Tests](https://github.com/mgraffg/EvoDAG/actions/workflows/test.yml/badge.svg)](https://github.com/mgraffg/EvoDAG/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/mgraffg/EvoDAG/badge.svg?branch=develop)](https://coveralls.io/github/mgraffg/EvoDAG?branch=develop)
[![PyPI version](https://badge.fury.io/py/evodag.svg)](https://badge.fury.io/py/evodag)
[![azure](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/evodag-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16226&branchName=main)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/evodag.svg)](https://anaconda.org/conda-forge/evodag) 
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/evodag.svg)](https://anaconda.org/conda-forge/evodag) 

# EvoDAG #

Evolving Directed Acyclic Graph (EvoDAG) is a steady-state Genetic Programming system
with tournament selection (for classification, the default parameters choose the
arguments of the sum and of the classifiers by agreement and the individual that
leaves at random; see the documentation). The main characteristic of EvoDAG is that
the genetic operation is performed at the root. EvoDAG was inspired
by the geometric semantic crossover proposed by 
[Alberto Moraglio](https://scholar.google.com.mx/citations?user=0y4XRI0AAAAJ&hl=en&oi=ao)
_et al._ and the implementation performed by
[Leonardo Vanneschi](https://scholar.google.com.mx/citations?user=uR5K07QAAAAJ&hl=en&oi=ao)
_et al_.

EvoDAG is described in the following conference paper
[EvoDAG: A semantic Genetic Programming Python library](http://ieeexplore.ieee.org/document/7830633/)
Mario Graff, Eric S. Tellez, Sabino Miranda-Jiménez, Hugo Jair Escalante.
2016 IEEE International Autumn Meeting on Power, Electronics and Computing (ROPEC)
pp 1-6. A pre-print version can be download from [here](http://ws.ingeotec.mx/~mgraffg/publications/pdf/ropec2016.pdf).

**Documentation:** the user guide is at <https://mgraffg.github.io/EvoDAG/>:
installation, using EvoDAG from Python and from the command line, how the
method works, the parameters, the performance results and how to cite it.

## Quick Start ##

There are two options to use EvoDAG, one is as a library and the other is using the command line interface.

### Using EvoDAG as library

Let us assume that `X` contains the inputs and `y` contains the
classes. Then in order to train an ensemble of 30 EvoDAG and predict
`X` one uses the following instructions:

```python
# Importing EvoDAG ensemble
from EvoDAG.model import EvoDAGE
# Importing iris dataset from sklearn
from sklearn.datasets import load_iris

# Reading data
data = load_iris()
X = data.data
y = data.target

#train the model
m = EvoDAGE(n_estimators=30, n_jobs=4).fit(X, y)

#predict X using the model
hy = m.predict(X)
```

`EvoDAGE`, `Model` and `Ensemble` are importable from the top-level package.
There are two classes called `EvoDAG`: `EvoDAG.EvoDAG` (also `RGP` and
`RootGP`) is the search engine, whereas `EvoDAG.model.EvoDAG` is a single
model with the `fit`/`predict` interface. The `EvoDAG.model.*` imports keep
working.

### Using EvoDAG from the command line

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

```bash   
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

#### Training EvoDAG

Let us assume, you do not want to optimise the EvoDAG parameters, so the
default parameters are used when flag `-P` is not present, i.e., 

```bash   
EvoDAG-train -C -m model.evodag -n 100 -u 4 iris.data 
```
`-C` flag indicates that it is a classification problem, and 
`-R` is for regression problems; there are different default parameters
for each type of problems.

The performance of EvoDAG without optimising the parameters is
presented in the [Performance](https://mgraffg.github.io/EvoDAG/pages/performance.html)
page of the documentation.

#### Predict 

Once the model is obtained, it is time to use it; given that `iris.data` 
was not split into a training and test set, let us assume 
that `iris.data` contains some unknown data. 
In order to predict `iris.data` one would do:

```bash   
EvoDAG-predict -m model.evodag -o iris.predicted iris.data
```

where `-o` indicates the file name used to store the predictions, `-m`
contains the model, and `iris.data` is the test set.

# Performance #

The benchmark results (EvoDAG against the classifiers of scikit-learn, PSMS,
auto-sklearn and an SVM) are in the
[Performance](https://mgraffg.github.io/EvoDAG/pages/performance.html) page
of the documentation.

## Development ##

The development guide is [`CLAUDE.md`](CLAUDE.md) (architecture, commands, rules and
conventions), and the developer reference, with every module documented, is at
<https://mgraffg.github.io/EvoDAG/dev/>.

## Citing EvoDAG ##

If you like EvoDAG, and it is used in a scientific publication, I would
appreciate citations to either the conference paper or the book chapter:

[EvoDAG: A semantic Genetic Programming Python library](http://ieeexplore.ieee.org/document/7830633/)
Mario Graff, Eric S. Tellez, Sabino Miranda-Jiménez, Hugo Jair Escalante.
2016 IEEE International Autumn Meeting on Power, Electronics and Computing (ROPEC)
pp 1-6.
```bibtex
@inproceedings{graff_evodag:_2016,
	title = {{EvoDAG}: {A} semantic {Genetic} {Programming} {Python} library},
	shorttitle = {{EvoDAG}},
	doi = {10.1109/ROPEC.2016.7830633},
	abstract = {Genetic Programming (GP) is an evolutionary algorithm that has received a lot of attention lately due to its success in solving hard real-world problems. Lately, there has been considerable interest in GP's community to develop semantic genetic operators, i.e., operators that work on the phenotype. In this contribution, we describe EvoDAG (Evolving Directed Acyclic Graph) which is a Python library that implements a steady-state semantic Genetic Programming with tournament selection using an extension of our previous crossover operators based on orthogonal projections in the phenotype space. To show the effectiveness of EvoDAG, it is compared against state-of-the-art classifiers on different benchmark problems, experimental results indicate that EvoDAG is very competitive.},
	booktitle = {2016 {IEEE} {International} {Autumn} {Meeting} on {Power}, {Electronics} and {Computing} ({ROPEC})},
	author = {Graff, M. and Tellez, E. S. and Miranda-Jiménez, S. and Escalante, H. J.},
	month = nov,
	year = {2016},
	keywords = {directed graphs, Electronic mail, EvoDAG, evolving directed acyclic graph, Genetic algorithms, GP community, Libraries, semantic genetic operators, semantic genetic programming Python library, Semantics, Sociology, Statistics, Steady-state, steady-state semantic genetic programming, Training},
	pages = {1--6}
}
```

[Semantic Genetic Programming for Sentiment Analysis](http://link.springer.com/chapter/10.1007/978-3-319-44003-3_2)
Mario Graff, Eric S. Tellez, Hugo Jair Escalante, Sabino
Miranda-Jiménez. NEO 2015
Volume 663 of the series Studies in Computational Intelligence pp 43-65.
```bibtex
@incollection{graff_semantic_2017,
	series = {Studies in {Computational} {Intelligence}},
	title = {Semantic {Genetic} {Programming} for {Sentiment} {Analysis}},
	copyright = {©2017 Springer International Publishing Switzerland},
	isbn = {9783319440026 9783319440033},
	url = {http://link.springer.com/chapter/10.1007/978-3-319-44003-3_2},
	abstract = {Sentiment analysis is one of the most important tasks in text mining. This field has a high impact for government and private companies to support major decision-making policies. Even though Genetic Programming (GP) has been widely used to solve real world problems, GP is seldom used to tackle this trendy problem. This contribution starts rectifying this research gap by proposing a novel GP system, namely, Root Genetic Programming, and extending our previous genetic operators based on projections on the phenotype space. The results show that these systems are able to tackle this problem being competitive with other state-of-the-art classifiers, and, also, give insight to approach large scale problems represented on high dimensional spaces.},
	language = {en},
	number = {663},
	urldate = {2016-09-20},
	booktitle = {{NEO} 2015},
	publisher = {Springer International Publishing},
	author = {Graff, Mario and Tellez, Eric S. and Escalante, Hugo Jair and Miranda-Jiménez, Sabino},
	editor = {Schütze, Oliver and Trujillo, Leonardo and Legrand, Pierrick and Maldonado, Yazmin},
	year = {2017},
	note = {DOI: 10.1007/978-3-319-44003-3\_2},
	keywords = {Artificial Intelligence (incl. Robotics), Big Data/Analytics, Computational intelligence, Computer Imaging, Vision, Pattern Recognition and Graphics, Genetic programming, optimization, Semantic Crossover, sentiment analysis, Text mining},
	pages = {43--65}
}
```

## Where to find the rest ##

* [Command line](https://mgraffg.github.io/EvoDAG/pages/cli.html): the four
  console commands with every option and the data formats, including the
  random search on the parameter space (`EvoDAG-params`).
* [Parameters](https://mgraffg.github.io/EvoDAG/pages/parameters.html): what
  each parameter does, its default, and the space that the random search
  explores.
* [How it works](https://mgraffg.github.io/EvoDAG/pages/how-it-works.html)
  and [Selection](https://mgraffg.github.io/EvoDAG/pages/selection.html): the
  method and the selection heuristics.

# Install EvoDAG #

* Installing `evodag` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with: 
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict   
conda install evodag
```
or with `mamba`:

```
mamba install evodag
```

* Install using pip  
```bash   
pip install EvoDAG
```

The page [Install](https://mgraffg.github.io/EvoDAG/pages/install.html) lists
the requirements and how to build the package from its source code.
