[![Build Status](https://travis-ci.org/mgraffg/EvoDAG.svg?branch=master)](https://travis-ci.org/mgraffg/EvoDAG)
[![Anaconda-Server Badge](https://anaconda.org/mgraffg/evodag/badges/version.svg)](https://anaconda.org/mgraffg/evodag)
[![Anaconda-Server Badge](https://anaconda.org/mgraffg/evodag/badges/installer/conda.svg)](https://conda.anaconda.org/mgraffg)
[![PyPI version](https://badge.fury.io/py/EvoDAG.svg)](https://badge.fury.io/py/EvoDAG)
[![Coverage Status](https://coveralls.io/repos/github/mgraffg/EvoDAG/badge.svg?branch=master)](https://coveralls.io/github/mgraffg/EvoDAG?branch=master)

# EvoDAG #

Evolving Directed Acyclic Graph (EvoDAG) is a steady-state Genetic Programming system
with tournament selection. The main characteristic of EvoDAG is that
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

## EvoDAG from command line ##

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

```bash   
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

### Random search on the EvoDAG's parameters space

Firstly, it is recommended to optimize the parameters used by
EvoDAG. In order to free the user from this task, EvoDAG can perform a random
search on the parameter space. EvoDAG selects the best configuration found
on the random search. This can be performed as follows:

```bash__
EvoDAG-params -C -P params.evodag -r 734 -u 4 iris.data
```

where `-C` indicates that the task is classification, 
`-P` indicates the file name where the parameters sampled are
stored, `-r` specifies the number of samples 
(all the experiments presented here sampled 734 points which 
corresponded, in early versions, the 0.1% of the search space), `-u` indicates the number of
cpu cores, and `iris.data` is the dataset.

`params.evodag` looks like:

```json   
[
    {
        "Add": 30,
        "Cos": false,
        "Div": true,
        "Exp": true,
        "Fabs": true,
        "If": true,
        "Ln": true,
        "Max": 5,
        "Min": 30,
        "Mul": 0,
        "Sigmoid": true,
        "Sin": false,
        "Sq": true,
        "Sqrt": true,
        "classifier": true,
        "early_stopping_rounds": 2000,
        "fitness": [
            0.0,
            0.0,
            0.0
        ],
        "popsize": 500,
        "seed": 0,
        "unique_individuals": true
    },
...
```

where `fitness` is the balance error rate on a validation set, which
is randomly taken from the training set, in this case the 20% of `iris.data`.

### Training EvoDAG

At this point, we are in the position to train a model. Let us assume
one would like to create an ensemble of 10 classifiers on
`iris.data`. The following command performs this action: 

```bash   
EvoDAG-train -P params.evodag -m model.evodag -n 100 -u 4 iris.data 
```

where `-m` specifies the file name used to store the model, `-n` is
the size of the ensemble, `-P` receives EvoDAG's parameters, `-u` is
the number of cpu cores, and `iris.data` is the dataset. 


### Predict using EvoDAG model

At this point, EvoDAG has been trained and the model is stored in
`model.evodag`, the next step is to use this model to predict some
unknown data. Given that `iris.data` was not split into a training
and test set, let us assume that `iris.data` contains some unknown
data. In order to predict `iris.data` one would do:

```bash   
EvoDAG-predict -m model.evodag -o iris.predicted iris.data
```

where `-o` indicates the file name used to store the predictions, `-m`
contains the model, and `iris.data` has the test set.

`iris.predicted` looks like:

```
Iris-setosa
Iris-setosa
Iris-setosa
Iris-setosa
Iris-setosa
...
```

# Performance #

The next table presents the average performance in terms of the
balance error rate (BER) of PSMS, auto-sklearn, SVM, and EvoDAG on thirteen classification
problems (these benchmarks can be found:
[matlab](http://theoval.cmp.uea.ac.uk/matlab/benchmarks) and [text](http://ws.ingeotec.mx/~mgraffg/classification)).
  

The best performance among each classification dataset is in
bold face to facilitate the reading. It can be observed from the table
that PSMS obtained the best performance on five datasets, SVM
on one dataset, and EvoDAG obtained the
best performance in the rest of the datasets (seven). One
characteristic that caught our attention is the high confidence
intervals of auto-sklearn (not shown here), it is one order of magnitude higher than
the other systems. Analyzing the predictions performed by
auto-sklearn, it is found that in some of the trails the algorithm
predicts only one class, obtaining, consequently, the worst possible
performance, i.e., BER equals 50. This behaviour, clearly, can be
automatically spotted, and, one possible solution could be as simple
as execute auto-sklearn again on that particular case. Nonetheless, we
decided to keep auto-sklearn without modifications.

|Dataset|[PSMS](http://www.jmlr.org/papers/v10/escalante09a.html)|[auto-sklearn](https://github.com/automl/auto-sklearn)|[SVC(sklearn)](http://scikit-learn.org/stable/)|EvoDAG (0.10.2)|
| ----- | ---------------------------------------------: | --------------------------------------------: | -----------------------------------: | -----------: |
|banana          |     **11.08**      |  28.00  |11.27 | 11.91 | 
|titanic          |    **24.18**    |  37.18   |  30.27  |  29.68 |
|thyroid        |      **4.32**      |  23.38  |  6.13  |  8.63  | 
|diabetis        |     27.06   |   37.65  |  26.65  |  **25.54** |
|breast-cancer    |    **33.01**    |  42.36  |  36.25  |  34.12 | 
|flare-solar     |     34.81     |  39.05 |  33.41  |  **32.97** | 
|heart             |   20.69        |  27.69  |  18.12  |  **16.20** |
|ringnorm       |      7.98      |  15.49  |  1.96  |  **1.59** |
|twonorm       |       3.09      |  20.87  |  2.90  |  **2.48** | 
|german         |      30.10    | 39.45  |  29.00  | **28.55** | 
|image         |       **2.90** | 21.29  |  3.32  | 4.55 | 
|waveform      |       12.80   | 22.67  |  **10.62**  |  10.90 |
|splice       |        14.63  | 10.79  |  11.23  |  **9.99** | 


# Install EvoDAG #

* Install using [conda](https://www.continuum.io)  
```bash   
conda install -c mgraffg evodag
```
Currently, EvoDAG is available for Python 2.7, 3.3, 3.4, 3.5, and 3.6 for linux-64 and osx-64

* Install using pip  
```bash   
pip install EvoDAG
```

## Using source code ##

* Clone the repository  
```git clone  https://github.com/mgraffg/EvoDAG.git```
* Install the package as usual  
```python setup.py install```
* To install only for the use then  
```python setup.py install --user```


