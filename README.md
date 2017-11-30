[![Build Status](https://travis-ci.org/mgraffg/EvoDAG.svg?branch=master)](https://travis-ci.org/mgraffg/EvoDAG)
[![Build status](https://ci.appveyor.com/api/projects/status/vx09qqgluff3ko5e?svg=true)](https://ci.appveyor.com/project/mgraffg/evodag)
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

## Quick Start ##

There are two options to use EvoDAG, one is as a library
and the other is using the command line interface.

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
presented in the last column of the performance table. 

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

The next table presents the average performance in terms of the
balance error rate (BER) of different classifiers found in
[scikit-learn](http://scikit-learn.org) on nine classification
problems (these benchmarks can be found:
[matlab](http://theoval.cmp.uea.ac.uk/matlab/benchmarks) and [text](http://ws.ingeotec.mx/~mgraffg/classification)).
The best performance among each classification dataset is in
bold face to facilitate the reading. EvoDAG is trained using the
commands describe in Quick Start Section. 

|Classifier|banana | thyroid | diabetis | heart | ringnorm | twonorm |german| image | waveform|Average rank|
|-------|------:|------:|-------:|----:|--------:|--------:|------:|-----:|--------:|---------:|
EvoDAG |11.93 | 7.79 | **24.87** | 16.86 | 2.00 | 2.64 | 28.83 | 3.42 | **10.69**|3.00|
SVC|**11.59** | 8.08 | 29.82 | 17.75 | 1.84 | 2.73 | 33.31 | 8.86 | 10.74|5.33|
GaussianNB|41.65 | 11.51 | 28.61 | **16.34** | **1.44** | 2.40 | 30.58 | 36.67 | 12.21|5.67|
GradientBoostingClassifier|13.84 | 8.25 | 28.52 | 21.06 | 6.65 | 5.74 | 31.08 | **2.09** | 13.62|6.11|
MLPClassifier|18.75 | 9.31 | 28.61 | 18.33 | 10.96 | 2.82 | 32.03 | 3.35 | 11.38|6.44|
NearestCentroid|46.33 | 22.49 | 28.19 | 16.66 | 24.09 | **2.32** | **27.68** | 37.09 | 12.96|7.33|
AdaBoostClassifier|28.16 | 8.73 | 29.32 | 22.51 | 7.01 | 5.35 | 31.58 | 3.20 | 14.64|8.33|
LinearSVC|49.97 | 16.91 | 28.47 | 16.80 | 25.18 | 3.58 | 32.44 | 18.65 | 14.20|8.67|
ExtraTreesClassifier|13.49 | **6.56** | 32.99 | 21.06 | 8.42 | 7.76 | 36.95 | 2.38 | 16.48|8.67|
RandomForestClassifier|13.59 | 7.86 | 31.93 | 21.42 | 9.58 | 8.89 | 36.72 | 2.10 | 16.87|9.11|
LogisticRegression |49.99 | 20.18 | 28.29 | 16.97 | 25.29 | 2.92 | 32.07 | 18.74 | 13.73|9.11|
KNeighborsClassifier|11.92 | 12.06 | 32.12 | 18.64 | 43.72 | 3.76 | 36.18 | 5.26 | 13.80|9.44|
BernoulliNB|45.46 | 32.47 | 31.88 | 16.62 | 28.13 | 5.83 | 32.78 | 39.03 | 14.33|11.44|
DecisionTreeClassifier|15.21 | 9.28 | 33.85 | 27.06 | 19.20 | 20.87 | 36.51 | 3.38 | 20.78|11.78|
SGDClassifier |50.28 | 17.69 | 34.03 | 22.19 | 32.65 | 3.89 | 38.50 | 25.12 | 17.61|14.00|
PassiveAggressiveClassifier|49.04 | 19.48 | 34.68 | 23.31 | 31.48 | 3.85 | 38.65 | 26.24 | 17.10|14.11|
Perceptron|50.15 | 18.18 | 34.56 | 21.59 | 33.49 | 3.92 | 37.47 | 26.69 | 19.03|14.44|


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

In order to boost the performance of EvoDAG, it is recommended to optimize the parameters used by
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


# Install EvoDAG #

* Install using [conda](https://www.continuum.io)  
```bash   
conda install -c mgraffg evodag
```
Currently, EvoDAG is available for Python 3.4, 3.5, and 3.6 for
windows-32, windows-64, linux-64 and osx-64

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


