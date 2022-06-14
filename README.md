[![Tests](https://github.com/mgraffg/EvoDAG/actions/workflows/test.yaml/badge.svg)](https://github.com/mgraffg/EvoDAG/actions/workflows/test.yaml)
[![Coverage Status](https://coveralls.io/repos/github/mgraffg/EvoDAG/badge.svg?branch=develop)](https://coveralls.io/github/mgraffg/EvoDAG?branch=develop)
[![PyPI version](https://badge.fury.io/py/evodag.svg)](https://badge.fury.io/py/evodag)
[![azure](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/evodag-feedstock?branchName=main)](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=16226&branchName=main)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/evodag.svg)](https://anaconda.org/conda-forge/evodag) 
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/evodag.svg)](https://anaconda.org/conda-forge/evodag) 

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

|Classifier|Average rank|banana | thyroid | diabetis | heart | ringnorm | twonorm |german| image | waveform|
|-------|------:|------:|-------:|----:|--------:|--------:|------:|-----:|--------:|---------:|
|EvoDAG|2.4|12.0|7.7|**25.0**|**16.1**|1.5|2.3|28.3|3.6|10.8|
|SVC|5.3|11.6|8.1|29.8|17.7|1.8|2.7|33.3|8.9|**10.7**|
|GaussianNB|5.9|41.6|11.5|28.6|16.3|1.4|2.4|30.6|36.7|12.2|
|GradientBoosting|6.1|13.8|8.2|28.5|21.1|6.6|5.7|31.1|2.1|13.6|
|MLP|6.4|18.8|9.3|28.6|18.3|11.0|2.8|32.0|3.4|11.4|
|NearestCentroid|7.4|46.3|22.5|28.2|16.7|24.1|2.3|27.7|37.1|13.0|
|AdaBoost|8.3|28.2|8.7|29.3|22.5|7.0|5.4|31.6|3.2|14.6|
|ExtraTrees|8.7|13.5|6.6|33.0|21.1|8.4|7.8|36.9|2.4|16.5|
|LinearSVC|8.8|50.0|16.9|28.5|16.8|25.2|3.6|32.4|18.6|14.2|
|LogisticRegression|9.1|50.0|20.2|28.3|17.0|25.3|2.9|32.1|18.7|13.7|
|RandomForest|9.1|13.6|7.9|31.9|21.4|9.6|8.9|36.7|2.1|16.9|
|KNeighbors|9.4|11.9|12.1|32.1|18.6|43.7|3.8|36.2|5.3|13.8|
|BernoulliNB|11.6|45.5|32.5|31.9|16.6|28.1|5.8|32.8|39.0|14.3|
|DecisionTree|11.8|15.2|9.3|33.9|27.1|19.2|20.9|36.5|3.4|20.8|
|SGD|14.0|50.3|17.7|34.0|22.2|32.6|3.9|38.5|25.1|17.6|
|PassiveAggressive|14.1|49.0|19.5|34.7|23.3|31.5|3.9|38.7|26.2|17.1|
|Perceptron|14.4|50.2|18.2|34.6|21.6|33.5|3.9|37.5|26.7|19.0|

The predictions of EvoDAG were obtained using the following script:

```bash   
dirname=evodag-`EvoDAG-params --version | awk '{print $2}'`
[ ! -d $dirname ] && mkdir $dirname

echo Haciendo `EvoDAG-params --version`

for train in csv/*train_data*.csv;
do
        test=`python -c "import sys; print(sys.argv[1].replace('train', 'test'))" $train`;
        output=`basename ${test} .csv`
        predict=${dirname}/${output}.predict
        model=${dirname}/${output}.model
        if [ ! -f $model ]
        then
            EvoDAG-train -C -u 32 -m $model -t $test $train
        fi
        if [ ! -f $predict ]
        then
            EvoDAG-predict -u 32 -m $model -o $predict $test
        fi
done
```

The predictions sklearn classifiers were obtained using the following
code:

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from glob import glob
import numpy as np


def predict(train, test, alg):
    X = np.loadtxt(train, delimiter=',')
    Xtrain = X[:, :-1]
    ytrain = X[:, -1]
    Xtest = np.loadtxt(test, delimiter=',')
    m = alg().fit(Xtrain, ytrain)
    return m.predict(Xtest)

ALG = [LogisticRegression, SGDClassifier, Perceptron,
       PassiveAggressiveClassifier, SVC, LinearSVC, KNeighborsClassifier, NearestCentroid,
       GaussianNB, BernoulliNB, DecisionTreeClassifier, RandomForestClassifier,
       ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, MLPClassifier]

for dataset in ['banana', 'thyroid', 'diabetis', 'heart',
                'ringnorm', 'twonorm', 'german', 'image',
                'waveform']:
    for train in glob('csv/%s_train_data*.csv' % dataset):
        test = train.replace('_train_', '_test_')
        hy_alg = [predict(train, test, alg) for alg in ALG]
```

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

## Using source code ##

* Clone the repository  
```git clone  https://github.com/mgraffg/EvoDAG.git```
* Install the package as usual  
```python setup.py install```
* To install only for the use then  
```python setup.py install --user```


