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

Currently, it is under review (submitted to
[ROPEC 2016](http://ropec.org)) a paper that describes EvoDAG. The
paper can be download from [here](http://ws.ingeotec.mx/~mgraffg/publications/pdf/ropec2016.pdf).

## Citing EvoDAG ##

If you like EvoDAG, and it is used in a scientific publication, I would
appreciate citations to the following paper:

[Semantic Genetic Programming Operators Based on Projections in the Phenotype Space](http://www.rcs.cic.ipn.mx/rcs/2015_94/Semantic%20Genetic%20Programming%20Operators%20Based%20on%20Projections%20in%20the%20Phenotype%20Space.pdf).
M Graff, ES Tellez, E Villasenor, S Miranda-Jiménez. Research in Computing Science 94, 73-85

```bibtex
@article{graff2015semantic,
  title={Semantic Genetic Programming Operators Based on Projections in the Phenotype Space},
  author={Graff, Mario and Tellez, Eric Sadit and Villasenor, Elio and Miranda-Jim{\'e}nez, Sabino},
  journal={Research in Computing Science},
  volume={94},
  pages={73--85},
  year={2015},
  publisher={National Polytechnic Institute, Mexico}
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
EvoDAG-params -C -P params.evodag -r 8 -u 4 iris.data
```

where `-C` indicates that the task is classification, `-P` indicates the file name where the parameters sampled are
stored, `-r` specifies the number of samples, `-u` indicates the number of
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
EvoDAG-train -P params.evodag -m model.evodag -n 10 -u 4 iris.data 
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
intervals of auto-sklearn, it is one order of magnitude higher than
the other systems. Analyzing the predictions performed by
auto-sklearn, it is found that in some of the trails the algorithm
predicts only one class, obtaining, consequently, the worst possible
performance, i.e., BER equals 50. This behaviour, clearly, can be
automatically spotted, and, one possible solution could be as simple
as execute auto-sklearn again on that particular case. Nonetheless, we
decided to keep auto-sklearn without modifications.

 | Dataset  | [PSMS](http://www.jmlr.org/papers/v10/escalante09a.html)  |  [auto-sklearn](https://github.com/automl/auto-sklearn)  |  [SVC(sklearn)](http://scikit-learn.org/stable/)  |  EvoDAG (0.2.33) | 
 | ------- | ---------: | ---------------------------------------------: | ------------------------------: | -------: | 
 | banana          |     **11.08**      |  28.00  |  11.27   |  12.07  | 
 | titanic          |    **24.18**    |  37.18   |  30.27  |  29.66  | 
 | thyroid        |      **4.32**      |  23.38  |  6.13  |  7.94  | 
 | diabetis        |     27.06   |   37.65  |  26.65  |  **24.54** | 
 | breast-cancer    |    **33.01**    |  42.36  |  36.25  |  34.25 | 
 | flare-solar     |     34.81     |  39.05 |  33.41  |  **32.84**  | 
 | heart             |   20.69        |  27.69  |  18.12  |  **16.52** | 
 | ringnorm       |      7.98      |  15.49  |  **1.96**  |  2.48 | 
 | twonorm       |       3.09      |  20.87  |  2.90  |  **2.67** | 
 | german         |      30.10    | 39.45  |  29.00  |  **28.34**  | 
 | image         |       **2.90** | 21.29  |  3.32  |  3.98 | 
 | waveform      |       12.80   | 22.67  |  10.62  |  **10.37**  | 
 | splice       |        14.63  | 10.79  |  11.23  |  **9.91**  | 


# Install EvoDAG #

* Install using [conda](https://www.continuum.io)  
```bash   
conda install -c mgraffg evodag
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


