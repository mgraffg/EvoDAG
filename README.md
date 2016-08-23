[![Build Status](https://travis-ci.org/mgraffg/EvoDAG.svg?branch=master)](https://travis-ci.org/mgraffg/EvoDAG)

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
EvoDAG-params -P params.evodag -r 8 -u 4 iris.data
```

where `-P` indicates the file name where the parameters sampled are
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

The next table presents the average performance with 95%
confidence intervals in terms of the balance error rate (BER) of
auto-sklearn, SVM, and EvoDAG on thirteen classification
problems (these benchmarks can be found:
[matlab](http://theoval.cmp.uea.ac.uk/matlab/benchmarks) and [text](http://ws.ingeotec.mx/~mgraffg/classification)).
  

The best performance among each classification dataset is in
bold face to facilitate the reading; it was compared against the others in order to know whether
the difference in performance was statistically significant. The
superscript $^*$ indicates that the difference between the best
performance and the one having the superscript is statistically
significant with a 95% confidence. This statistical test was
performed using the Wilcoxon signed-rank test
and the $p$-values were adjusted using 
the Holm-Bonferroni method in order to consider the multiple
comparisons performed.

It can be observed from the table that SVM obtained
the best performance in four of the datasets, and EvoDAG obtained the
best performance in the rest of the datasets (nine). In all the cases, the
difference in performance was statistically significant. One
characteristic that caught our attention is the high confidence
intervals of auto-sklearn, it is one order of magnitude higher than
the other systems. Analyzing the predictions performed by
auto-sklearn, it is found that in some of the trails the algorithm
predicts only one class, obtaining, consequently, the worst possible
performance, i.e., BER equals 50. This behaviour, clearly, can be
automatically spotted, and, one possible solution could be as simple
as execute auto-sklearn again on that particular case. Nonetheless, we
decided to keep auto-sklearn without modification, and, instead, it
is decided to include another table that presents the performance using the median.


|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG (SteadyState)|EvoDAG (Generational)|
|------|---------------------------------------------:|------------------------------:|-------:|-----------:|
|banana | $25.63 \pm 3.22^*$ | **$11.09 \pm 0.19$** | $12.04 \pm 0.19^*$ | $11.86 \pm 0.18^*$|
|titanic | $27.06 \pm 1.47^*$ | **$23.61 \pm 0.49$** | $24.09 \pm 0.46^*$ | $23.92 \pm 0.41^*$|
|thyroid | $14.61 \pm 2.32^*$ | **$4.37 \pm 0.48$** | $5.53 \pm 0.52^*$ | $5.19 \pm 0.50^*$|
|diabetis | $29.00 \pm 0.99^*$ | $26.67 \pm 0.40^*$ | $24.83 \pm 0.40^*$ | **$24.52 \pm 0.36$**|
|breast-cancer | **$27.74 \pm 0.93$** | $33.96 \pm 1.09^*$ | $32.65 \pm 0.99^*$ | $32.88 \pm 0.97^*$|
|flare-solar | $41.15 \pm 1.92^*$ | $34.45 \pm 0.40^*$ | **$34.20 \pm 0.34$** | $34.23 \pm 0.32^*$|
|heart | $25.87 \pm 2.48^*$ | $17.91 \pm 0.63^*$ | $16.29 \pm 0.70^*$ | **$16.10 \pm 0.69$**|
|ringnorm | $15.35 \pm 4.20^*$ | **$1.95 \pm 0.10$** | $2.55 \pm 0.08^*$ | $2.44 \pm 0.09^*$|
|twonorm | $20.87 \pm 4.49^*$ | $2.90 \pm 0.09^*$ | $2.70 \pm 0.04^*$ | **$2.66 \pm 0.04$**|
|german | **$26.39 \pm 0.66$** | $29.43 \pm 0.59^*$ | $28.84 \pm 0.55^*$ | $28.81 \pm 0.52^*$|
|image | $24.17 \pm 12.16^*$ | **$3.27 \pm 0.28$** | $3.83 \pm 0.41^*$ | $4.54 \pm 0.40^*$|
|waveform | $17.08 \pm 2.05^*$ | $12.09 \pm 0.27^*$ | $12.08 \pm 0.16^*$ | **$11.99 \pm 0.15$**|
|splice | $10.52 \pm 7.11^*$ | $11.28 \pm 0.37^*$ | **$9.36 \pm 0.57$** | $10.04 \pm 0.53^*$|

The next table presents the median performance (BER) of
the different classifiers, it is observed from the table, that
auto-sklearn obtained the best performance on two datasets, SVM
obtained the best performance on four datasets, and EvoDAG had the
best performance in seven datasets. Comparing the average and median
performance, it can be observed that EvoDAG had the best average
performance in _titanic_ and _splice_, and on median performance SVM and
auto-sklearn have the best performance on these dataset
respectively; whereas, SVM had the best average performance in _image_
dataset, and on median performance auto-skearn has the best
performance on this dataset.

|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG (SteadyState)|EvoDAG (Generational)|
|------|---------------------------------------------:|------------------------------:|-------:|-----------:|
|banana | $13.01$ | **$10.90$** | $11.92$ | $11.76$|
|titanic | $24.82$ | **$22.87$** | $23.01$ | $23.11$|
|thyroid | $9.33$ | **$4.00$** | $5.33$ | $5.33$|
|diabetis | $26.83$ | $26.67$ | $24.67$ | **$24.50$**|
|breast-cancer | **$27.27$** | $33.77$ | $32.47$ | $33.77$|
|flare-solar | $35.88$ | **$34.25$** | $34.38$ | $34.25$|
|heart | $20.00$ | $18.00$ | **$16.00$** | $16.00$|
|ringnorm | $2.06$ | **$1.82$** | $2.51$ | $2.38$|
|twonorm | $3.29$ | $2.83$ | $2.69$ | **$2.63$**|
|german | **$26.00$** | $29.33$ | $28.67$ | $28.67$|
|image | **$2.87$** | $3.42$ | $3.61$ | $4.46$|
|waveform | **$10.63$** | $11.95$ | $12.02$ | $11.95$|
|splice | **$3.43$** | $11.10$ | $9.26$ | $10.11$|

# Install EvoDAG #

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


