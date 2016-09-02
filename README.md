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


|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG (0.2.31)|
|------|---------------------------------------------:|------------------------------:|-------:|
|banana | $28.00 \pm 3.69^*$ | **$11.27 \pm 0.18$** | $12.07 \pm 0.16^*$|
|titanic | $37.18 \pm 1.64^*$ | $30.27 \pm 0.36^*$ | **$29.66 \pm 0.15$**|
|thyroid | $23.38 \pm 3.99^*$ | **$6.13 \pm 0.76$** | $7.94 \pm 0.84^*$|
|diabetis | $37.65 \pm 2.01^*$ | $26.65 \pm 0.44^*$ | **$24.54 \pm 0.44$**|
|breast-cancer | $42.36 \pm 1.38^*$ | $36.25 \pm 1.04^*$ | **$34.25 \pm 1.06$**|
|flare-solar | $39.05 \pm 1.49^*$ | $33.41 \pm 0.38^*$ | **$32.84 \pm 0.32$**|
|heart | $27.69 \pm 2.85^*$ | $18.12 \pm 0.63^*$ | **$16.52 \pm 0.69$**|
|ringnorm | $15.49 \pm 4.24^*$ | **$1.96 \pm 0.10$** | $2.48 \pm 0.10^*$|
|twonorm | $20.87 \pm 4.49^*$ | $2.90 \pm 0.09^*$ | **$2.67 \pm 0.04$**|
|german | $39.45 \pm 1.62^*$ | $29.00 \pm 0.50^*$ | **$28.34 \pm 0.50$**|
|image | $21.29 \pm 10.54^*$ | **$3.32 \pm 0.29$** | $3.98 \pm 0.44^*$|
|waveform | $22.67 \pm 3.53^*$ | $10.62 \pm 0.21^*$ | **$10.37 \pm 0.10$**|
|splice | $10.79 \pm 7.43^*$ | $11.23 \pm 0.37^*$ | **$9.91 \pm 0.50$**|

The next table presents the median performance (BER) of
the different classifiers, it is observed from the table, that
auto-sklearn obtained the best performance on two datasets, SVM
obtained the best performance on three datasets, and EvoDAG had the
best performance in eight datasets. Comparing the average and median
performance, it can be observed that EvoDAG had the best average
performance in _titanic_ and _splice_, and on median performance SVM and
auto-sklearn have the best performance on these dataset
respectively; whereas, SVM had the best average performance in _image_
dataset, and on median performance auto-skearn has the best
performance on this dataset.

|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG (0.2.31)|
|------|---------------------------------------------:|------------------------------:|-------:|
|banana | $13.39$ | **$11.02$** | $12.05$|
|titanic | $33.10$ | $29.63$ | **$29.39$**|
|thyroid | $11.96$ | **$5.95$** | $7.32$|
|diabetis | $31.17$ | $26.58$ | **$24.42$**|
|breast-cancer | $41.64$ | $35.38$ | **$34.41$**|
|flare-solar | $34.90$ | $33.31$ | **$32.71$**|
|heart | $20.68$ | $18.28$ | **$16.39$**|
|ringnorm | $2.07$ | **$1.83$** | $2.38$|
|twonorm | $3.29$ | $2.83$ | **$2.67$**|
|german | $36.25$ | $28.88$ | **$28.22$**|
|image | **$3.06$** | $3.41$ | $3.75$|
|waveform | $11.41$ | $10.45$ | **$10.33$**|
|splice | **$3.39$** | $11.07$ | $9.72$|

## Performance of different versions of EvoDAG ##

|dataset| EvoDAG (0.2.31) |Steady State| S. S. Inputs| S.S. Inputs Random Generation | Generational|G. Inputs|G. Inputs Random Generation|
|-----:|-------------:|---------:|---------:|------------------------:|----------:|-------:|-----------------------:|
|banana | $12.07 \pm 0.16^*$ | $12.20 \pm 0.19^*$ | $13.99 \pm 0.54^*$ | $12.22 \pm 0.20^*$ | **$12.02 \pm 0.18$** | $13.76 \pm 0.42^*$ | $12.15 \pm 0.17^*$|
|titanic | **$29.66 \pm 0.15$** | $30.04 \pm 0.26^*$ | $29.94 \pm 0.19^*$ | $29.71 \pm 0.22^*$ | $29.86 \pm 0.24^*$ | $29.95 \pm 0.21^*$ | $29.75 \pm 0.23^*$|
|thyroid | $7.94 \pm 0.84^*$ | $8.06 \pm 0.83^*$ | $8.89 \pm 0.83^*$ | $7.97 \pm 0.75^*$ | **$7.49 \pm 0.76$** | $8.98 \pm 0.83^*$ | $8.06 \pm 0.80^*$|
|diabetis | $24.54 \pm 0.44^*$ | $24.85 \pm 0.42^*$ | $24.80 \pm 0.44^*$ | **$24.43 \pm 0.44$** | $24.62 \pm 0.42^*$ | $24.81 \pm 0.43^*$ | $24.52 \pm 0.42^*$|
|breast-cancer | **$34.25 \pm 1.06$** | $34.67 \pm 1.08^*$ | $35.15 \pm 1.00^*$ | $34.68 \pm 1.02^*$ | $34.53 \pm 1.08^*$ | $35.00 \pm 1.00^*$ | $34.38 \pm 1.04^*$|
|flare-solar | $32.84 \pm 0.32^*$ | $32.88 \pm 0.32^*$ | $32.79 \pm 0.30^*$ | $32.84 \pm 0.31^*$ | $32.87 \pm 0.31^*$ | $32.79 \pm 0.32^*$ | **$32.73 \pm 0.31$**|
|heart | $16.52 \pm 0.69^*$ | $16.58 \pm 0.73^*$ | $16.72 \pm 0.72^*$ | $16.35 \pm 0.69^*$ | $16.38 \pm 0.71^*$ | $16.62 \pm 0.74^*$ | **$16.32 \pm 0.73$**|
|ringnorm | $2.48 \pm 0.10^*$ | $2.56 \pm 0.08^*$ | $3.52 \pm 0.11^*$ | **$2.41 \pm 0.08$** | $2.45 \pm 0.09^*$ | $3.52 \pm 0.11^*$ | $2.45 \pm 0.08^*$|
|twonorm | $2.67 \pm 0.04^*$ | $2.70 \pm 0.04^*$ | $2.77 \pm 0.04^*$ | **$2.63 \pm 0.03$** | $2.66 \pm 0.04^*$ | $2.78 \pm 0.05^*$ | $2.65 \pm 0.03^*$|
|german | **$28.34 \pm 0.50$** | $28.77 \pm 0.53^*$ | $28.49 \pm 0.45^*$ | $28.54 \pm 0.49^*$ | $28.71 \pm 0.49^*$ | $28.35 \pm 0.46^*$ | $28.48 \pm 0.51^*$|
|image | $3.98 \pm 0.44^*$ | $3.88 \pm 0.43^*$ | **$3.41 \pm 0.29$** | $4.09 \pm 0.56^*$ | $4.66 \pm 0.42^*$ | $3.61 \pm 0.32^*$ | $4.83 \pm 0.50^*$|
|waveform | $10.37 \pm 0.10^*$ | $10.45 \pm 0.11^*$ | $10.59 \pm 0.10^*$ | $10.34 \pm 0.09^*$ | $10.37 \pm 0.10^*$ | $10.59 \pm 0.10^*$ | **$10.29 \pm 0.09$**|
|splice | $9.91 \pm 0.50^*$ | **$9.33 \pm 0.56$** | $9.37 \pm 0.48^*$ | $9.56 \pm 0.43^*$ | $10.01 \pm 0.53^*$ | $9.66 \pm 0.51^*$ | $10.07 \pm 0.55^*$|


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


