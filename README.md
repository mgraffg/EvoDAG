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

## Example using command line ##

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

```bash   
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
```

### Random search on the EvoDAG's parameters space

The recommended first step is to optimize the parameters used by
EvoDAG. In order to free from this task, EvoDAG can perform a random
search on the parameter space and select the best configuration found
in this random search. This can be performed as follows:

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

where `fitness` is the balance error rate on a validation set randomly
taken from the training set, in this case the 20\% of the `iris.data`.

### Training EvoDAG

At this point we are in the position to train a model. Let us assume
one would like to create an ensemble of 10 classifiers on
`iris.data`. Then the following command would do the job:

```bash   
EvoDAG-train -P params.evodag -m model.evodag -n 10 -u 4 iris.data 
```

where `-m` specifies the file name used to store the model, `-n` is
the size of the ensemble, `-P` receives EvoDAG's parameters, `-u` is
the number of cpu cores, and `iris.data` is the dataset. 


### Predict using EvoDAG model

At this point, EvoDAG has been trained and the model is store in
`model.evodag`, the next step is to use this model to predict some
unknown data. Given that we did not split `iris.data` in a training
and test set let us assume that `iris.data` contains the unknown
data. This can be achieved as follows:

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

|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG |
|------|---------------------------------------------:|------------------------------:|-------:|
|banana | $28.00 \pm 3.69^*$ | **$11.27 \pm 0.18$** | $12.20 \pm 0.19^*$|
|titanic | $37.18 \pm 1.64^*$ | $30.27 \pm 0.36^*$ | **$30.04 \pm 0.26$**|
|thyroid | $23.38 \pm 3.99^*$ | **$6.13 \pm 0.76$** | $8.06 \pm 0.83^*$|
|diabetis | $37.65 \pm 2.01^*$ | $26.65 \pm 0.44^*$ | **$24.85 \pm 0.42$**|
|breast-cancer | $42.36 \pm 1.38^*$ | $36.25 \pm 1.04^*$ | **$34.67 \pm 1.08$**|
|flare-solar | $39.05 \pm 1.49^*$ | $33.41 \pm 0.38^*$ | **$32.88 \pm 0.32$**|
|heart | $27.69 \pm 2.85^*$ | $18.12 \pm 0.63^*$ | **$16.58 \pm 0.73$**|
|ringnorm | $15.49 \pm 4.24^*$ | **$1.96 \pm 0.10$** | $2.56 \pm 0.08^*$|
|twonorm | $20.87 \pm 4.49^*$ | $2.90 \pm 0.09^*$ | **$2.70 \pm 0.04$**|
|german | $39.45 \pm 1.62^*$ | $29.00 \pm 0.50^*$ | **$28.77 \pm 0.53$**|
|image | $21.29 \pm 10.54^*$ | **$3.32 \pm 0.29$** | $3.88 \pm 0.43^*$|
|waveform | $22.67 \pm 3.53^*$ | $10.62 \pm 0.21^*$ | **$10.45 \pm 0.11$**|
|splice | $10.79 \pm 7.43^*$ | $11.23 \pm 0.37^*$ | **$9.33 \pm 0.56$**|


|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [SVC](http://scikit-learn.org/stable/) | EvoDAG |
|------|---------------------------------------------:|------------------------------:|-------:|
|banana | $13.39$ | **$11.02$** | $12.05$|
|titanic | $33.10$ | **$29.63$** | $29.75$|
|thyroid | $11.96$ | **$5.95$** | $8.27$|
|diabetis | $31.17$ | $26.58$ | **$24.51$**|
|breast-cancer | $41.64$ | $35.38$ | **$34.83$**|
|flare-solar | $34.90$ | $33.31$ | **$32.99$**|
|heart | $20.68$ | $18.28$ | **$15.96$**|
|ringnorm | $2.07$ | **$1.83$** | $2.51$|
|twonorm | $3.29$ | $2.83$ | **$2.69$**|
|german | $36.25$ | $28.88$ | **$28.57$**|
|image | **$3.06$** | $3.41$ | $3.66$|
|waveform | $11.41$ | $10.45$ | **$10.36$**|
|splice | **$3.39$** | $11.07$ | $9.26$|


## Dataset ##

```python   
import glob
for train in glob.glob('data/*train_data*.asc'):
    label = train.split('/')[1].replace('data', 'labels')
    label = train.split('/')[0] + '/' + label
    with open(train) as fpt:
        t = fpt.readlines()
    with open(label) as fpt:
        l = fpt.readlines()
    with open(train, 'w') as fpt:
        for a, b in zip(t, l):
            d = [x.rstrip().rstrip() for x in a.split(" ")]
            d = [x for x in d if len(x)]
            d.append(b.rstrip().lstrip())
            fpt.write(",".join(d) + '\n')

for test in glob.glob('data/*test_data*.asc'):
    with open(test) as fpt:
        l = fpt.readlines()
    with open(test, 'w') as fpt:
        for a in l:
            d = [x.rstrip().rstrip() for x in a.split(" ")]
            d = [x for x in d if len(x)]
            fpt.write(",".join(d) + '\n')
```

## Execute EvoDAG in the dataset (using a cluster with SGE) ##

```bash
#!/bin/sh
for i in data/*train_data*.asc;
do
    seed=`python -c "import sys; print(sys.argv[1].split('.asc')[0].split('_')[-1])" $i`;
    test=`python -c "import sys; print(sys.argv[1].replace('train', 'test'))" $i`;
    output=data/`basename $test .asc`.evodag
    cache=$i.rs.evodag
    cpu=2
    if [ ! -f $cache ]
       then
	echo \#!/bin/sh > job.sh
	echo \#$ -N `basename $i .asc` >> job.sh
	echo \#$ -S /bin/sh >> job.sh
	echo \#$ -cwd >> job.sh
	echo \#$ -j y >> job.sh
	echo ~/miniconda3/bin/EvoDAG -s $seed -u $cpu -o $output -m $output -t $test --cache-file $cache -r 734 $i >> job.sh
	qsub job.sh
    fi
done
```

## Install EvoDAG ##

* Install using pip  
```bash   
pip install EvoDAG
```

### Using source code ###

* Clone the repository  
```git clone  https://github.com/mgraffg/EvoDAG.git```
* Install the package as usual  
```python setup.py install```
* To install only for the use then  
```python setup.py install --user```


