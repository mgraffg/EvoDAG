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

In order to train the EvoDAG using a population of 10 individuals,
using early stopping to 11, sampling 100 different parameter configurations, creating
an ensemble of 12, and using 4 cores then the following command is used:

```bash   
~/.local/bin/EvoDAG -e 10 -p 11 -r 100 -u 4 -n 12 iris.data
```

The EvoDAG ensemble is stored in iris.evodag.gz. 

Now that the ensemble has been initialized one can predict a test set
and store the output in file called output.csv using the following command.

```bash   
~/.local/bin/EvoDAG -m iris.evodag.gz -t iris.data -o output.csv
```


# Performance #

|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [LinearSVC](http://scikit-learn.org/stable/) | [SVC (kernel rbf o sigmoid)](http://scikit-learn.org/stable/) | EvoDAG |
|-----|-------:|----------:|-------------------:|-------:|
|banana | $28.00 \pm 3.69^*$ | $46.28 \pm 0.81^*$ | **$11.27 \pm 0.18$** | $12.20 \pm 0.19^*$|
|titanic | $37.18 \pm 1.64^*$ | **$29.53 \pm 0.15$** | $30.01 \pm 0.28^*$ | $30.04 \pm 0.26^*$|
|thyroid | $23.38 \pm 3.99^*$ | $14.56 \pm 0.77^*$ | **$6.51 \pm 0.75$** | $8.06 \pm 0.83^*$|
|diabetis | $37.65 \pm 2.01^*$ | $26.16 \pm 0.39^*$ | $27.17 \pm 0.40^*$ | **$24.85 \pm 0.42$**|
|breast-cancer | $42.36 \pm 1.38^*$ | $35.09 \pm 1.01^*$ | $36.18 \pm 1.01^*$ | **$34.67 \pm 1.08$**|
|flare-solar | $39.05 \pm 1.49^*$ | $33.51 \pm 0.33^*$ | $33.35 \pm 0.43^*$ | **$32.88 \pm 0.32$**|
|heart | $27.69 \pm 2.85^*$ | $17.44 \pm 0.59^*$ | $17.76 \pm 0.67^*$ | **$16.58 \pm 0.73$**|
|ringnorm | $15.49 \pm 4.24^*$ | $24.71 \pm 0.16^*$ | **$1.96 \pm 0.10$** | $2.56 \pm 0.08^*$|
|twonorm | $20.87 \pm 4.49^*$ | $3.16 \pm 0.10^*$ | $2.87 \pm 0.08^*$ | **$2.70 \pm 0.04$**|
|german | $39.45 \pm 1.62^*$ | $28.98 \pm 0.47^*$ | $28.98 \pm 0.49^*$ | **$28.77 \pm 0.53$**|
|image | $21.29 \pm 10.54^*$ | $18.32 \pm 0.40^*$ | **$3.31 \pm 0.29$** | $3.88 \pm 0.43^*$|
|waveform | $22.67 \pm 3.53^*$ | $11.66 \pm 0.12^*$ | $10.46 \pm 0.20^*$ | **$10.45 \pm 0.11$**|
|splice | $10.79 \pm 7.43^*$ | $16.45 \pm 0.34^*$ | $11.09 \pm 0.31^*$ | **$9.33 \pm 0.56$**|

|dataset| [auto-sklearn](https://github.com/automl/auto-sklearn) | [LinearSVC](http://scikit-learn.org/stable/) | [SVC (kernel rbf o sigmoid)](http://scikit-learn.org/stable/) | EvoDAG |
|-----|-------:|----------:|-------------------:|-------:|
|banana | $13.39$ | $46.14$ | **$11.02$** | $12.05$|
|titanic | $33.10$ | **$29.37$** | $29.68$ | $29.75$|
|thyroid | $11.96$ | $14.60$ | **$6.19$** | $8.27$|
|diabetis | $31.17$ | $26.15$ | $27.39$ | **$24.51$**|
|breast-cancer | $41.64$ | $34.96$ | $35.95$ | **$34.83$**|
|flare-solar | $34.90$ | $33.64$ | $33.21$ | **$32.99$**|
|heart | $20.68$ | $18.07$ | $17.48$ | **$15.96$**|
|ringnorm | $2.07$ | $24.60$ | **$1.83$** | $2.51$|
|twonorm | $3.29$ | $3.09$ | $2.88$ | **$2.69$**|
|german | $36.25$ | $28.80$ | $28.90$ | **$28.57$**|
|image | **$3.06$** | $18.27$ | $3.41$ | $3.66$|
|waveform | $11.41$ | $11.58$ | **$10.35$** | $10.36$|
|splice | **$3.39$** | $16.68$ | $11.07$ | $9.26$|

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


