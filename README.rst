|Build Status|

|PyPI version|

|Coverage Status|

EvoDAG
======

Evolving Directed Acyclic Graph (EvoDAG) is a steady-state Genetic
Programming system with tournament selection. The main characteristic of
EvoDAG is that the genetic operation is performed at the root. EvoDAG
was inspired by the geometric semantic crossover proposed by `Alberto
Moraglio <https://scholar.google.com.mx/citations?user=0y4XRI0AAAAJ&hl=en&oi=ao>`__
*et al.* and the implementation performed by `Leonardo
Vanneschi <https://scholar.google.com.mx/citations?user=uR5K07QAAAAJ&hl=en&oi=ao>`__
*et al*.

Example using command line
--------------------------

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

.. code:: bash

    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

Random search on the EvoDAG's parameters space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended first step is to optimize the parameters used by EvoDAG.
In order to free from this task, EvoDAG can perform a random search on
the parameter space and select the best configuration found in this
random search. This can be performed as follows:

.. code:: bash__

    EvoDAG-params -P params.evodag -r 8 -u 4 iris.data

where ``-P`` indicates the file name where the parameters sampled are
stored, ``-r`` specifies the number of samples, ``-u`` indicates the
number of cpu cores, and ``iris.data`` is the dataset.

``params.evodag`` looks like:

.. code:: json

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

where ``fitness`` is the balance error rate on a validation set randomly
taken from the training set, in this case the 20% of the ``iris.data``.

Training EvoDAG
~~~~~~~~~~~~~~~

At this point we are in the position to train a model. Let us assume one
would like to create an ensemble of 10 classifiers on ``iris.data``.
Then the following command would do the job:

.. code:: bash

    EvoDAG-train -P params.evodag -m model.evodag -n 10 -u 4 iris.data 

where ``-m`` specifies the file name used to store the model, ``-n`` is
the size of the ensemble, ``-P`` receives EvoDAG's parameters, ``-u`` is
the number of cpu cores, and ``iris.data`` is the dataset.

Predict using EvoDAG model
~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point, EvoDAG has been trained and the model is store in
``model.evodag``, the next step is to use this model to predict some
unknown data. Given that we did not split ``iris.data`` in a training
and test set let us assume that ``iris.data`` contains the unknown data.
This can be achieved as follows:

.. code:: bash

    EvoDAG-predict -m model.evodag -o iris.predicted iris.data

where ``-o`` indicates the file name used to store the predictions,
``-m`` contains the model, and ``iris.data`` has the test set.

``iris.predicted`` looks like:

::

    Iris-setosa
    Iris-setosa
    Iris-setosa
    Iris-setosa
    Iris-setosa
    ...

Performance
===========

+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| dataset         | `auto-sklearn <https://github.com/automl/auto-sklearn>`__   | `SVC <http://scikit-learn.org/stable/>`__   | EvoDAG                       |
+=================+=============================================================+=============================================+==============================+
| banana          | :math:`28.00 \pm 3.69^*`                                    | **:math:`11.27 \pm 0.18`**                  | :math:`12.20 \pm 0.19^*`     |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| titanic         | :math:`37.18 \pm 1.64^*`                                    | :math:`30.27 \pm 0.36^*`                    | **:math:`30.04 \pm 0.26`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| thyroid         | :math:`23.38 \pm 3.99^*`                                    | **:math:`6.13 \pm 0.76`**                   | :math:`8.06 \pm 0.83^*`      |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| diabetis        | :math:`37.65 \pm 2.01^*`                                    | :math:`26.65 \pm 0.44^*`                    | **:math:`24.85 \pm 0.42`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| breast-cancer   | :math:`42.36 \pm 1.38^*`                                    | :math:`36.25 \pm 1.04^*`                    | **:math:`34.67 \pm 1.08`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| flare-solar     | :math:`39.05 \pm 1.49^*`                                    | :math:`33.41 \pm 0.38^*`                    | **:math:`32.88 \pm 0.32`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| heart           | :math:`27.69 \pm 2.85^*`                                    | :math:`18.12 \pm 0.63^*`                    | **:math:`16.58 \pm 0.73`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| ringnorm        | :math:`15.49 \pm 4.24^*`                                    | **:math:`1.96 \pm 0.10`**                   | :math:`2.56 \pm 0.08^*`      |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| twonorm         | :math:`20.87 \pm 4.49^*`                                    | :math:`2.90 \pm 0.09^*`                     | **:math:`2.70 \pm 0.04`**    |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| german          | :math:`39.45 \pm 1.62^*`                                    | :math:`29.00 \pm 0.50^*`                    | **:math:`28.77 \pm 0.53`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| image           | :math:`21.29 \pm 10.54^*`                                   | **:math:`3.32 \pm 0.29`**                   | :math:`3.88 \pm 0.43^*`      |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| waveform        | :math:`22.67 \pm 3.53^*`                                    | :math:`10.62 \pm 0.21^*`                    | **:math:`10.45 \pm 0.11`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+
| splice          | :math:`10.79 \pm 7.43^*`                                    | :math:`11.23 \pm 0.37^*`                    | **:math:`9.33 \pm 0.56`**    |
+-----------------+-------------------------------------------------------------+---------------------------------------------+------------------------------+

+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| dataset         | `auto-sklearn <https://github.com/automl/auto-sklearn>`__   | `SVC <http://scikit-learn.org/stable/>`__   | EvoDAG              |
+=================+=============================================================+=============================================+=====================+
| banana          | :math:`13.39`                                               | **:math:`11.02`**                           | :math:`12.05`       |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| titanic         | :math:`33.10`                                               | **:math:`29.63`**                           | :math:`29.75`       |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| thyroid         | :math:`11.96`                                               | **:math:`5.95`**                            | :math:`8.27`        |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| diabetis        | :math:`31.17`                                               | :math:`26.58`                               | **:math:`24.51`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| breast-cancer   | :math:`41.64`                                               | :math:`35.38`                               | **:math:`34.83`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| flare-solar     | :math:`34.90`                                               | :math:`33.31`                               | **:math:`32.99`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| heart           | :math:`20.68`                                               | :math:`18.28`                               | **:math:`15.96`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| ringnorm        | :math:`2.07`                                                | **:math:`1.83`**                            | :math:`2.51`        |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| twonorm         | :math:`3.29`                                                | :math:`2.83`                                | **:math:`2.69`**    |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| german          | :math:`36.25`                                               | :math:`28.88`                               | **:math:`28.57`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| image           | **:math:`3.06`**                                            | :math:`3.41`                                | :math:`3.66`        |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| waveform        | :math:`11.41`                                               | :math:`10.45`                               | **:math:`10.36`**   |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+
| splice          | **:math:`3.39`**                                            | :math:`11.07`                               | :math:`9.26`        |
+-----------------+-------------------------------------------------------------+---------------------------------------------+---------------------+

Dataset
-------

.. code:: python

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

Execute EvoDAG in the dataset (using a cluster with SGE)
--------------------------------------------------------

.. code:: bash

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

Install EvoDAG
--------------

-  Install using pip

   .. code:: bash

       pip install EvoDAG

Using source code
~~~~~~~~~~~~~~~~~

-  Clone the repository
   ``git clone  https://github.com/mgraffg/EvoDAG.git``
-  Install the package as usual
   ``python setup.py install``
-  To install only for the use then
   ``python setup.py install --user``

.. |Build Status| image:: https://travis-ci.org/mgraffg/EvoDAG.svg?branch=master
   :target: https://travis-ci.org/mgraffg/EvoDAG
.. |PyPI version| image:: https://badge.fury.io/py/EvoDAG.svg
   :target: https://badge.fury.io/py/EvoDAG
.. |Coverage Status| image:: https://coveralls.io/repos/github/mgraffg/EvoDAG/badge.svg?branch=master
   :target: https://coveralls.io/github/mgraffg/EvoDAG?branch=master
