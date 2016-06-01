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
==========================

Let us assume one wants to create a classifier of iris dataset. The
first step is to download the dataset from the UCI Machine Learning
Repository

.. code:: bash

    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

In order to train the EvoDAG using a population of 10 individuals, using
early stopping to 11, sampling 100 different parameter configurations,
creating an ensemble of 12, and using 4 cores then the following command
is used:

.. code:: bash

    ~/.local/bin/EvoDAG -e 10 -p 11 -r 100 -u 4 -n 12 iris.data

The EvoDAG ensemble is stored in iris.evodag.gz.

Now that the ensemble has been initialized one can predict a test set
and store the output in file called output.csv using the following
command.

.. code:: bash

    ~/.local/bin/EvoDAG -m iris.evodag.gz -t iris.data -o output.csv

Install EvoDAG
--------------

-  Install using pip
   ``pip install EvoDAG``

Using the source code
~~~~~~~~~~~~~~~~~~~~~

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
