# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import json
import os
import gzip


def line_iterator(filename):
    """Iterate over the non-empty lines of a text file, stripped.

    Parameters
    ----------
    filename : str
        Path of the file; a name ending in ``.gz`` is read compressed.

    Yields
    ------
    str
        Each line without surrounding white space. Blank lines are skipped.
    """
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
    else:
        try:
            f = open(filename, encoding='utf8')
        except TypeError:
            f = open(filename)
    while True:
        line = f.readline()
        # Test the type of the line and encode it if neccesary...
        if type(line) is bytes:
            try:
                line = str(line, encoding='utf8')
            except TypeError:
                line = str(line)
        # If the line is empty, we are done...
        if len(line) == 0:
            break
        line = line.strip()
        # If line is empty, jump to next...
        if len(line) == 0:
            continue
        yield line
    # Close the file...
    f.close()


def json_iterator(filename):
    """Iterate over a file with one JSON object per line, yielding the decoded objects.

    It accepts the same files as ``line_iterator``.
    """
    for line in line_iterator(filename):
        yield json.loads(line)


def tonparray(a):
    """Convert ``a`` into a numpy array."""
    return np.array(a)


def index_data(pairs, size):
    "Dense vector of ``size`` with the value of each ``(position, value)``"
    res = np.zeros(size)
    if len(pairs):
        res[[i for i, _ in pairs]] = [v for _, v in pairs]
    return res


def BER(y, yh):
    """Balanced error rate, in percent.

    It is the mean, over the classes in ``y``, of the percentage of the
    examples of the class that are not predicted as that class.

    Parameters
    ----------
    y : numpy.ndarray
        True classes.
    yh : numpy.ndarray
        Predicted classes.

    Returns
    -------
    float
        0 is perfect; 50 is the error of a classifier that is no better
        than chance on two balanced classes.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG.utils import BER
    >>> y = np.array([0, 0, 0, 0, 1, 1])
    >>> yh = np.array([0, 0, 0, 1, 1, 0])
    >>> float(BER(y, yh))
    37.5
    """
    u = np.unique(y)
    b = 0
    for cl in u:
        m = y == cl
        b += (~(y[m] == yh[m])).sum() / float(m.sum())
    return (b / float(u.shape[0])) * 100.


def RSE(x, y):
    """Relative squared error: the squared error of ``y`` with respect to ``x`` over the variance of ``x`` (not divided by the size).

    Parameters
    ----------
    x : numpy.ndarray
        True values.
    y : numpy.ndarray
        Predicted values.

    Returns
    -------
    float
        0 is perfect, and 1 is the error of always predicting the mean of ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from EvoDAG.utils import RSE
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> float(RSE(x, x)), float(RSE(x, np.full(3, 2.0)))
    (0.0, 1.0)
    """
    return ((x - y)**2).sum() / ((x - x.mean())**2).sum()


params_fname = os.path.join(os.path.dirname(__file__), 'conf', 'parameter_values.json')
with open(params_fname, 'r') as fpt:
    PARAMS = json.loads(fpt.read())


class Inputs(object):
    """Reader of the data files of the command line tools (CSV and JSON).

    Values that are not numbers are replaced by integers; the dictionaries
    ``word2id`` (for the inputs) and ``label2id`` (for the classes) store
    the correspondence, and they are saved with the models.

    Examples
    --------
    >>> import os, tempfile
    >>> from EvoDAG.utils import Inputs
    >>> path = os.path.join(tempfile.mkdtemp(), "train.csv")
    >>> with open(path, "w") as fpt:
    ...     for row in ("1.0,2.0,a", "3.0,4.0,b", "5.0,6.0,a"):
    ...         print(row, file=fpt)
    >>> inputs = Inputs()
    >>> X, y = inputs.read_csv(path, 1)
    >>> X.shape, y.tolist()
    ((3, 2), [0, 1, 0])
    >>> inputs.label2id
    {'a': 0, 'b': 1}
    """
    def __init__(self):
        self._word2id = {}
        self._label2id = {}

    @property
    def word2id(self):
        "Dictionary that maps the inputs that are not numbers to integers"
        return self._word2id

    @property
    def label2id(self):
        "Dictionary that maps the class labels that are not numbers to integers"
        return self._label2id

    @word2id.setter
    def word2id(self, s):
        self._word2id = s

    @label2id.setter
    def label2id(self, s):
        self._label2id = s

    @staticmethod
    def _num_terms(a):
        if 'num_terms' in a:
            num_terms = a['num_terms']
        else:
            num_terms = len(a)
            if 'klass' in a:
                num_terms -= 1
        return num_terms

    def convert(self, x):
        """Convert an input value to a float; text is mapped to an integer with ``word2id``."""
        try:
            return float(x)
        except ValueError:
            if x not in self.word2id:
                self.word2id[x] = len(self.word2id)
            return self.word2id[x]

    def convert_label(self, x):
        """Convert a class label to a float; text, and not finite numbers, are mapped to an integer with ``label2id``."""
        try:
            x = float(x)
            if np.isfinite(x):
                return x
            x = str(x)
        except ValueError:
            pass
        if x not in self.label2id:
            self.label2id[x] = len(self.label2id)
        return self.label2id[x]

    def _read_csv(self, fname):
        X = []
        for i in line_iterator(fname):
            x = i.rstrip().lstrip()
            if len(x):
                X.append([i for i in x.split(',')])
        return X

    def read_csv(self, fname, dim):
        """Read a CSV file without header: the inputs and, optionally, the targets.

        Parameters
        ----------
        fname : str
            File name; ``.gz`` files are read compressed.
        dim : int
            Number of target columns at the end of each row: 1 for a class
            or a real target, greater than 1 for several real targets, 0
            when the file has only inputs.

        Returns
        -------
        tuple
            ``(X, y)``. ``X`` is an array with one row per example. ``y`` is
            an array when ``dim`` is 1, a list with one array per target
            when it is greater, and ``None`` when it is 0.
        """
        X = []
        y = []
        d = self._read_csv(fname)
        if dim > 1:
            for x in d:
                X.append([self.convert(i) for i in x[:-dim]])
                y.append(x[-dim:])
            X = np.array(X)
            y = [np.array([float(x[i]) for x in y]) for i in range(dim)]
        elif dim == 1:
            for x in d:
                X.append([self.convert(i) for i in x[:-1]])
                y.append(self.convert_label(x[-1]))
            X = np.array(X)
            y = np.array(y)
        else:
            X, y = np.array([[self.convert(i) for i in x] for x in d]), None
        return X, y

    def read_data_json(self, fname, iterable=None):
        """Read a file with one JSON object per line: the inputs and the target.

        Each object gives the inputs as keys that are the index of the input
        (a string), and the target under ``klass`` (environment variable
        ``KLASS`` changes the name). Inputs that are missing are zero. An
        object with the key ``vec`` (environment variable ``VEC``) is read
        by ``read_data_json_vec``.

        Parameters
        ----------
        fname : str
            File name.
        iterable : iterable of dict, optional
            Objects to read instead of the file.

        Returns
        -------
        tuple
            ``(X, y)``: ``X`` is a list with one array per input and ``y``
            an array with the targets, or ``None`` if there are none.
        """
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if iterable is None:
            iterable = json_iterator(fname)
        VEC = os.getenv('VEC', 'vec')
        for row, a in enumerate(iterable):
            if VEC in a:
                return self.read_data_json_vec(fname)
            if X is None:
                X = [list() for i in range(self._num_terms(a))]
            for k, v in a.items():
                try:
                    k = int(k)
                    X[k].append((row, self.convert(v)))
                except ValueError:
                    if k == dependent:
                        y.append(self.convert_label(v))
        num_rows = row + 1
        X = [index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y

    def read_data_json_vec(self, fname, iterable=None):
        """Read JSON lines whose inputs are in a list under ``vec``.

        Without ``vecsize`` the list holds the value of every input in
        order; with ``vecsize`` it holds ``[index, value]`` pairs of a
        vector of that size (sparse). The target is under ``klass``. It
        returns ``(X, y)`` as ``read_data_json``.
        """
        X = None
        y = []
        dependent = os.getenv('KLASS')
        if dependent is None:
            dependent = 'klass'
        if iterable is None:
            iterable = json_iterator(fname)
        VEC = os.getenv('VEC', 'vec')
        for row, a in enumerate(iterable):
            vec = a[VEC]
            try:
                vecsize = a['vecsize']
            except KeyError:
                vecsize = len(vec)
                vec = enumerate(vec)
            if X is None:
                X = [list() for i in range(vecsize)]
            for k, v in vec:
                k = int(k)
                X[k].append((row, self.convert(v)))
            try:
                y.append(self.convert_label(a[dependent]))
            except KeyError:
                pass
        num_rows = row + 1
        X = [index_data(x, num_rows) for x in X]
        if len(y) == 0:
            y = None
        else:
            y = np.array(y)
        return X, y


class RandomParameterSearch(object):
    """Random sample of the space of parameters of EvoDAG.

    The space is the product of the lists of values of ``parameter_values.json``
    (``EvoDAG/conf``). Iterating over the object yields ``npoints`` different
    configurations, drawn at random without repetition and without those
    that are not consistent (see ``constraints``). It is what the
    ``EvoDAG-params`` command uses. The space is enormous and ``len`` gives
    its size, so do not build a list from it directly; use a loop or
    ``itertools.islice``.

    Parameters
    ----------
    params : dict, optional
        Maps each parameter to the list of its values (default: the content
        of ``parameter_values.json``).
    npoints : int, optional
        Number of configurations to yield (default 1468).
    training_size : int, optional
        Number of training examples; the values of ``popsize`` and
        ``early_stopping_rounds`` larger than it are not used.
    seed : int, optional
        Seed of the sampling (default 0).

    Examples
    --------
    >>> from itertools import islice
    >>> from EvoDAG.utils import RandomParameterSearch
    >>> search = RandomParameterSearch(npoints=3, training_size=500, seed=0)
    >>> configurations = list(islice(search, 3))
    >>> len(configurations)
    3
    >>> all(c["popsize"] <= 500 for c in configurations)
    True
    """
    def __init__(self, params=PARAMS,
                 npoints=1468,
                 training_size=5000,
                 seed=0):
        self._training_size = training_size
        self.popsize_constraint(params)
        self._params = sorted(params.items())
        assert len(self._params)
        self._params.reverse()
        self._len = None
        self._npoints = npoints
        self._seed = seed
        self.fix_early_popsize()

    def popsize_constraint(self, params):
        """Keep, in ``params['popsize']``, the values not larger than the training size."""
        try:
            params['popsize'] = [x for x in params['popsize'] if x <= self._training_size]
        except KeyError:
            pass

    def fix_early_popsize(self):
        """Add the training size as a value of ``popsize`` or ``early_stopping_rounds`` when none is left."""
        try:
            popsize = [x for x in self._params if x[0] == 'popsize'][0]
            if len(popsize[1]) == 0:
                popsize[1].append(self._training_size)
        except IndexError:
            pass
        try:
            early = [x for x in self._params if x[0] == 'early_stopping_rounds'][0]
            early_min = min(early[1])
            if early_min > self._training_size:
                early[1].append(self._training_size)
        except IndexError:
            pass

    def __len__(self):
        if self._len is None:
            _ = np.array([len(x[1]) for x in self._params], dtype=np.uint)
            _ = np.prod(_)
            assert _ >= 0
            self._len = _
        return self._len

    def __getitem__(self, key):
        res = {}
        lens = [len(x[1]) for x in self._params]
        for l, k_v in zip(lens, self._params):
            k, v = k_v
            key, residual = divmod(key, l)
            res[k] = v[int(residual)]
        return res

    def constraints(self, k):
        """Whether configuration ``k`` is consistent.

        A generational population needs ``early_stopping_rounds`` of at
        least ``popsize``, and ``early_stopping_rounds`` cannot exceed the
        number of training examples.
        """
        try:
            if k['population_class'] == 'Generational' and\
               k['early_stopping_rounds'] < k['popsize']:
                return False
            if k['early_stopping_rounds'] > self._training_size:
                return False
        except KeyError:
            return True
        return True

    def __iter__(self):
        np.random.seed(self._seed)
        m = {}
        _len = self.__len__()
        npoints = self._npoints if _len > self._npoints else _len
        while npoints:
            k = np.round(np.random.uniform(0, _len)).astype(np.uint)
            if len(m) == _len:
                return
            while k in m:
                k = np.random.randint(_len)
            m[k] = 1
            p = self[k]
            if self.constraints(p):
                npoints -= 1
                yield p

    @staticmethod
    def process_params(a):
        """Convert a configuration into the arguments of the engine.

        The keys that are the name of a function (``Add``, ``Mul``, ...)
        become the ``function_set``: a boolean switches the function on or
        off and an integer sets its number of arguments (0 removes it). The
        other keys are returned unchanged.

        Parameters
        ----------
        a : dict
            A configuration, as yielded by the search or read from a file.

        Returns
        -------
        dict
            The keyword arguments for ``EvoDAG.EvoDAG``.

        Examples
        --------
        >>> from EvoDAG.utils import RandomParameterSearch
        >>> args = RandomParameterSearch.process_params({"Add": 10, "Mul": 0, "Sin": True, "popsize": 100})
        >>> [f.__name__ for f in args["function_set"]]
        ['Add', 'Sin']
        >>> args["popsize"]
        100
        """
        from EvoDAG import EvoDAG
        fs_class = {}
        function_set = []
        for x in EvoDAG()._function_set:
            fs_class[x.__name__] = x
        args = {}
        for k, v in a.items():
            if k in fs_class:
                if not isinstance(v, bool):
                    fs_class[k].nargs = v
                if v:
                    function_set.append(fs_class[k])
            else:
                args[k] = v
            fs_evo = EvoDAG()._function_set
            fs_evo = filter(lambda x: x in function_set, fs_evo)
            args['function_set'] = [x for x in fs_evo]
        return args
