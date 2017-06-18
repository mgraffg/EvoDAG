# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nose.tools import assert_almost_equals


def create_problem_node2(nargs=4, seed=0):
    from EvoDAG import RootGP
    from test_root import X, cl
    import numpy as np
    gp = RootGP(generations=1, popsize=nargs,
                multiple_outputs=True, seed=seed)
    X1 = np.concatenate((X, np.atleast_2d(np.zeros(X.shape[0])).T), axis=1)
    for i in range(10, 20):
        X1[i, -1] = 1
    gp.X = X1
    gp.Xtest = X1
    y = cl.copy()
    gp.nclasses(y)
    gp.y = y
    return gp, [gp.X[x] for x in range(nargs)]


def test_mean_std_pk():
    import numpy as np
    from EvoDAG.naive_bayes import NaiveBayes
    gp, args = create_problem_node2(nargs=5, seed=0)
    mask = np.array(gp._mask_ts.sign().full_array(), dtype=np.bool)
    klass = np.array(gp._y_klass.full_array())[mask]
    unique_klass = np.unique(klass)
    naive_bayes = NaiveBayes(mask=gp._mask_ts, klass=gp._y_klass,
                             nclass=gp._labels.shape[0])
    for v in args:
        var = np.array(v._eval_tr.full_array())[mask]
        mean = [np.mean(var[k == klass]) for k in unique_klass]
        std2 = [np.var(var[k == klass]) for k in unique_klass]
        p_klass = [(k == klass).mean() for k in unique_klass]
        _mean, _std2, _klass = naive_bayes.coef(v._eval_tr)
        for a, b in zip(mean, _mean):
            assert_almost_equals(a, b)
        for a, b in zip(std2, _std2):
            assert_almost_equals(a, b)
        for a, b in zip(p_klass, _klass):
            assert_almost_equals(a, b)


def test_MN_Nc_pk():
    import numpy as np
    from EvoDAG.naive_bayes import NaiveBayes
    gp, args = create_problem_node2(nargs=5, seed=0)
    mask = np.array(gp._mask_ts.sign().full_array(), dtype=np.bool)
    klass = np.array(gp._y_klass.full_array())[mask]
    unique_klass = np.unique(klass)
    naive_bayes = NaiveBayes(mask=gp._mask_ts, klass=gp._y_klass,
                             nclass=gp._labels.shape[0])
    for v in args:
        var = np.array(v._eval_tr.full_array())[mask]
        Nc = [np.sum([x for x in var[k == klass] if x > 0]) for k in
              unique_klass]
        if len(Nc) > len([x for x in Nc if x > 0]):
            continue
        N = np.sum(Nc).reshape(-1, 1)
        w = np.log((Nc / N)[0])
        p_klass = np.log([(k == klass).mean() for k in unique_klass])
        _w, _klass = naive_bayes.coef_MN(v._eval_tr)
        for a, b in zip(w, _w):
            assert_almost_equals(a, b)
        for a, b in zip(p_klass, _klass):
            assert_almost_equals(a, b)
