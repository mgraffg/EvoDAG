# Copyright 2016 Mario Graff Guerrero

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
cimport numpy as np
from SparseArray.sparse_array cimport SparseArray


cpdef compute_weight(list r, SparseArray ytr, mask):
    """Returns the weight (w) using OLS of r * w = gp._ytr """
    cdef Py_ssize_t i, j, size=len(r)
    cdef SparseArray f, ri, rj
    cdef np.ndarray[np.double_t, ndim=2] A = np.empty((len(r), len(r)), dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] b = np.array([(f * ytr).sum() for f in r], dtype=np.double)
    r = [x for x in r]
    for i in range(size):
        r[i] = r[i] * mask
        ri = r[i]
        for j in range(i, size):
            rj = r[j]
            A[i, j] = (ri.mul(rj)).sum()
            A[j, i] = A[i, j]
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        return None
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return coef
