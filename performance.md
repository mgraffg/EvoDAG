# Performance #

The next table presents the average performance in terms of the
balance error rate (BER) of PSMS, auto-sklearn, SVM, and EvoDAG on nine classification
problems (these benchmarks can be found:
[matlab](http://theoval.cmp.uea.ac.uk/matlab/benchmarks) and [text](http://ws.ingeotec.mx/~mgraffg/classification)).
  

The best performance among each classification dataset is in
bold face to facilitate the reading. It can be observed from the table
that PSMS obtained the best performance on three datasets, SVM
on one dataset, and EvoDAG (0.10.6) obtained the
best performance in the rest of the datasets (five). One
characteristic that caught our attention is the high confidence
intervals of auto-sklearn (not shown here), it is one order of magnitude higher than
the other systems. Analyzing the predictions performed by
auto-sklearn, it is found that in some of the trails the algorithm
predicts only one class, obtaining, consequently, the worst possible
performance, i.e., BER equals 50. This behaviour, clearly, can be
automatically spotted, and, one possible solution could be as simple
as execute auto-sklearn again on that particular case. Nonetheless, we
decided to keep auto-sklearn without modifications.

The table also presents the performance of EvoDAG (0.10.7), being the
only difference that EvoDAG (0.10.7) does not optimise the parameters,
thus it is only necessary to train the model as:

```bash   
EvoDAG-train -m model.evodag -n 100 -u 4 iris.data 
```
assuming the dataset is *iris.data*. 

|Dataset|[PSMS](http://www.jmlr.org/papers/v10/escalante09a.html)|[auto-sklearn](https://github.com/automl/auto-sklearn)|[SVC(sklearn)](http://scikit-learn.org/stable/)|EvoDAG - Inductive| EvoDAG - Shuffle | EvoDAG - Transductive|
| ----- | ---------------------------------------------: | --------------------------------------------: | -----------------------------------: | -----------: |-----------: |-----------: |
|banana          |     **11.08**      |  28.00  |11.27 | 12.88 | 12.43 | 11.93|
|thyroid        |      **4.32**      |  23.38  |  6.13  |  8.56 | 8.21 |7.79 |
|diabetis        |     27.06   |   37.65  |  26.65  |  24.85 | **24.82** | 24.87 |
|heart             |   20.69        |  27.69  |  18.12  |  17.24 | 16.87 | **16.86**|
|ringnorm       |      7.98      |  15.49  |  **1.96**  |  2.93 | 2.71 |2.00 |
|twonorm       |       3.09      |  20.87  |  2.90  |  3.03 | 2.99 | **2.64** |
|german         |      30.10    | 39.45  |  29.00  | 28.71 | **28.64** | 28.83 |
|image         |       **2.90** | 21.29  |  3.32  | 4.07 | 3.40 | 3.42 |
|waveform      |       12.80   | 22.67  |  **10.62**  |  10.88 | 10.79 |10.69 |

