# Selection heuristics experiments

This directory has the scripts that run again the experiments of

> Sánchez, C. N. and Graff, M. *Selection Heuristics on Semantic Genetic Programming for Classification Problems.* Evolutionary Computation, 2022. DOI [10.1162/evco_a_00297](https://doi.org/10.1162/evco_a_00297)

with the current code. The article evaluates EvoDAG on 30 classification problems of the UCI repository (its Table 1).

## Where the files go

Everything the experiments read and write lives in a directory that git ignores: `experiments/` at the root of the repository, or the directory named by the environment variable `EVODAG_EXPERIMENTS`, or the one given with `--root`. Inside it, `raw/<name>/` has the downloaded archives, `data/<name>.npz` the prepared datasets and `data/manifest.json` the manifest. Nothing in it is committed.

## The datasets

```bash
python benchmarks/ecj2022/datasets.py                 # download and prepare the 30 datasets
python benchmarks/ecj2022/datasets.py --datasets iris wine
python benchmarks/ecj2022/datasets.py --list          # the manifest as a Markdown table
python benchmarks/ecj2022/datasets.py --force         # prepare again
```

The script downloads the archive of each dataset from `https://archive.ics.uci.edu` (its sha256 is verified against the digests kept in the script), extracts it and writes `data/<name>.npz` with `X_train`, `X_test` (`float64`), `y_train`, `y_test` (`int64`, the classes renumbered `0 .. k-1` in sorted order) and `classes` (the original labels). A dataset that is already prepared is skipped, and one that fails does not stop the others (the exit status is 1 at the end). It needs only the standard library and NumPy.

The original preprocessing of the article is lost (it says only that the UCI split was used when there is one and that otherwise the data were split at random, 70% and 30%), so these rules are applied to every dataset:

- A column is numeric when all its values are numbers; otherwise it is categorical and is encoded with the position of each value in the sorted list of its values (ordinal encoding).
- The missing values (`?`, `na`, `NA`, `nan`, `NaN`, empty) become the median of the training part in a numeric column and the category `?` in a categorical one. Constant columns are kept.
- Without an official split, a stratified 70/30 split with `numpy.random.RandomState(0)` is used, so it is the same on every run.
- The classes of agaricus-lepiota, drug-consumption and ml-prove are the last column, the only reading that gives the numbers of Table 1.

`manifest.json` records, for each dataset, its sizes, the entropy of the classes, the split used and the five numbers of Table 1 of the article, whether it matches them (sizes within two examples, same number of variables and classes) and a list of the differences. Several totals of the article are one or two rows short of the UCI files.

## The experiments

For each of the 30 datasets the article reports the macro-F1 on the test set and the training time per training example of

- 14 combinations of parent and negative selection,
- 15 selection schemes (angle-driven selection and novelty search included),
- EvoDAG against the 16 classifiers of scikit-learn with default parameters (TPOT and auto-sklearn are not reproduced).

The scripts, all of them writing only below the experiments directory:

| Script | What it does |
|---|---|
| `configs.py` | the 22 named configurations, the three tables of the article and the common parameters |
| `run.py` | runs one unit: a dataset with a configuration, or with the scikit-learn classifiers |
| `submit.py` | submits the units to Slurm with the cores and memory of each dataset, and shows the state |
| `summary.py` | turns the results into the tables of the article |

### Configurations

The name is `<parent>-<negative>[--<first>][-all]` (`-all` is the `*` of the article). Parent selection: `fit` fitness, `rnd` random, `agr` agreement, `sim` cosine similarity, `prs` Pearson, `ads` angle-driven, `nvs` novelty search. Negative selection: `fit` or `rnd`. First parent: `rnd` (default) or `fit`. Functions: the heuristic is applied to the tailored functions, or to all of them (`-all`).

| Name | Article name | Parent | Negative | First | Functions |
|---|---|---|---|---|---|
| `fit-fit` | fit-fit | fit | fit | – | – |
| `rnd-rnd` | rnd-rnd | rnd | rnd | – | – |
| `agr-rnd` | agr-rnd, acc-rnd, agr-rnd--rnd | agr | rnd | rnd | tailored |
| `agr-fit` | agr-fit, agr-fit--rnd | agr | fit | rnd | tailored |
| `sim-rnd` | sim-rnd, sim-rnd--rnd | sim | rnd | rnd | tailored |
| `sim-fit` | sim-fit, sim-fit--rnd | sim | fit | rnd | tailored |
| `prs-rnd` | prs-rnd, prs-rnd--rnd | prs | rnd | rnd | tailored |
| `prs-fit` | prs-fit, prs-fit--rnd | prs | fit | rnd | tailored |
| `agr-rnd-all` | agr-rnd* | agr | rnd | rnd | all |
| `agr-fit-all` | agr-fit* | agr | fit | rnd | all |
| `sim-rnd-all` | sim-rnd* | sim | rnd | rnd | all |
| `sim-fit-all` | sim-fit* | sim | fit | rnd | all |
| `prs-rnd-all` | prs-rnd* | prs | rnd | rnd | all |
| `prs-fit-all` | prs-fit* | prs | fit | rnd | all |
| `agr-fit--fit` | agr-fit--fit | agr | fit | fit | tailored |
| `agr-rnd--fit` | agr-rnd--fit | agr | rnd | fit | tailored |
| `ads-rnd` | ads-rnd--rnd, ads-rnd | ads | rnd | rnd | tailored |
| `ads-fit--fit` | ads-fit--fit | ads | fit | fit | tailored |
| `ads-rnd-all` | ads-rnd--rnd* | ads | rnd | rnd | all |
| `ads-fit--fit-all` | ads-fit--fit* | ads | fit | fit | all |
| `nvs-rnd` | nvs-rnd | nvs | rnd | – | tailored |
| `default` | (not in the article) | released defaults | | | |

Every configuration but `default` also has `fitness_function="macro-F1"`, `random_generations=0` and `time_limit=None`; the rest comes from `EvoDAG/conf/default_parameters.json` (population 4000, early stopping after 4000 evaluations). `default` is the released default, only without the time limit. `--quick` replaces the population and the early stopping by 50 and the 30 members by 3: it tests the pipeline and its numbers are not results.

### Layout of the experiments directory

```
experiments/data/<dataset>.npz, manifest.json      (datasets.py)
experiments/results/<dataset>/<config>/member-<i>.model.gz, member-<i>.json, result.json
experiments/results/<dataset>/sklearn/<Name>.json
experiments/results-quick/                          (the same, for --quick)
experiments/logs/                                   (Slurm output, <dataset>.tasks.json, submissions.jsonl)
```

### Running a unit by hand

```bash
python benchmarks/ecj2022/run.py --dataset iris --config agr-rnd [--workers 4] [--members 30] [--quick]
python benchmarks/ecj2022/run.py --dataset iris --sklearn [--sklearn-timeout HOURS]
```

An ensemble of 30 members is trained, one per seed `0 .. 29`, each on a random half of the training set, with one core (the BLAS and OpenMP thread variables are set to 1) and without a test set, so that the memory is that of the training examples only. Each member is a fresh process; its files are written with the JSON last, so a unit that is interrupted resumes at the member that was missing, and one that is done does nothing. `result.json` has the macro-F1 of the majority vote (`macro_f1`, the prediction of `EvoDAGE.predict`) and of the median decision function (`macro_f1_median`), the sum of the training times of the members (`fit_seconds`, the one-core time of the article) divided by the training examples (`fit_seconds_per_sample`), and the peak memory. The scikit-learn classifiers are fitted with default parameters and `random_state=0` on the prepared data without scaling, in a child process with a timeout (24 hours by default); an error or a timeout is recorded as a result, like the `-` cells of the article.

### Slurm

```bash
python benchmarks/ecj2022/submit.py --dry-run          # print the sbatch commands, submit nothing
python benchmarks/ecj2022/submit.py --smoke            # the five smallest datasets, every configuration
python benchmarks/ecj2022/submit.py                    # the full campaign
python benchmarks/ecj2022/submit.py --status           # finished, partial, failed, in the queue
python benchmarks/ecj2022/summary.py                   # the tables
```

(use the interpreter of the environment, `.venv/bin/python`: the jobs use the one that submits them). There is one job array per dataset with one task per pending configuration and a separate one-task job for the scikit-learn classifiers (16000 MB). The resources of the array follow the size of the dataset: a member needs `ceil(1.5 * 8 * popsize * n_classes * n_train / 1e6) + 1500` MB with `popsize = 4000`; a task runs `workers = max(1, min(--max-workers, --max-mem-mb // member_mb))` members at the same time (defaults 8 and 120000) and asks for `workers` cores and `workers * member_mb` MB. `--max-parallel` limits the tasks of an array that run together, `--datasets` and `--configs` restrict the campaign. Submitting again is safe: finished units are left out, the others resume at the member level, and a dataset that still has jobs in the queue is skipped. The peak memory of every member is recorded, to check the factor of the formula.

`summary.py` prints, for the three tables, the macro-F1 with the rank of each system on each dataset (ties share the best rank), the average rank over the datasets where every system has a result, the Friedman p-value and, for the classifiers, the training time per training example. `--median` uses the median decision function, `--quick` reads the quick results, and `--paper-tex PATH` (for example `../articulos/ECJ2022/paper.tex` next to the repository) reads the numbers of the tables of the article from its LaTeX source and prints the differences; no file of the article is ever copied.

### What differs from the article

- The preprocessing of the datasets (see above): the original is lost.
- EvoDAG's prediction is the majority vote of the members (the median is recorded too), while the article says "average of the predictions".
- The scikit-learn defaults are those of the installed version, and TPOT and auto-sklearn are not run.
- It is another machine and the time is measured on one core of it: times only compare among themselves.

### Cost

The article reports 0.5 to 5 seconds per training example for an ensemble of 30 members. The 30 datasets have about 600 000 training examples in total, so one configuration is on the order of hundreds of CPU hours and the 22 of them thousands. The largest datasets (census-income, dota2, aps-failure, sensorless, adult, bank) dominate. The full campaign is launched by hand with the command above.

## Profile

The fit of one member was profiled with `profile_fit.py` (cProfile, one core, `--max-seconds 900`, which no profile reached), and the engine was optimised in steps that must give **the same bits**: `parity.py` records 106 small fits (the 22 configurations, the released defaults, two ensembles, extreme values, a regression with the agreement heuristic, a generational population and a test set given to `fit`) before a change and `check` compares them again exactly, including a digest of every individual created, and `tests/test_golden.py` must pass without a skipped test. Measured on 2026-11-17 on one core of a compute node of the cluster (the benchmark runs were on the node `geoint0`; the node of each profile was not recorded; the profiles ran twelve at a time on two nodes, the same load before and after, and the 900-second limit was not reached by any), one member of an ensemble (`seed=0`, the parameters of the configuration without `--quick`).

```bash
python benchmarks/ecj2022/profile_fit.py --dataset optdigits --config agr-rnd --max-seconds 900 --output /tmp/optdigits.pstats
python benchmarks/ecj2022/parity.py record --output /tmp/parity    # before a change
python benchmarks/ecj2022/parity.py check --reference /tmp/parity  # after it
```

| Dataset | Configuration | Individuals | Fit before (s) | Fit after (s) | Speed-up | Top functions before (own time, s) |
|---|---|---:|---:|---:|---:|---|
| optdigits | `agr-rnd` | 5462 | 36.8 | 19.4 | 1.90x | base.py _get_args_orthogonal 15.1s; node.py compute_weight 9.2s; node.py eval 1.9s |
| optdigits | `fit-fit` | 18428 | 89.8 | 75.6 | 1.19x | node.py compute_weight 42.4s; node.py _raw_outputs 5.5s; node.py eval 5.5s |
| optdigits | `prs-rnd` | 6894 | 146.9 | 84.4 | 1.74x | base.py _masked_outputs 54.1s; base.py _pearson_value 33.2s; node.py compute_weight 11.4s |
| optdigits | `sim-rnd` | 4058 | 54.9 | 36.6 | 1.50x | base.py _masked_outputs 33.4s; base.py _cosine_value 6.9s; node.py compute_weight 3.9s |
| optdigits | `ads-rnd` | 7032 | 542.1 | 357.0 | 1.52x | base.py _masked_outputs 260.3s; base.py _angle_value 123.2s; base.py _relative_semantics 43.9s |
| optdigits | `nvs-rnd` | 5813 | 29.5 | 27.0 | 1.09x | node.py compute_weight 8.6s; base.py _novelty_value 3.5s; node.py eval 2.3s |
| optdigits | `default` | 7149 | 55.1 | 30.7 | 1.79x | base.py _get_args_orthogonal 20.2s; node.py compute_weight 17.1s; node.py eval 2.5s |
| magic04 | `agr-rnd` | 18584 | 175.3 | 76.3 | 2.30x | node.py compute_weight 85.9s; base.py _get_args_orthogonal 60.0s; node.py _mul 5.6s |
| magic04 | `fit-fit` | 12803 | 97.9 | 50.8 | 1.93x | node.py compute_weight 75.3s; node.py _mul 4.0s; node.py _raw_outputs 3.5s |
| magic04 | `nvs-rnd` | 7099 | 59.8 | 42.2 | 1.41x | node.py compute_weight 32.6s; base.py _novelty_value 11.3s; node.py _mul 2.2s |
| letter-recognition | `agr-rnd` | 3499 | 100.1 | 65.0 | 1.54x | base.py _get_args_orthogonal 39.9s; node.py eval 23.2s; node.py compute_weight 8.8s |
| adult | `agr-rnd` | 8263 | 167.7 | 67.1 | 2.50x | node.py compute_weight 70.6s; base.py _get_args_orthogonal 69.8s; node.py _mul 7.0s |

The fit times are those under cProfile and include its overhead; the number of individuals created is identical before and after in every case. Without the profiler, on an idle node: `magic04` `agr-rnd` 157 s to 68 s and `optdigits` `agr-rnd` 31 s to 21 s with the first changes, and `optdigits` `fit-fit` 79 s to 64 s. The peak memory fell too (the maximum resident set of the process, before and after): `optdigits` `agr-rnd` 1.66 GB to 1.38 GB, `magic04` `agr-rnd` 3.75 GB to 1.29 GB, `optdigits` `fit-fit` 5.5 GB to 1.3 GB, because the engine kept the outputs of every individual ever created (see below).

`benchmarks/bench.py`, three runs before and three after on the same node (the best of each is the one that counts; the spread between the three repeats was at most 2.4% before and 2.0% after), the fastest `fit_min` of the three seeds in seconds (`benchmarks/results-profiled.json` is the best "after" run; the recorded column is `benchmarks/results-optimized.json`, an earlier version and run):

| Case | Recorded | Before | After | Speed-up |
|---|---:|---:|---:|---:|
| classification_1000x20_3c | 0.71 | 0.73 | 0.66 | 1.11x |
| classification_10000x50_5c | 11.49 | 11.25 | 10.44 | 1.08x |
| regression_1000x20 | 0.61 | 0.62 | 0.37 | 1.67x |
| ensemble_1000x20_4 | 3.66 | 5.11 | 2.46 | n/a |

The ensemble row has no speed-up on purpose. `bench.py` times **one** `EvoDAGE` fit there (no minimum over seeds, unlike the other rows), and the recorded column, from an earlier tree and run, differs from the "before" column by 40% on what should be the same work, so that case cannot be compared across runs. What the three repeats show is only that they agree with each other (5.11 to 5.21 s before, 2.46 to 2.47 s after, the same individuals created); do not read it as a measured gain of the ensemble.

The training accuracy and the relative squared error of the benchmark are identical before and after, and so are all 106 cases of `parity.py` and the size of a pickled model.

### Optimisations kept

Each one is covered by a test that compares the fast path with the plain computation (`tests/test_optimisations.py`, `tests/test_linalg_solve.py`) and by the parity check.

- **Predicted labels of an individual are remembered** (`Variable.labels`, used by the agreement heuristic of `EvoDAG/base.py`). The labels (`argmax` of the outputs, or the sign in regression) were recomputed for every candidate of every tournament, about 60 000 to 200 000 times per fit. They depend only on `hy`, which is not changed while the individual is in the population, and the cache is dropped whenever `hy` is replaced or released, so every value is the one computed before. It is not stored in the models (their size in bytes is unchanged). It costs one small array per individual in the population (the labels, as long as the training set), and the cache is dropped with the outputs, so it is bounded by the population like them. It is the largest gain for many classes (`optdigits` `agr-rnd`: 15 s of 37).
- **The train accuracy of the agreement heuristic does not count the validation part** (`Score._accuracy_index`). `Score.accuracy` also counted the rest of the examples, and only the first number was used. The counts are integers, so the quotient is the same one; the method is private because the golden test asks every public kernel to be recorded.
- **`compute_weight` computes the dot products of the Gram matrix four at a time** (`EvoDAG/linalg_solve.pyx`) when all the vectors are finite. Every sum adds the same numbers in the same order as `dot`, so the results are identical; four independent chains stop waiting for the latency of the additions. The guard of `dot` is not needed then: when everything is finite a product with a zero is a zero and adding it changes nothing (the sum is never `-0.0`). The columns are taken by their address, once. With a non-finite value the old guarded path is used. `tests/test_linalg_solve.py` compares it with a plain Python version of the whole algorithm, signs of zero included. It halves the time of an `Add` with many arguments, the main cost of the datasets with two classes and many examples. `dot` itself keeps its original branch: a branch-free version cannot be fused into a multiply-add by the compiler, so on builds that contract floating point expressions it would not give the bits of the previous code.
- **`array_utils.mul` has no branch** (the product where both factors are not zero and `+0.0` elsewhere, as before). The masks have random zeros, so the branch was unpredictable: 25 to 10 microseconds for 3823 values. It helps the cosine, Pearson and angle heuristics, which multiply every output by the mask for every candidate.
- **Pearson and angle selection**: the quantities that depend only on the reference (its centred vector and norm) are computed once per reference instead of once per candidate, `u.mean()` is `np.add.reduce(u) / u.size` (the same sum and division without the method overhead), and `np.clip` of a number is `min(max(...))`, which gives the same value (also for NaN). The regression index of the heuristic is computed once per mask.
- **A node does not keep its density cache once it is released** (`BasePopulation.clean`). The cache stores a reference to the outputs, so the outputs of every individual ever created stayed in memory, not only those of the population. This is a memory fix, not a speed one, and it is what keeps the labels cache from raising the peak memory: the memory the experiments request (`submit.py`, `1.5 * 8 * popsize * classes * examples`) was sized for a population and is now respected.

### Candidates tried and rejected

- **One Gram matrix for all the outputs of a node** (Context of the issue): it does not apply. The masks of the outputs are different (each class has its own balanced sample, `bagging_fitness.py`), and for `Add` and the functions of one argument the arguments of each output are different (the output `k` of each argument). Only the right-hand side is shared, which is nothing.
- **Eight dot products at a time** instead of four: no gain (the time is bounded by the loads).
- **A branch-free `dot`** (also without the four chains): it helps on sparse vectors only, and it is not bit-identical on builds that fuse multiply-adds, so it was dropped.
- **Remembering the masked outputs of an individual** for the cosine, Pearson and angle heuristics (`_masked_outputs` is the first function for them): it would double the memory of the population, which the memory request of `submit.py` does not allow for. A smarter layout is a decision for the maintainer.
- **Replacing `gauss_jordan` by a LAPACK solve**, changing the order of the sums, `np.random` in blocks: not allowed (they change bits).

### What remains the bottleneck

`compute_weight` (the Gram matrix and the elimination of an `Add` with up to 60 arguments, which is a quadratic number of dot products of vectors that each depend on the mask), then `_masked_outputs` and `_angle_value` of the heuristics that look at the semantics (`ads-rnd` is slow because each argument draws many candidates), the `eval` of the Naive Bayes and Centroid nodes for many classes (`letter-recognition`), and `_novelty_value` (the work is one gather and one sum per candidate). Going further needs a blocked kernel for the Gram matrix that keeps the order of every sum (a SIMD layout across the dot products), or an algorithmic change (fewer arguments, the Gram matrix updated incrementally), which is a decision for the maintainer.
