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

The scripts (those that run the experiments write only below the experiments directory; `profile_fit.py` and `parity.py` belong to the [Profile](#profile) section):

| Script | What it does |
|---|---|
| `datasets.py` | downloads and prepares the 30 datasets |
| `configs.py` | the 21 named configurations, the tables of the article and the common parameters |
| `run.py` | runs one unit: a dataset with a configuration, or with the scikit-learn classifiers |
| `submit.py` | submits the units to Slurm with the cores and memory of each dataset, shows the state and waits for the queue |
| `summary.py` | turns the results into the tables of the article |
| `profile_fit.py` | profiles the fit of one member (see Profile) |
| `parity.py` | records and checks the bit-level parity of the engine (see Profile) |

The measure is the macro-F1 of the article, on the test set.

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

Every configuration also has `fitness_function="macro-F1"`, `random_generations=0` and `time_limit=None`; the rest comes from `EvoDAG/conf/default_parameters.json` (population 4000, early stopping after 4000 evaluations). `--quick` replaces the population and the early stopping by 50 and the 30 members by 3: it tests the pipeline and its numbers are not results.

The configuration `agr-rnd` is the default of classification of the library (`EvoDAG/conf/default_parameters.json`).

### Layout of the experiments directory

```
experiments/data/<dataset>.npz, manifest.json      (datasets.py)
experiments/results/<dataset>/<config>/member-<i>.model.gz, member-<i>.json, result.json, predictions.npz
experiments/results/<dataset>/sklearn/<Name>.json, <Name>.npz
experiments/results-quick/                          (the same, for --quick)
experiments/logs/                                   (Slurm output, <dataset>.<time>.tasks.json, submissions.jsonl)
```

### Running a unit by hand

```bash
python benchmarks/ecj2022/run.py --dataset iris --config agr-rnd [--workers 4] [--members 30] [--quick]
python benchmarks/ecj2022/run.py --dataset iris --task-file logs/iris.<time>.tasks.json [--task-index N]
python benchmarks/ecj2022/run.py --dataset iris --sklearn [--sklearn-timeout HOURS]
python benchmarks/ecj2022/run.py --dataset iris --sklearn --classifiers GaussianNB SVC
python benchmarks/ecj2022/run.py --dataset iris --sklearn --classifier-index 2
```

An ensemble of 30 members is trained, one per seed `0 .. 29`, each on a random half of the training set, with one core (the BLAS and OpenMP thread variables are set to 1) and without a test set, so that the memory is that of the training examples only. Each member is a fresh process, started with the `fork` start method when the platform has it (`--workers` of them at a time); a member killed by a signal (the out-of-memory killer, a crash of a kernel) is recorded as a failure in `member-<i>.error.txt` ("process died with exit code ...") and the unit ends with the exit status 1 instead of waiting for it for ever. The files of a member are written with the JSON last, so a unit that is interrupted resumes at the member that was missing, and one that is done does nothing. A `result.json` counts as the finished unit only when its `members` is the number asked for (30, 3 with `--quick`): a run with `--members 2` leaves a result that `submit.py` and `summary.py` ignore, and running the unit again without it reuses the two members and fits the rest.

`--task-file` is the JSON list of configurations that `submit.py` writes for an array (`<dataset>.<time>.tasks.json`, one file per submission) and `--task-index` (default `$SLURM_ARRAY_TASK_ID`) the position of this task in it. `--classifiers` and `--classifier-index` (the position, 0 to 15, in the table of classifiers; the jobs pass `$SLURM_ARRAY_TASK_ID`) select some of the sixteen classifiers; a result is skipped when it exists.

`result.json` has the macro-F1 of the majority vote (`macro_f1`, the prediction of `EvoDAG.predict`) and of the median decision function (`macro_f1_median`), the macro-recall of both (`macro_recall`, `macro_recall_median`), the accuracy, the sum of the training times of the members (`fit_seconds`, the one-core time of the article) divided by the training examples (`fit_seconds_per_sample`), the peak memory and the `environment` (versions, `host`, `git_commit` and `git_dirty`, whether the tree had uncommitted changes). `predictions.npz` holds the test labels `y_test`, the predictions of the majority vote `hy` and of the median `hy_median` (all `int64`) and the median decision function (`float32`), so that any other measure can be computed without a new run. Each `member-<i>.json` records the `host`. The scikit-learn classifiers are fitted with default parameters and `random_state=0` on the prepared data without scaling, in a child process with a timeout (24 hours by default); each writes `<Name>.npz` (`y_test`, `hy`) and then `<Name>.json` (with `macro_f1`, `macro_recall`); an error or a timeout is recorded as a result, like the `-` cells of the article, and has no `.npz`.

### Slurm

```bash
python benchmarks/ecj2022/submit.py --dry-run          # print the sbatch commands and the configurations of each array; writes and submits nothing
python benchmarks/ecj2022/submit.py --smoke            # the five smallest datasets, every configuration
python benchmarks/ecj2022/submit.py                    # every unit that has no result
python benchmarks/ecj2022/submit.py --status           # finished, partial, failed, in the queue
python benchmarks/ecj2022/submit.py --wait 600         # wait up to 600 s for the queue to empty, print one line, submit nothing
python benchmarks/ecj2022/summary.py                   # the tables
```

(use the interpreter of the environment, `.venv/bin/python`: the jobs use the one that submits them). The other options of `submit.py`: `--datasets` and `--configs` restrict the units, `--quick` uses the tiny parameters, `--no-sklearn` leaves the scikit-learn jobs out, `--max-workers` (8) and `--max-mem-mb` (120000) bound a task, `--max-parallel` limits the tasks of an array that run together and `--sklearn-timeout HOURS` (24) is passed to `run.py`.

There is one job array per dataset with one task per pending configuration, and for the scikit-learn classifiers one more array with one task per classifier that has no result (`--array` lists their positions, for example `0,3,7`; 1 core and 16000 MB each, so a dataset takes the time of its slowest classifier and not the sum of the sixteen). The datasets are submitted heaviest first (`n_train * n_classes`), so that the longest units start first. The resources of the EvoDAG array follow the size of the dataset: a member needs `ceil(MEMORY_FACTOR * 8 * popsize * n_classes * n_train / 1e6) + 1500` MB with `popsize = 4000` (`MEMORY_FACTOR` is in `submit.py`, see [Memory and time](#memory-and-time)); a task runs `workers = max(1, min(--max-workers, --max-mem-mb // member_mb))` members at the same time and asks for `workers` cores and `workers * member_mb` MB. **The cluster does not enforce the memory a job asks for** (`TaskPlugin=task/none`, no accounting): it is used only to place jobs, so a job that uses more than it asked for can exhaust a node, and the request has to be right. The task file is `logs/<dataset>.<YYYYmmdd-HHMMSS>.tasks.json`, a name per call, so that the file a running array reads is never rewritten. Submitting again is safe: finished units are left out, the others resume at the member level, and a dataset that still has jobs in the queue is skipped; if `squeue` fails (after two retries) nothing is submitted, because the busy datasets would not be known. The peak memory of every member is recorded.

`--wait SECONDS` checks the queue once a minute until no job of the experiments (name `ecj-...`) is left or the time runs out, and prints `queue: N jobs left; units finished F/T, failed X`; a failure of `squeue` counts as unknown and the wait goes on.

`summary.py` prints, for the three tables of the article, the macro-F1 with the rank of each system on each dataset (ties share the best rank), the average rank over the datasets where every system has a result, the Friedman p-value and, for the classifiers, the training time per training example. The datasets of the manifest without a result are listed as incomplete. `--median` uses the median decision function, `--quick` reads the quick results, `--output FILE` writes the Markdown to a file instead of the standard output, and `--paper-tex PATH` (for example `../articulos/ECJ2022/paper.tex` next to the repository) reads the numbers of the tables of the article from its LaTeX source and prints the differences in macro-F1 and, for the classifiers, the ratio of our training time per example to the article's; if the file or a table cannot be read one sentence says so. No file of the article is ever copied.

### What differs from the article

- The first individual of a heuristic is the first argument of the function, as the article describes. In the program that produced the article's results, and in EvoDAG up to version 0.18, it was used to compute the measure but not passed to the function; version 0.19 corrects it, so the results here differ from the article's numbers.
- The preprocessing of the datasets (see above): the original is lost. The 70/30 splits are ours (stratified, seed 0), and 12 datasets differ from the article's Table 1 by one or two rows, because the article's files were not available; `ad` has 1558 variables against 1557.
- In `ml-prove` the class is a deterministic function of five other columns (the other one-hot label columns), as in the article, where almost every system scores 1.0. `yeast` keeps the sequence name as an ordinal variable. Class labels that are all numbers are ordered numerically.
- EvoDAG's prediction is the majority vote of the members (the median is recorded too), while the article says "average of the predictions".
- The predictions of the test set are stored.
- The scikit-learn defaults are those of the installed version, and TPOT and auto-sklearn are not run.
- The training times are measured with 5 to 8 members sharing a node without CPU binding, so they include contention, and they compare only among themselves.

### Memory and time

The memory a job requests is not enforced by the cluster, so the request of `submit.py` has to be right. To size it, one member (`seed=0`, `--members 1`, one core, the parameters of the configuration) was fitted on the four datasets with the largest requests, with the cheapest kind of configuration (`agr-rnd`) and the most expensive (`ads-rnd`), each in its own Slurm job of 60000 MB. The ratio is `(max_rss_kb * 1024 - 1e9) / (8 * popsize * classes * n_train)` with `popsize = 4000`: the peak memory of the process, less 1 GB for the interpreter, the data and NumPy, over the size of the outputs of the population.

| Dataset | Configuration | `n_train` | Classes | Max RSS (GB) | Ratio | Fit (s) |
|---|---|---:|---:|---:|---:|---:|
| sensorless | `agr-rnd` | 40953 | 11 | 16.06 | 1.04 | 569 |
| census-income | `agr-rnd` | 199523 | 2 | 19.20 | 1.43 | 438 |
| letter-recognition | `agr-rnd` | 14000 | 26 | 10.72 | 0.83 | 65 |
| krkopt | `agr-rnd` | 19639 | 18 | 11.92 | 0.97 | 81 |
| sensorless | `ads-rnd` | 40953 | 11 | 14.75 | 0.95 | 8329 |
| census-income | `ads-rnd` | 199523 | 2 | 13.01 | 0.94 | 2933 |
| letter-recognition | `ads-rnd` | 14000 | 26 | 8.95 | 0.68 | 1927 |
| krkopt | `ads-rnd` | 19639 | 18 | 10.59 | 0.85 | 2481 |

The largest ratio is 1.43 (`census-income` with `agr-rnd`). With a margin of 1.25 and rounded up, `MEMORY_FACTOR` in `submit.py` is **1.8**, so a member of `census-income` is asked 24486 MB and a task runs 4 members at the same time. The `ads-rnd` members use less memory than the `agr-rnd` ones on these datasets. The configurations `prs`, `sim` and `nvs` were not measured on a large dataset, so their request rests on the same factor; the peak memory of every `member-<i>.json` shows whether it holds.

The fit times of one member on one core of a node with other jobs: 1 to 10 minutes with `agr-rnd` (65 s to 569 s here) and 0.5 to 2.3 hours with `ads-rnd` (1927 s to 8329 s, `sensorless` being the slowest of the four), and a unit has 30 members.

## Results

The results of the 21 configurations and the 16 classifiers on the 30 datasets, with the code of this repository. Every EvoDAG system is an ensemble of 30 members with macro-F1 as fitness, and its prediction is the majority vote of the members. The average ranks are over the 30 datasets (1 is best; ranks are computed within each table, ties share the best rank), sorted, with the p-value of the Friedman test. `summary.py` writes the per-dataset tables, with the rank of each system on each dataset, and the training time per training example of the classifiers, to `experiments/summary-macro-f1.md` (`experiments/summary-macro-f1-median.md` for the median decision function); they are not committed.

**Combinations of parent and negative selection, macro-F1:** `agr-fit` 4.9, `agr-rnd-all` 5.1, `agr-fit-all` 5.1, `rnd-rnd` 5.1, `agr-rnd` 5.6, `sim-fit` 5.8, `prs-fit-all` 5.8, `prs-rnd-all` 6.6, `sim-rnd` 6.7, `prs-fit` 7.1, `fit-fit` 7.3, `prs-rnd` 7.4, `sim-fit-all` 8.6, `sim-rnd-all` 8.6. Friedman p-value 1.91e-05.

**Selection schemes, macro-F1:** `agr-fit` 4.6, `rnd-rnd` 4.8, `agr-rnd` 5.1, `sim-fit` 6.3, `agr-fit--fit` 6.6, `sim-rnd` 6.6, `nvs-rnd` 6.8, `agr-rnd--fit` 7.0, `fit-fit` 7.0, `prs-fit` 7.2, `ads-rnd` 7.3, `prs-rnd` 7.6, `ads-fit--fit-all` 8.5, `ads-fit--fit` 8.7, `ads-rnd-all` 9.3. Friedman p-value 0.000228.

**Comparison with classifiers, macro-F1:** `ExtraTrees` 10.2, `agr-fit` 10.4, `rnd-rnd` 10.8, `agr-rnd` 11.0, `RandomForest` 11.0, `agr-fit-all` 11.0, `GradientBoosting` 11.0, `agr-rnd-all` 11.1, `prs-fit-all` 11.9, `sim-fit` 12.2, `prs-rnd-all` 12.9, `agr-fit--fit` 13.2, `sim-rnd` 13.4, `nvs-rnd` 13.5, `prs-fit` 13.8, `agr-rnd--fit` 13.8, `fit-fit` 14.0, `prs-rnd` 14.3, `ads-rnd` 14.7, `ads-fit--fit-all` 15.5, `sim-fit-all` 16.1, `ads-fit--fit` 16.2, `sim-rnd-all` 16.2, `MLP` 16.3, `ads-rnd-all` 17.3, `DecisionTree` 17.3, `KNeighbors` 21.5, `LinearSVC` 21.6, `LogisticRegression` 22.0, `SVC` 25.3, `AdaBoost` 25.8, `GaussianNB` 26.8, `SGD` 29.6, `PassiveAggressive` 31.5, `Perceptron` 31.7, `BernoulliNB` 31.9, `NearestCentroid` 32.8. Friedman p-value 8.41e-68.

EvoDAG is slower to train than every scikit-learn classifier (time ranks 17 to 37 of 37).

## Profile

The fit of one member is profiled with `profile_fit.py` (cProfile, one core, `--max-seconds 900`). The engine has been optimised in steps that must give **the same bits**: `parity.py` records 102 small fits (the 22 configurations, the released defaults, two ensembles, extreme values, a regression with the agreement heuristic, a generational population and a test set given to `fit`) before a change and `check` compares them again exactly, including a digest of every individual created, and `tests/test_golden.py` must pass without a skipped test.

```bash
python benchmarks/ecj2022/profile_fit.py --dataset optdigits --config agr-rnd --max-seconds 900 --output /tmp/optdigits.pstats
python benchmarks/ecj2022/parity.py record --output /tmp/parity    # before a change
python benchmarks/ecj2022/parity.py check --reference /tmp/parity  # after it
```

One member of an ensemble (`seed=0`, the parameters of the configuration without `--quick`) on one core, under cProfile (so the times include its overhead), without and with the optimisations listed below; the number of individuals created is the same in both. The last column is the own time of the top functions without them.

| Dataset | Configuration | Individuals | Fit without (s) | Fit with (s) | Speed-up | Top functions without (own time, s) |
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

The peak memory is lower too (the maximum resident set of the process, without and with the optimisations): `optdigits` `agr-rnd` 1.66 GB to 1.38 GB, `magic04` `agr-rnd` 3.75 GB to 1.29 GB, `optdigits` `fit-fit` 5.5 GB to 1.3 GB, because the engine does not keep the outputs of every individual ever created (see below).

`benchmarks/bench.py`, the fastest `fit_min` of the three seeds in seconds (`benchmarks/results-profiled.json` has the "with" run):

| Case | Without | With | Speed-up |
|---|---:|---:|---:|
| classification_1000x20_3c | 0.73 | 0.66 | 1.11x |
| classification_10000x50_5c | 11.25 | 10.44 | 1.08x |
| regression_1000x20 | 0.62 | 0.37 | 1.67x |

The training accuracy and the relative squared error of the benchmark, all 102 cases of `parity.py` and the size of a pickled model are the same with and without the optimisations.

### Optimisations kept

Each one is covered by a test that compares the fast path with the plain computation (`tests/test_optimisations.py`, `tests/test_linalg_solve.py`) and by the parity check.

- **Predicted labels of an individual are remembered** (`Variable.labels`, used by the agreement heuristic of `EvoDAG/base.py`). The labels (`argmax` of the outputs, or the sign in regression) would be needed for every candidate of every tournament, about 60 000 to 200 000 times per fit. They depend only on `hy`, which is not changed while the individual is in the population, and the cache is dropped whenever `hy` is replaced or released, so every value is the one that is computed from `hy`. It is not stored in the models (their size in bytes is unchanged). It costs one small array per individual in the population (the labels, as long as the training set), and the cache is dropped with the outputs, so it is bounded by the population like them. It is the largest gain for many classes (`optdigits` `agr-rnd`: 15 s of 37).
- **The train accuracy of the agreement heuristic does not count the validation part** (`Score._accuracy_index`). `Score.accuracy` also counts the rest of the examples, and only the first number is used. The counts are integers, so the quotient is the same one; the method is private because the golden test asks every public kernel to be recorded.
- **`compute_weight` computes the dot products of the Gram matrix four at a time** (`EvoDAG/linalg_solve.pyx`) when all the vectors are finite. Every sum adds the same numbers in the same order as `dot`, so the results are identical; four independent chains stop waiting for the latency of the additions. The guard of `dot` is not needed then: when everything is finite a product with a zero is a zero and adding it changes nothing (the sum is never `-0.0`). The columns are taken by their address, once. With a non-finite value the guarded path is used. `tests/test_linalg_solve.py` compares it with a plain Python version of the whole algorithm, signs of zero included. It halves the time of an `Add` with many arguments, the main cost of the datasets with two classes and many examples. `dot` itself keeps its branch: a branch-free version cannot be fused into a multiply-add by the compiler, so on builds that contract floating point expressions it would not give the same bits.
- **`array_utils.mul` has no branch** (the product where both factors are not zero and `+0.0` elsewhere). The masks have random zeros, so the branch was unpredictable: 25 to 10 microseconds for 3823 values. It helps the cosine, Pearson and angle heuristics, which multiply every output by the mask for every candidate.
- **Pearson and angle selection**: the quantities that depend only on the reference (its centred vector and norm) are computed once per reference instead of once per candidate, `u.mean()` is `np.add.reduce(u) / u.size` (the same sum and division without the method overhead), and `np.clip` of a number is `min(max(...))`, which gives the same value (also for NaN). The regression index of the heuristic is computed once per mask.
- **A node does not keep its density cache once it is released** (`BasePopulation.clean`). The cache stores a reference to the outputs, so keeping it would keep the outputs of every individual ever created in memory, not only those of the population. This is what keeps the labels cache from raising the peak memory: the memory the experiments request (`submit.py`, `1.8 * 8 * popsize * classes * examples`) is sized for a population.

### Candidates tried and rejected

- **One Gram matrix for all the outputs of a node**: it does not apply. The masks of the outputs are different (each class has its own balanced sample, `bagging_fitness.py`), and for `Add` and the functions of one argument the arguments of each output are different (the output `k` of each argument). Only the right-hand side is shared, which is nothing.
- **Eight dot products at a time** instead of four: no gain (the time is bounded by the loads).
- **A branch-free `dot`** (also without the four chains): it helps on sparse vectors only, and it is not bit-identical on builds that fuse multiply-adds, so it is not used.
- **Remembering the masked outputs of an individual** for the cosine, Pearson and angle heuristics (`_masked_outputs` is the first function for them): it would double the memory of the population, which the memory request of `submit.py` does not allow for. A smarter layout is a decision for the maintainer.
- **Replacing `gauss_jordan` by a LAPACK solve**, changing the order of the sums, `np.random` in blocks: not allowed (they change bits).

### What remains the bottleneck

`compute_weight` (the Gram matrix and the elimination of an `Add` with up to 60 arguments, which is a quadratic number of dot products of vectors that each depend on the mask), then `_masked_outputs` and `_angle_value` of the heuristics that look at the semantics (`ads-rnd` is slow because each argument draws many candidates), the `eval` of the Naive Bayes and Centroid nodes for many classes (`letter-recognition`), and `_novelty_value` (the work is one gather and one sum per candidate). Going further needs a blocked kernel for the Gram matrix that keeps the order of every sum (a SIMD layout across the dot products), or an algorithmic change (fewer arguments, the Gram matrix updated incrementally), which is a decision for the maintainer.
