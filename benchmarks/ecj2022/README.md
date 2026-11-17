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
