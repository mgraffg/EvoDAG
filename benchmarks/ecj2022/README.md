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
