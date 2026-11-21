# Copyright 2024 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download and prepare the 30 UCI datasets of the selection heuristics article.

    python benchmarks/ecj2022/datasets.py
    python benchmarks/ecj2022/datasets.py --datasets iris wine
    python benchmarks/ecj2022/datasets.py --list
    python benchmarks/ecj2022/datasets.py --force

The article (Sanchez and Graff, *Selection Heuristics on Semantic Genetic
Programming for Classification Problems*, Evolutionary Computation, 2022)
uses 30 classification problems of the UCI repository (its Table 1). Its
preprocessing is lost, so each archive is downloaded from
``https://archive.ics.uci.edu`` and prepared with the uniform rules below.

Where the files go: ``--root``, else the environment variable
``EVODAG_EXPERIMENTS``, else the directory ``experiments`` at the root of the
repository, which git ignores. Inside it, ``raw/<name>/`` has the archives and
their extracted files, ``data/<name>.npz`` the prepared data and
``data/manifest.json`` one entry per dataset that compares it with Table 1.

Uniform rules (a dataset may override the part that its row says):

1. Every file is read as text with the standard library, fields are stripped
   and empty lines are dropped.
2. A column is numeric when every non-missing value of it, over train and test
   together, is a finite number; otherwise it is categorical and each value is
   replaced by its position in the sorted list of the distinct values of the
   column (ordinal encoding).
3. The missing values are ``?``, ``na``, ``NA``, ``nan``, ``NaN`` and the
   empty string. In a numeric column they become the median of the training
   part; in a categorical one ``?`` is one more category.
4. Constant columns are kept.
5. Without an official split, 70% of the examples of each class (a
   deterministic, stratified split with ``numpy.random.RandomState(0)``) are
   for training and 30% for test.
6. The classes are renumbered ``0 .. k - 1`` in the sorted order of the
   original labels (numerically when all of them are numbers).

Only the standard library and NumPy are used. ``.Z`` files (Unix compress) are
expanded with the ``uncompress`` or ``gzip`` command.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

BASE_URL = "https://archive.ics.uci.edu/static/public"
ENV_ROOT = "EVODAG_EXPERIMENTS"
MISSING = {"?", "na", "NA", "nan", "NaN", ""}
TEST_FRACTION = 0.3
SIZE_TOLERANCE = 2

# sha256 of the archive of each dataset, taken from the first successful run
# and verified on later downloads
SHA256 = {
    "ad": "ec24f1b9285079146dad6af5edb1f9c1bde3811576cc53368e60708d2165e47c",
    "adult": "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
    "agaricus-lepiota": "face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a",
    "aps-failure": "5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89",
    "banknote": "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227",
    "bank": "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4",
    "biodeg": "b5f7cf1b05564d94d65265a9a4aeae5fffcd15b8d3bcafe4225a8827d7ae08ac",
    "car": "1559d51dcf327f4f8c71b711ceed7fd95a382fc8d8e1998667f4f23b82860403",
    "census-income": "54fd206becffcfaf099544c3938c681d64a65709800c94c00a4fba0a00df10c9",
    "cmc": "ca2e59780133ce0040cbe3939243ee71419fd2d1c72ae1edb5c90c0977d55785",
    "dota2": "5bf3d2fe75c55ffba506f4aaee0d88684aa8e651f622ff32a0ec7928ae476717",
    "drug-consumption": "0fb006913b8ecde52560dd04e4b8d10c75aad938d337a9e7e11fcab7dd1f6993",
    "fertility": "9b346e4908040f17c8980b04f13749d215af9c8b268ec0f64cffab6e94b188e0",
    "IndianLiverPatient": "60e18fbfe7f84daf58c178ea280ba5fe8db7e2507aabce01b57425b3b822cf0b",
    "iris": "d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9",
    "krkopt": "39fc7ec218477c21d61032917d74ddb98d6fdbe3400fdaa6f514fad0b603712f",
    "letter-recognition": "3b5f07a334697b6cace4fbae22940393a18fee596e73f68d97ce5973d52dc60f",
    "magic04": "252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d",
    "ml-prove": "b1a9a19a7e16db42f0c74762e1a1446ca9b55a198ce8f2d9c03ffe3e75e66260",
    "musk1": "84fd967a312cfe1d48d6f5a2becdda68f122767c0d51a3246cf699ff92a66031",
    "musk2": "8bdddb011a201e211974ae584c96b779fac2108516cbfbb511e5fc21437331f4",
    "optdigits": "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0",
    "page-blocks": "b766faf029e05afd8c3969376fbdd01e377c7580eb88262772aa9c9e02fe28bf",
    "parkinsons": "9baccfbcdac15d6d9347b37c2a352fb31d65c52e367df5a47d7543a55c75b0ab",
    "pendigits": "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64",
    "segmentation": "db4297f9a7d6acfe63495452118bdeff9c0d0eab14c75bd0ee7678f79f488914",
    "sensorless": "fbce04b326362156f007838cec6b12ab462113a467e7cc529f43471065460c77",
    "tae": "65f77e4ac3e356db99e8f7c4a06834d06316e9ae040492d31fcc64705d6dd0a0",
    "wine": "2bae62c4481220623579d4c4fb36b55652b6b75e06e49fa1981b8198362dfdab",
    "yeast": "8fe9a4490edfa67532f1dee7ec11f1804437ad43f9f8b7536e32d5be8bce0bb5",
}


@dataclass
class Spec:
    """How to read one dataset.

    ``paper`` is the row of Table 1: training examples, test examples,
    variables, classes and class entropy. ``train`` (and ``test`` for an
    official split) are the file names; several training files are
    concatenated. ``sep`` is the field separator (``None``: any whitespace).
    ``skip`` is the number of non-empty lines to ignore at the top of each
    file (one number, or a pair for the training and the test files) and
    ``header`` says that the next line holds the names of the columns.
    ``target`` is ``"first"``, ``"last"``, an index or a column name and
    ``drop`` lists columns (indices or names) that are not variables.
    ``strip_dot`` removes a final dot of the labels.
    """
    name: str
    uci_id: int
    slug: str
    paper: Tuple[int, int, int, int, float]
    train: Tuple[str, ...]
    test: Tuple[str, ...] = ()
    sep: Optional[str] = ","
    skip: Union[int, Tuple[int, int]] = 0
    header: bool = False
    target: Union[str, int] = "last"
    drop: Tuple[Union[int, str], ...] = ()
    strip_dot: bool = False
    note: str = field(default="", compare=False)

    def __post_init__(self):
        # a single file name or column may be given instead of a tuple
        for key in ("train", "test", "drop"):
            value = getattr(self, key)
            if isinstance(value, (str, int)):
                setattr(self, key, (value,))
            else:
                setattr(self, key, tuple(value))

    @property
    def official(self):
        return bool(self.test)

    @property
    def url(self):
        return "%s/%d/%s.zip" % (BASE_URL, self.uci_id, self.slug)


def _spec(name, uci_id, slug, paper, train, **kwargs):
    return Spec(name, uci_id, slug, tuple(paper), train, **kwargs)


SPECS = [
    _spec("ad", 51, "internet+advertisements",
          (2295, 984, 1557, 2, 0.58), "ad.data"),
    _spec("adult", 2, "adult", (32561, 16281, 14, 2, 0.8), "adult.data",
          test="adult.test", skip=(0, 1), strip_dot=True),
    # the article has 7 classes and 22 variables: the class is the habitat
    _spec("agaricus-lepiota", 73, "mushroom", (5686, 2438, 22, 7, 0.81),
          "agaricus-lepiota.data"),
    _spec("aps-failure", 421, "aps+failure+at+scania+trucks",
          (60000, 16000, 170, 2, 0.12), "aps_failure_training_set.csv",
          test="aps_failure_test_set.csv", skip=14, header=True,
          target="first"),
    _spec("banknote", 267, "banknote+authentication",
          (960, 412, 4, 2, 0.99), "data_banknote_authentication.txt"),
    _spec("bank", 222, "bank+marketing", (31647, 13564, 16, 2, 0.52),
          "bank-full.csv", sep=";", skip=1),
    _spec("biodeg", 254, "qsar+biodegradation", (738, 317, 41, 2, 0.91),
          "biodeg.csv", sep=";"),
    _spec("car", 19, "car+evaluation", (1209, 519, 6, 4, 0.6), "car.data"),
    _spec("census-income", 117, "census+income+kdd",
          (199523, 99762, 41, 2, 0.34), "census-income.data",
          test="census-income.test", strip_dot=True),
    _spec("cmc", 30, "contraceptive+method+choice",
          (1031, 442, 9, 3, 0.98), "cmc.data"),
    _spec("dota2", 367, "dota2+games+results", (92650, 10294, 116, 2, 1.0),
          "dota2Train.csv", test="dota2Test.csv", target="first"),
    _spec("drug-consumption", 373, "drug+consumption+quantified",
          (1319, 566, 30, 7, 0.44), "drug_consumption.data", drop=0),
    _spec("fertility", 244, "fertility", (69, 30, 9, 2, 0.43),
          "fertility_Diagnosis.txt"),
    _spec("IndianLiverPatient", 225, "ilpd+indian+liver+patient+dataset",
          (407, 175, 10, 2, 0.85), "Indian Liver Patient Dataset (ILPD).csv"),
    _spec("iris", 53, "iris", (105, 45, 4, 3, 1.0), "iris.data"),
    _spec("krkopt", 23, "chess+king+rook+vs+king",
          (19639, 8417, 6, 18, 0.84), "krkopt.data"),
    _spec("letter-recognition", 59, "letter+recognition",
          (14000, 6000, 16, 26, 1.0), "letter-recognition.data",
          target="first"),
    _spec("magic04", 159, "magic+gamma+telescope",
          (13314, 5706, 10, 2, 0.93), "magic04.data"),
    _spec("ml-prove", 249, "first+order+theorem+proving",
          (4588, 1530, 56, 2, 0.98), ("ml-prove/train.csv",
                                      "ml-prove/validation.csv"),
          test="ml-prove/test.csv"),
    _spec("musk1", 74, "musk+version+1", (333, 143, 166, 2, 0.99),
          "clean1.data", drop=(0, 1)),
    _spec("musk2", 75, "musk+version+2", (4618, 1980, 166, 2, 0.61),
          "clean2.data", drop=(0, 1)),
    _spec("optdigits", 80, "optical+recognition+of+handwritten+digits",
          (3823, 1797, 64, 10, 1.0), "optdigits.tra", test="optdigits.tes"),
    _spec("page-blocks", 78, "page+blocks+classification",
          (3831, 1642, 10, 5, 0.27), "page-blocks.data", sep=None),
    _spec("parkinsons", 174, "parkinsons", (135, 59, 22, 2, 0.79),
          "parkinsons.data", header=True, drop="name", target="status"),
    _spec("pendigits", 81, "pen+based+recognition+of+handwritten+digits",
          (7494, 3498, 16, 10, 1.0), "pendigits.tra", test="pendigits.tes"),
    _spec("segmentation", 50, "image+segmentation",
          (210, 2100, 19, 7, 1.0), "segmentation.data",
          test="segmentation.test", skip=3, target="first"),
    _spec("sensorless", 325, "dataset+for+sensorless+drive+diagnosis",
          (40956, 17553, 48, 11, 1.0), "Sensorless_drive_diagnosis.txt",
          sep=None),
    _spec("tae", 100, "teaching+assistant+evaluation", (105, 45, 5, 3, 0.99),
          "tae.data"),
    _spec("wine", 109, "wine", (123, 53, 13, 3, 0.99), "wine.data",
          target="first"),
    _spec("yeast", 110, "yeast", (1038, 446, 9, 10, 0.76), "yeast.data",
          sep=None),
]


# -- locations -------------------------------------------------------------

def resolve_root(root=None, environ=None):
    """The experiments directory: ``root``, else ``EVODAG_EXPERIMENTS``, else
    ``experiments`` at the root of the repository."""
    environ = os.environ if environ is None else environ
    if root:
        return Path(root)
    if environ.get(ENV_ROOT):
        return Path(environ[ENV_ROOT])
    return Path(__file__).resolve().parents[2] / "experiments"


def atomic_write(path, writer):
    """Call ``writer(file)`` on a temporary file and rename it to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fpt:
        writer(fpt)
    os.replace(tmp, path)


# -- download and extraction -----------------------------------------------

def sha256_of(path):
    "Hexadecimal sha256 of a file"
    digest = hashlib.sha256()
    with open(path, "rb") as fpt:
        for chunk in iter(lambda: fpt.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest, retries=3, pause=5.0):
    """Download ``url`` to ``dest`` through a ``.part`` file, trying up to
    ``retries`` times."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "evodag-benchmarks"})
    for attempt in range(1, retries + 1):
        try:
            print("downloading %s" % url, file=sys.stderr)
            with urllib.request.urlopen(request, timeout=120) as response, \
                    open(part, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            os.replace(part, dest)
            return dest
        except (OSError, ValueError) as error:
            if attempt == retries:
                raise
            print("download failed (%s), retrying" % error, file=sys.stderr)
            time.sleep(pause)
    return dest


def _safe_target(base, member):
    target = (base / member).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError("unsafe path in archive: %s" % member)
    return target


def extract_archive(path, dest):
    """Extract a ``.zip``, ``.tar``, ``.tar.gz`` or ``.tgz`` into ``dest``."""
    path, dest = Path(path), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zfile:
            for name in zfile.namelist():
                _safe_target(dest, name)
            zfile.extractall(dest)
    else:
        with tarfile.open(path) as tfile:
            for member in tfile.getmembers():
                _safe_target(dest, member.name)
            if hasattr(tarfile, "data_filter"):
                tfile.extractall(dest, filter="data")
            else:  # pragma: no cover
                tfile.extractall(dest)


def decompress(path):
    """Expand a ``.gz`` or ``.Z`` file next to itself and return the new path.
    ``.Z`` (Unix compress) needs the ``uncompress`` or ``gzip`` command."""
    path = Path(path)
    out = path.with_suffix("")
    tmp = out.with_name(out.name + ".part")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(tmp, out)
        return out
    if path.suffix != ".Z":
        raise ValueError("cannot decompress %s" % path)
    errors = []
    for command in (["uncompress", "-c", str(path)],
                    ["gzip", "-dc", str(path)]):
        if shutil.which(command[0]) is None:
            errors.append("%s: not found" % command[0])
            continue
        with open(tmp, "wb") as dst:
            done = subprocess.run(command, stdout=dst,
                                  stderr=subprocess.PIPE)
        if done.returncode == 0:
            os.replace(tmp, out)
            return out
        errors.append("%s: %s" % (command[0],
                                  done.stderr.decode("latin-1").strip()))
    if tmp.exists():
        tmp.unlink()
    raise RuntimeError("cannot expand %s (a Unix compress file); install the "
                       "uncompress or gzip command (%s)"
                       % (path, "; ".join(errors)))


def _walk(raw):
    for folder, _, names in os.walk(raw):
        for name in sorted(names):
            yield Path(folder) / name


def _find(raw, relative):
    parts = Path(relative).parts
    for path in _walk(raw):
        if path.parts[-len(parts):] == parts:
            return path
    return None


def locate(raw, relative):
    """Path of the file ``relative`` (a name or ``folder/name``) somewhere
    below ``raw``, expanding compressed files and nested archives as needed."""
    raw = Path(raw)
    for _ in range(100):
        hit = _find(raw, relative)
        if hit is not None:
            return hit
        done = False
        for suffix in (".gz", ".Z"):
            hit = _find(raw, str(relative) + suffix)
            if hit is not None:
                return decompress(hit)
        for path in _walk(raw):
            marker = path.with_name(path.name + ".extracted")
            archive = path.name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
            if archive and not marker.exists():
                extract_archive(path, path.parent)
                marker.write_text("")
                done = True
                break
        if not done:
            break
    raise FileNotFoundError("%s is not in %s or in the archives below it"
                            % (relative, raw))


def fetch(spec, root):
    """Download (if needed) and extract the archive of ``spec`` and return
    ``(raw_directory, sha256_of_the_archive)``."""
    raw = Path(root) / "raw" / spec.name
    archive = raw / (spec.slug + ".zip")
    if not archive.exists():
        download(spec.url, archive)
    digest = sha256_of(archive)
    expected = SHA256.get(spec.name)
    if expected is not None and digest != expected:
        raise ValueError("%s has sha256 %s but %s was expected; refusing to "
                         "use it (delete it to download it again)"
                         % (archive, digest, expected))
    marker = archive.with_name(archive.name + ".extracted")
    if not marker.exists():
        extract_archive(archive, raw)
        marker.write_text("")
    return raw, digest


# -- reading and encoding --------------------------------------------------

def read_rows(path, sep=",", skip=0, header=False):
    """Rows of a text file as lists of stripped strings: the first ``skip``
    non-empty lines are ignored, empty lines are dropped and, with ``header``,
    the next line is returned apart as the names of the columns.
    Returns ``(names or None, rows)``."""
    with open(path, encoding="latin-1", newline="") as fpt:
        lines = [line for line in fpt.read().splitlines() if line.strip()]
    lines = lines[skip:]
    names = None
    if sep is None:
        rows = [line.split() for line in lines]
    else:
        rows = [[field.strip() for field in row]
                for row in csv.reader(lines, delimiter=sep)]
    if header:
        names, rows = rows[0], rows[1:]
    return names, rows


def column_index(ref, names, width):
    "Position of a column given as ``first``, ``last``, an index or a name"
    if ref == "first":
        return 0
    if ref == "last":
        return width - 1
    if isinstance(ref, str):
        if names is None:
            raise ValueError("column %r needs a header" % ref)
        return names.index(ref)
    return int(ref)


def _parse_number(value):
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def encode_columns(train_rows, test_rows):
    """Turn rows of strings into float matrices. Numeric columns keep their
    value (missing: median of the training part), the others are ordinal
    encoded over the sorted distinct values of both parts (missing: the
    category ``?``)."""
    width = len(train_rows[0]) if train_rows else len(test_rows[0])
    n_train = len(train_rows)
    rows = list(train_rows) + list(test_rows)
    X = np.empty((len(rows), width), dtype=np.float64)
    for j in range(width):
        column = [row[j] for row in rows]
        present = [v for v in column if v not in MISSING]
        numbers = [_parse_number(v) for v in present]
        if present and all(n is not None for n in numbers):
            values = np.array([np.nan if v in MISSING else float(v)
                               for v in column])
            reference = values[:n_train]
            reference = reference[~np.isnan(reference)]
            if not len(reference):
                reference = values[~np.isnan(values)]
            if np.isnan(values).any():
                values[np.isnan(values)] = float(np.median(reference))
            X[:, j] = values
        else:
            column = ["?" if v in MISSING else v for v in column]
            codes = {v: i for i, v in enumerate(sorted(set(column)))}
            X[:, j] = [codes[v] for v in column]
    return X[:n_train], X[n_train:]


def encode_labels(train_labels, test_labels):
    """Integer labels ``0 .. k - 1`` following the sorted order of the
    original labels (numeric when all of them are numbers). Returns
    ``(y_train, y_test, classes)`` with ``classes`` the original strings."""
    distinct = set(train_labels) | set(test_labels)
    numbers = [_parse_number(v) for v in distinct]
    if all(n is not None for n in numbers):
        classes = sorted(distinct, key=lambda v: (float(v), v))
    else:
        classes = sorted(distinct)
    codes = {v: i for i, v in enumerate(classes)}
    return (np.array([codes[v] for v in train_labels], dtype=np.int64),
            np.array([codes[v] for v in test_labels], dtype=np.int64),
            classes)


def split_70_30(y, seed=0):
    """Deterministic stratified split. For each class, in sorted order, its
    indices (in file order) are shuffled with ``RandomState(seed)`` and the
    first ``round(0.3 n)`` (at least one when ``n >= 2``, none when ``n ==
    1``) go to the test set. Returns the sorted indices ``(train, test)``."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    train, test = [], []
    for label in np.unique(y):
        index = np.flatnonzero(y == label)
        rng.shuffle(index)
        n = len(index)
        n_test = 0 if n < 2 else max(1, int(round(TEST_FRACTION * n)))
        test.extend(index[:n_test])
        train.extend(index[n_test:])
    return (np.array(sorted(train), dtype=np.int64),
            np.array(sorted(test), dtype=np.int64))


def read_part(raw, files, spec, which):
    """Rows of one part (``0`` train, ``1`` test) split into the variables
    and the labels, concatenating the files."""
    skip = spec.skip[which] if isinstance(spec.skip, tuple) else spec.skip
    features, labels = [], []
    width = None
    for relative in files:
        names, rows = read_rows(locate(raw, relative), spec.sep, skip,
                                spec.header)
        if not rows:
            raise ValueError("%s has no rows" % relative)
        if width is None:
            width = len(rows[0])
        if any(len(row) != width for row in rows):
            bad = next(r for r in rows if len(r) != width)
            raise ValueError("%s: rows with %d and %d fields (%r)"
                             % (relative, width, len(bad), bad[:3]))
        target = column_index(spec.target, names, width)
        drop = {column_index(d, names, width) for d in spec.drop}
        keep = [j for j in range(width) if j != target and j not in drop]
        for row in rows:
            labels.append(row[target].rstrip(".") if spec.strip_dot
                          else row[target])
            features.append([row[j] for j in keep])
    return features, labels


def load_dataset(raw, spec):
    """Prepared arrays of a dataset: ``X_train``, ``X_test``, ``y_train``,
    ``y_test`` and ``classes`` (a dictionary), plus its split name and the
    description of the target column."""
    raw = Path(raw)
    f_train, l_train = read_part(raw, spec.train, spec, 0)
    if spec.official:
        f_test, l_test = read_part(raw, spec.test, spec, 1)
        split = "official"
    else:
        y_all, _, _ = encode_labels(l_train, [])
        i_train, i_test = split_70_30(y_all)
        f_test = [f_train[i] for i in i_test]
        l_test = [l_train[i] for i in i_test]
        f_train = [f_train[i] for i in i_train]
        l_train = [l_train[i] for i in i_train]
        split = "70/30 seed 0"
    X_train, X_test = encode_columns(f_train, f_test)
    y_train, y_test, classes = encode_labels(l_train, l_test)
    arrays = {"X_train": X_train, "X_test": X_test, "y_train": y_train,
              "y_test": y_test, "classes": np.array(classes, dtype=str)}
    return arrays, split


def class_entropy(y, n_classes):
    "Shannon entropy with the logarithm in base ``n_classes``, 2 decimals"
    if n_classes < 2:
        return 0.0
    p = np.bincount(y, minlength=n_classes) / len(y)
    p = p[p > 0]
    return round(float(-(p * np.log(p)).sum() / math.log(n_classes)), 2)


def compare_with_paper(spec, entry):
    """Differences between a manifest entry and Table 1: a list of strings
    and the boolean ``matches_paper`` (total and test sizes within the
    tolerance, same variables and classes)."""
    p_train, p_test, p_vars, p_classes, p_entropy = spec.paper
    differences = []
    n_train, n_test = entry["n_train"], entry["n_test"]
    if n_train != p_train or n_test != p_test:
        differences.append("size %d/%d against %d/%d of the article"
                           % (n_train, n_test, p_train, p_test))
    if entry["n_variables"] != p_vars:
        differences.append("%d variables against %d"
                           % (entry["n_variables"], p_vars))
    if entry["n_classes"] != p_classes:
        differences.append("%d classes against %d"
                           % (entry["n_classes"], p_classes))
    if abs(entry["class_entropy"] - p_entropy) > 0.02 + 1e-9:
        differences.append("class entropy %.2f against %.2f"
                           % (entry["class_entropy"], p_entropy))
    matches = (abs(n_train + n_test - p_train - p_test) <= SIZE_TOLERANCE
               and abs(n_test - p_test) <= max(SIZE_TOLERANCE, 0.01 * p_test)
               and entry["n_variables"] == p_vars
               and entry["n_classes"] == p_classes)
    return differences, matches


def describe_target(spec):
    "Which column is the class, as text"
    if spec.target in ("first", "last"):
        return "%s column" % spec.target
    if isinstance(spec.target, str):
        return "column %r" % spec.target
    return "column %d" % spec.target


def manifest_entry(spec, arrays, split, sha256):
    "The manifest entry of a prepared dataset"
    k = len(arrays["classes"])
    entry = {
        "name": spec.name, "uci_id": spec.uci_id, "url": spec.url,
        "sha256": sha256,
        "n_train": int(len(arrays["y_train"])),
        "n_test": int(len(arrays["y_test"])),
        "n_variables": int(arrays["X_train"].shape[1]),
        "n_classes": int(k),
        "class_entropy": class_entropy(arrays["y_train"], k),
        "split": split, "target": describe_target(spec),
        "paper": list(spec.paper),
    }
    differences, matches = compare_with_paper(spec, entry)
    missing = sorted(set(range(k)) - set(arrays["y_train"].tolist()))
    if missing:
        differences.append("no training example of the classes %s" % missing)
    entry["matches_paper"] = bool(matches)
    entry["differences"] = differences
    return entry


def check_arrays(arrays):
    "The invariants of a prepared dataset"
    for key in ("X_train", "X_test"):
        assert arrays[key].dtype == np.float64 and arrays[key].ndim == 2
        assert np.isfinite(arrays[key]).all(), "%s is not finite" % key
    assert arrays["X_train"].shape[1] == arrays["X_test"].shape[1]
    for key in ("y_train", "y_test"):
        assert arrays[key].dtype == np.int64


def save_npz(path, arrays):
    "Write the prepared dataset atomically"
    atomic_write(path, lambda fpt: np.savez_compressed(fpt, **arrays))


# -- manifest and driver ---------------------------------------------------

def read_manifest(root):
    "Entries of the manifest by name ({} if there is none)"
    path = Path(root) / "data" / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as fpt:
        return {entry["name"]: entry for entry in json.load(fpt)}


def write_manifest(root, entries):
    "Rewrite the manifest atomically, in the order of ``SPECS``"
    order = {spec.name: i for i, spec in enumerate(SPECS)}
    ordered = sorted(entries.values(),
                     key=lambda e: order.get(e["name"], len(order)))
    text = json.dumps(ordered, indent=2).encode("utf8")
    atomic_write(Path(root) / "data" / "manifest.json",
                 lambda fpt: fpt.write(text + b"\n"))


def prepare(spec, root, force=False):
    """Prepare one dataset and return its manifest entry (``None`` when it was
    already done and ``force`` is false)."""
    root = Path(root)
    npz = root / "data" / (spec.name + ".npz")
    done = read_manifest(root).get(spec.name)
    if not force and npz.exists() and done and "error" not in done:
        return None
    raw, digest = fetch(spec, root)
    arrays, split = load_dataset(raw, spec)
    check_arrays(arrays)
    entry = manifest_entry(spec, arrays, split, digest)
    save_npz(npz, arrays)
    return entry


def run(specs, root, force=False):
    "Prepare every dataset of ``specs``; returns the names that failed"
    entries = read_manifest(root)
    failed = []
    for spec in specs:
        try:
            entry = prepare(spec, root, force)
        except Exception as error:  # one dataset must not stop the others
            print("%s: FAILED: %s: %s" % (spec.name, type(error).__name__,
                                          error), file=sys.stderr)
            entries[spec.name] = {"name": spec.name,
                                  "error": "%s: %s" % (type(error).__name__,
                                                       error)}
            failed.append(spec.name)
        else:
            if entry is None:
                print("%s: already prepared" % spec.name)
                continue
            entries[spec.name] = entry
            print("%s: %d/%d examples, %d variables, %d classes%s"
                  % (spec.name, entry["n_train"], entry["n_test"],
                     entry["n_variables"], entry["n_classes"],
                     "" if entry["matches_paper"] else " (differs)"))
        write_manifest(root, entries)
    return failed


def markdown_table(entries):
    "The manifest as a Markdown table compared with Table 1"
    lines = ["| dataset | train | test | variables | classes | entropy | split "
             "| matches Table 1 | differences |",
             "|---|---|---|---|---|---|---|---|---|"]
    for entry in entries:
        if "error" in entry:
            lines.append("| %s | | | | | | | error | %s |"
                         % (entry["name"], entry["error"]))
            continue
        lines.append("| %s | %d | %d | %d | %d | %.2f | %s | %s | %s |" % (
            entry["name"], entry["n_train"], entry["n_test"],
            entry["n_variables"], entry["n_classes"], entry["class_entropy"],
            entry["split"], "yes" if entry["matches_paper"] else "no",
            "; ".join(entry["differences"])))
    return "\n".join(lines)


def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Download and prepare the 30 UCI datasets of the "
                    "selection heuristics article.")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="only these datasets (default: the 30)")
    parser.add_argument("--list", action="store_true",
                        help="print the manifest as a Markdown table")
    parser.add_argument("--force", action="store_true",
                        help="prepare again even if the .npz exists")
    parser.add_argument("--root", default=None,
                        help="experiments directory (default: "
                             "$%s or <repository>/experiments)" % ENV_ROOT)
    args = parser.parse_args(argv)
    root = resolve_root(args.root)
    names = [spec.name for spec in SPECS]
    selected = args.datasets or names
    unknown = [n for n in selected if n not in names]
    if unknown:
        parser.error("unknown datasets: %s" % ", ".join(unknown))
    if args.list:
        entries = read_manifest(root)
        print(markdown_table([entries[n] for n in selected if n in entries]))
        return 0
    failed = run([s for s in SPECS if s.name in selected], root, args.force)
    if failed:
        print("failed: %s" % ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
