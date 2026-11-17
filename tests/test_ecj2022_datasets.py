"""Tests of ``benchmarks/ecj2022/datasets.py``. Nothing is downloaded and
everything is written in ``tmp_path``."""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "ecj2022_datasets", ROOT / "benchmarks" / "ecj2022" / "datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_script()


def test_columns_are_typed_and_encoded():
    train = [["1.5", "b", "x"], ["?", "a", "y"], ["3.5", "?", "x"],
             ["10", "b", "z"]]
    test = [["na", "c", "y"], ["2", "a", ""]]
    X_train, X_test = ds.encode_columns(train, test)
    # numeric: median of the training values 1.5, 3.5, 10 is 3.5
    assert X_train[:, 0].tolist() == [1.5, 3.5, 3.5, 10.0]
    assert X_test[:, 0].tolist() == [3.5, 2.0]
    # categorical, sorted values of both parts: ?, a, b, c
    assert X_train[:, 1].tolist() == [2, 1, 0, 2]
    assert X_test[:, 1].tolist() == [3, 1]
    # a missing value of a categorical column is the category "?"
    assert X_train[:, 2].tolist() == [1, 2, 1, 3]
    assert X_test[:, 2].tolist() == [2, 0]
    assert X_train.dtype == np.float64


def test_a_column_with_a_non_number_is_categorical():
    X, _ = ds.encode_columns([["1"], ["2"], ["x"]], [])
    assert X[:, 0].tolist() == [0, 1, 2]


def test_labels_are_renumbered_in_sorted_order():
    y_train, y_test, classes = ds.encode_labels(["b", "a", "b"], ["c", "a"])
    assert classes == ["a", "b", "c"]
    assert y_train.tolist() == [1, 0, 1] and y_test.tolist() == [2, 0]
    assert y_train.dtype == np.int64
    # numeric labels are sorted as numbers
    _, _, classes = ds.encode_labels(["10", "2", "-1"], ["1"])
    assert classes == ["-1", "1", "2", "10"]


def test_the_70_30_split():
    y = np.array([0] * 100 + [1] * 10 + [2] * 2 + [3])
    train, test = ds.split_70_30(y)
    again = ds.split_70_30(y)
    assert train.tolist() == again[0].tolist()
    assert test.tolist() == again[1].tolist()
    assert not set(train) & set(test)
    assert sorted(train.tolist() + test.tolist()) == list(range(len(y)))
    assert train.tolist() == sorted(train.tolist())
    assert (y[test] == 0).sum() == 30 and (y[test] == 1).sum() == 3
    for label in (0, 1, 2):
        assert (y[train] == label).any() and (y[test] == label).any()
    # a class with one example only goes to training
    assert (y[train] == 3).sum() == 1 and (y[test] == 3).sum() == 0


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return path


def spec_for(**kwargs):
    values = dict(name="toy", uci_id=1, slug="toy", paper=(0, 0, 0, 0, 0.0),
                  train=("toy.data",))
    values.update(kwargs)
    return ds.Spec(**values)


def test_generic_reader(tmp_path):
    # class first, comma separated, blank lines and spaces around fields
    write(tmp_path / "a.data", "a, 1, 2\n\nb, 3, ?\n")
    spec = spec_for(train=("a.data",), target="first")
    features, labels = ds.read_part(tmp_path, spec.train, spec, 0)
    assert labels == ["a", "b"] and features == [["1", "2"], ["3", "?"]]
    # class last, semicolon, quotes, one line to skip
    write(tmp_path / "b.data", "junk\n1;\"x\";u\n2;\"y\";v\n")
    spec = spec_for(train=("b.data",), sep=";", skip=1)
    features, labels = ds.read_part(tmp_path, spec.train, spec, 0)
    assert labels == ["u", "v"] and features == [["1", "x"], ["2", "y"]]
    # whitespace, header with a named class column and a dropped column
    write(tmp_path / "c.data",
          "name  f1  status  f2\nr1  1  yes  2\nr2  3  no  4\n")
    spec = spec_for(train=("c.data",), sep=None, header=True, drop="name",
                    target="status")
    features, labels = ds.read_part(tmp_path, spec.train, spec, 0)
    assert labels == ["yes", "no"] and features == [["1", "2"], ["3", "4"]]
    # dropped index and a final dot of the labels, different skip for test
    write(tmp_path / "d.data", "id,1,A.\nid,2,B.\n")
    spec = spec_for(train=("d.data",), drop=(0,), strip_dot=True,
                    skip=(0, 1))
    features, labels = ds.read_part(tmp_path, spec.train, spec, 0)
    assert labels == ["A", "B"] and features == [["1"], ["2"]]
    features, labels = ds.read_part(tmp_path, spec.train, spec, 1)
    assert labels == ["B"]


def test_rows_of_different_width_fail(tmp_path):
    write(tmp_path / "e.data", "1,2,3\n1,2\n")
    spec = spec_for(train=("e.data",))
    with pytest.raises(ValueError):
        ds.read_part(tmp_path, spec.train, spec, 0)


def synthetic_rows(n=40):
    rng = np.random.RandomState(1)
    lines = []
    for i in range(n):
        lines.append("%.3f,%s,%s" % (rng.randn(), "ab"[i % 2],
                                     "yes" if i % 4 else "no"))
    return "\n".join(lines) + "\n"


def test_prepared_dataset_round_trips(tmp_path):
    raw = tmp_path / "raw" / "toy"
    write(raw / "toy.data", synthetic_rows())
    spec = spec_for(paper=(28, 12, 2, 2, 0.81))
    arrays, split = ds.load_dataset(raw, spec)
    ds.check_arrays(arrays)
    entry = ds.manifest_entry(spec, arrays, split, "00")
    assert split == "70/30 seed 0"
    assert (entry["n_train"], entry["n_test"]) == (28, 12)
    assert entry["n_variables"] == 2 and entry["n_classes"] == 2
    # 10 examples of "no" and 30 of "yes" in 40: training has 7 and 21
    p = np.array([0.25, 0.75])
    expected = round(float(-(p * np.log(p)).sum() / np.log(2)), 2)
    assert entry["class_entropy"] == expected == 0.81
    assert entry["matches_paper"] is True and entry["differences"] == []
    ds.save_npz(tmp_path / "toy.npz", arrays)
    with np.load(tmp_path / "toy.npz") as saved:
        for key in arrays:
            assert saved[key].tolist() == arrays[key].tolist()
        assert saved["X_train"].dtype == np.float64
        assert saved["y_train"].dtype == np.int64
        assert saved["classes"].tolist() == ["no", "yes"]
    assert not list(tmp_path.glob("*.tmp"))
    # a paper with other numbers does not match
    other = spec_for(paper=(100, 50, 3, 4, 0.5))
    entry = ds.manifest_entry(other, arrays, split, "00")
    assert entry["matches_paper"] is False
    assert len(entry["differences"]) == 4
    assert entry["paper"] == [100, 50, 3, 4, 0.5]


def test_size_tolerance():
    spec = spec_for(paper=(100, 50, 3, 2, 1.0))
    entry = dict(n_train=101, n_test=49, n_variables=3, n_classes=2,
                 class_entropy=1.0)
    differences, matches = ds.compare_with_paper(spec, entry)
    assert matches and differences
    entry["n_train"] = 105
    assert not ds.compare_with_paper(spec, entry)[1]


def test_official_split_keeps_the_files(tmp_path):
    raw = tmp_path / "raw" / "toy"
    write(raw / "tr.data", "1,a\n2,b\n3,a\n")
    write(raw / "te.data", "4,b\n5,c\n")
    spec = spec_for(train=("tr.data",), test=("te.data",))
    arrays, split = ds.load_dataset(raw, spec)
    assert split == "official"
    assert arrays["X_train"][:, 0].tolist() == [1, 2, 3]
    assert arrays["y_train"].tolist() == [0, 1, 0]
    assert arrays["y_test"].tolist() == [1, 2]
    entry = ds.manifest_entry(spec, arrays, split, "00")
    assert any("no training example" in d for d in entry["differences"])


def test_specification_table():
    assert len(ds.SPECS) == 30
    names = [s.name for s in ds.SPECS]
    assert len(set(names)) == 30
    by_name = {s.name: s for s in ds.SPECS}
    assert by_name["iris"].paper[:4] == (105, 45, 4, 3)
    assert by_name["adult"].paper[:4] == (32561, 16281, 14, 2)
    assert by_name["segmentation"].paper[:4] == (210, 2100, 19, 7)
    official = {s.name for s in ds.SPECS if s.official}
    assert official == {"adult", "aps-failure", "census-income", "dota2",
                        "ml-prove", "optdigits", "pendigits", "segmentation"}
    assert set(ds.SHA256) == set(names)
    assert by_name["adult"].url == \
        "https://archive.ics.uci.edu/static/public/2/adult.zip"


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv(ds.ENV_ROOT, raising=False)
    default = ds.resolve_root()
    assert default.name == "experiments"
    assert default.parent == ROOT
    monkeypatch.setenv(ds.ENV_ROOT, str(tmp_path / "env"))
    assert ds.resolve_root() == tmp_path / "env"
    assert ds.resolve_root(str(tmp_path / "opt")) == tmp_path / "opt"


def test_gitignore_has_the_experiments_directory():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert "experiments/" in lines


def fake_download(failing=()):
    def download(url, dest, retries=3, pause=5.0):
        name = Path(dest).parent.name
        if name in failing:
            raise OSError("no network for %s" % name)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as zfile:
            zfile.writestr("data/toy.data", synthetic_rows())
        return dest
    return download


def test_a_failing_dataset_is_recorded_and_the_others_are_prepared(
        tmp_path, monkeypatch, capsys):
    specs = [spec_for(name=n, slug=n, train=("toy.data",), paper=(28, 12, 2, 2, 0.81))
             for n in ("good", "bad", "also-good")]
    monkeypatch.setattr(ds, "SPECS", specs)
    monkeypatch.setattr(ds, "SHA256", {})
    monkeypatch.setattr(ds, "download", fake_download(failing=("bad",)))
    status = ds.main(["--root", str(tmp_path)])
    assert status == 1
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert [e["name"] for e in manifest] == ["good", "bad", "also-good"]
    assert "error" in manifest[1] and "no network" in manifest[1]["error"]
    assert "error" not in manifest[0] and manifest[0]["matches_paper"]
    assert (tmp_path / "data" / "good.npz").exists()
    assert (tmp_path / "data" / "also-good.npz").exists()
    assert not (tmp_path / "data" / "bad.npz").exists()
    # a second run only retries the failed one and downloads nothing else
    calls = []
    inner = fake_download()

    def counting(url, dest, **kwargs):
        calls.append(Path(dest).parent.name)
        return inner(url, dest, **kwargs)
    monkeypatch.setattr(ds, "download", counting)
    assert ds.main(["--root", str(tmp_path)]) == 0
    assert calls == ["bad"]
    assert ds.main(["--root", str(tmp_path)]) == 0 and calls == ["bad"]
    capsys.readouterr()
    assert ds.main(["--root", str(tmp_path), "--list"]) == 0
    assert "| good |" in capsys.readouterr().out


def test_sha256_mismatch_is_refused(tmp_path, monkeypatch):
    spec = spec_for(name="good", slug="good", paper=(28, 12, 2, 2, 0.81))
    monkeypatch.setattr(ds, "download", fake_download())
    monkeypatch.setattr(ds, "SHA256", {"good": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        ds.fetch(spec, tmp_path)


def test_unknown_dataset_name_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        ds.main(["--root", str(tmp_path), "--datasets", "nothing"])


def test_compressed_and_nested_files_are_found(tmp_path, monkeypatch):
    import gzip
    import io
    import tarfile
    raw = tmp_path / "raw"
    # nested zip
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zfile:
        zfile.writestr("bank-full.csv", "1;2\n")
    raw.mkdir()
    with zipfile.ZipFile(raw / "outer.zip", "w") as zfile:
        zfile.writestr("bank.zip", inner.getvalue())
    assert locate_text(ds.locate(raw, "bank-full.csv")) == "1;2\n"
    # tar.gz with a folder
    with tarfile.open(raw / "x.tar.gz", "w:gz") as tfile:
        data = b"a,b\n"
        info = tarfile.TarInfo("ml-prove/train.csv")
        info.size = len(data)
        tfile.addfile(info, io.BytesIO(data))
    assert locate_text(ds.locate(raw, "ml-prove/train.csv")) == "a,b\n"
    # a gzip-compressed file
    with gzip.open(raw / "only.data.gz", "wb") as gfile:
        gfile.write(b"zzz\n")
    assert locate_text(ds.locate(raw, "only.data")) == "zzz\n"
    with pytest.raises(FileNotFoundError):
        ds.locate(raw, "absent.data")


def locate_text(path):
    return Path(path).read_text()


def test_unix_compress_files(tmp_path):
    import shutil
    import subprocess
    if shutil.which("compress") is None:
        pytest.skip("the compress command is not installed")
    plain = tmp_path / "f.data"
    plain.write_text("1,2\n3,4\n" * 50)
    subprocess.run(["compress", "-f", str(plain)], check=True)
    assert not plain.exists() and (tmp_path / "f.data.Z").exists()
    assert ds.locate(tmp_path, "f.data").read_text() == "1,2\n3,4\n" * 50


def test_no_uncompress_available_is_a_clear_error(tmp_path, monkeypatch):
    (tmp_path / "g.data.Z").write_bytes(b"\x1f\x9d\x90")
    monkeypatch.setattr(ds.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uncompress"):
        ds.decompress(tmp_path / "g.data.Z")


def test_unsafe_archive_is_refused(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as zfile:
        zfile.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        ds.extract_archive(tmp_path / "evil.zip", tmp_path / "out")
