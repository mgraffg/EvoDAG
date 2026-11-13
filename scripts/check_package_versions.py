#!/usr/bin/env python3
"""Check EvoDAG's local version against PyPI.

`publish-pypi.yml` only checks, at the moment a tag is pushed, that the tag
matches the package version. Nothing checks *before* that whether the local
version is actually ahead of what PyPI has, or whether it jumped more than
one release past it.

This reports PyPI's latest version, the local version (read from
`pyproject.toml`'s dynamic `EvoDAG.__version__` attribute without importing
the package, so it needs none of its dependencies installed), and whether the
local version is a valid single-step jump ahead of PyPI: the next patch, or
the next minor with patch reset to 0. A package PyPI has never published
(`pypi_latest_version` returns `None`) always passes this check, since there
is nothing to jump ahead of yet.

The repository is the package, so `PACKAGE_DIRS` maps its one name to the
repository root.

Usage:

    python scripts/check_package_versions.py
"""

import argparse
import ast
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

# The repository *is* `EvoDAG`: one package, at the root.
PACKAGE_DIRS = {"EvoDAG": ROOT}


def pypi_latest_version(package: str) -> str | None:
    """The latest version PyPI reports for `package`, or `None` if it has
    never been published (a 404 from PyPI's JSON API)."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)["info"]["version"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def local_version(package_dir: Path) -> str:
    """The version `pyproject.toml` resolves dynamically, read straight out
    of the source file its `attr` points at (no import, so this needs
    none of the package's own dependencies installed)."""
    pyproject = tomllib.load(open(package_dir / "pyproject.toml", "rb"))
    attr = pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    module_name, attr_name = attr.rsplit(".", 1)
    module_path = package_dir / module_name.replace(".", "/") / "__init__.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == attr_name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{attr} not found in {module_path}")


def next_version(version: Version) -> tuple[Version, Version]:
    """The two versions immediately ahead of `version`: the next patch, and
    the next minor with patch reset to 0."""
    release = version.release
    major, minor, patch = (release + (0, 0, 0))[:3]
    next_patch = Version(f"{major}.{minor}.{patch + 1}")
    next_minor = Version(f"{major}.{minor + 1}.0")
    return next_patch, next_minor


def check_package(package: str) -> tuple[str, str | None, str, bool, str]:
    """Returns (package, pypi_version, local_version, ok, detail)."""
    pypi = pypi_latest_version(package)
    package_dir = PACKAGE_DIRS[package]
    local = local_version(package_dir)
    local_v = Version(local)

    if pypi is None:
        return package, pypi, local, True, "never published"

    pypi_v = Version(pypi)
    if local_v <= pypi_v:
        return package, pypi, local, False, f"local is not ahead of PyPI ({pypi})"

    next_patch, next_minor = next_version(pypi_v)
    if local_v not in (next_patch, next_minor):
        return (
            package,
            pypi,
            local,
            False,
            f"jumps more than one release past PyPI ({pypi}); "
            f"expected {next_patch} or {next_minor}",
        )
    return package, pypi, local, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        metavar="NAME",
        choices=sorted(PACKAGE_DIRS),
        help="check only this package (repeatable); default: every package in PACKAGE_DIRS",
    )
    args = parser.parse_args()

    packages = args.packages or sorted(PACKAGE_DIRS)

    rows = [check_package(package) for package in packages]

    name_width = max(len(row[0]) for row in rows)
    print(f"{'package':<{name_width}}  {'pypi':<10}  {'local':<10}  verdict")
    all_ok = True
    for package, pypi, local, ok, detail in rows:
        all_ok = all_ok and ok
        verdict = "OK" if ok else "FAIL"
        print(f"{package:<{name_width}}  {pypi or '-':<10}  {local:<10}  {verdict}: {detail}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
