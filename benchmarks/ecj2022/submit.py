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
"""Submit the selection heuristics experiments to Slurm.

    python benchmarks/ecj2022/submit.py --dry-run
    python benchmarks/ecj2022/submit.py --datasets iris wine --configs agr-rnd fit-fit
    python benchmarks/ecj2022/submit.py --smoke
    python benchmarks/ecj2022/submit.py
    python benchmarks/ecj2022/submit.py --status

The prepared datasets and their manifest come from ``datasets.py``. There is
one job array per dataset, with one task per pending configuration (a unit of
``run.py``), plus a one-task job per dataset for the scikit-learn
classifiers. The cores and the memory of the array follow the size of the
dataset (see :func:`member_mb` and :func:`resources`). Units that have a
result are left out, and unfinished ones resume at the member level, so
submitting again is always safe. A dataset that still has jobs of this
campaign in the queue is skipped.

The full campaign is thousands of CPU hours. ``--smoke`` runs the five
smallest datasets only.
"""
import argparse
import getpass
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import configs  # noqa: E402
import run as runner  # noqa: E402

SMOKE_DATASETS = ["fertility", "iris", "tae", "wine", "parkinsons"]
POPSIZE = 4000
QUICK_POPSIZE = configs.QUICK["popsize"]
MEMORY_FACTOR = 1.5
BASE_MB = 1500
SKLEARN_MB = 16000
JOB_PREFIX = "ecj-"


# -- resources -------------------------------------------------------------

def member_mb(n_classes, n_train, popsize=POPSIZE):
    """Memory in megabytes requested for one member: 1.5 times the outputs of
    the population (``8 * popsize * n_classes * n_train`` bytes) plus 1.5 GB."""
    return int(math.ceil(MEMORY_FACTOR * 8 * popsize * n_classes * n_train
                         / 1e6)) + BASE_MB


def resources(n_classes, n_train, quick=False, max_workers=8,
              max_mem_mb=120000):
    """``(workers, mem_mb)`` of the array of a dataset: as many members at
    the same time as ``max_workers`` and ``max_mem_mb`` allow (at least one),
    and the memory of all of them."""
    one = member_mb(n_classes, n_train,
                    QUICK_POPSIZE if quick else POPSIZE)
    workers = max(1, min(max_workers, max_mem_mb // one))
    return workers, workers * one


# -- manifest and state of the results -------------------------------------

def read_manifest(root):
    "Entries of ``data/manifest.json`` by name; a clear error if it is missing"
    path = Path(root) / "data" / "manifest.json"
    if not path.exists():
        raise SystemExit("%s does not exist: prepare the data with "
                         "benchmarks/ecj2022/datasets.py" % path)
    with open(path) as fpt:
        return {entry["name"]: entry for entry in json.load(fpt)}


def select_datasets(manifest, names):
    "The requested datasets (all by default), each of them prepared"
    names = list(names) if names else list(manifest)
    bad = [n for n in names
           if n not in manifest or "error" in manifest[n]]
    if bad:
        raise SystemExit("not prepared: %s; run "
                         "benchmarks/ecj2022/datasets.py first"
                         % ", ".join(bad))
    return names


def unit_done(root, dataset, config, quick=False):
    "Whether ``result.json`` of a unit exists"
    return (runner.unit_dir(root, dataset, config, quick)
            / "result.json").exists()


def sklearn_done(root, dataset, quick=False):
    "Whether the sixteen classifiers of a dataset have a result"
    directory = runner.unit_dir(root, dataset, "sklearn", quick)
    return all((directory / (n + ".json")).exists()
               for n in configs.SKLEARN_NAMES)


def pending_configs(root, dataset, config_names, quick=False):
    "The configurations without a result"
    return [c for c in config_names if not unit_done(root, dataset, c, quick)]


# -- the jobs --------------------------------------------------------------

def running_jobs():
    "Names of the jobs of this campaign in the queue (``[]`` without Slurm)"
    if shutil.which("squeue") is None:
        return []
    try:
        done = subprocess.run(
            ["squeue", "-u", os.environ.get("USER") or getpass.getuser(),
             "-h", "-o", "%j %T"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    jobs = []
    for line in done.stdout.decode().splitlines():
        parts = line.split()
        if parts and parts[0].startswith(JOB_PREFIX):
            jobs.append(parts[0])
    return jobs


def job_name(dataset, sklearn=False):
    "Name of the job of a dataset"
    return "%s%s%s" % (JOB_PREFIX, dataset, "-sklearn" if sklearn else "")


def plan(root, manifest, datasets, config_names, quick=False, max_workers=8,
         max_mem_mb=120000, max_parallel=None, with_sklearn=True,
         busy=()):
    """The jobs to submit: a list of dictionaries with the ``sbatch``
    argument list (``argv``), the tasks and the resources. Datasets whose
    jobs are in ``busy`` are left out."""
    root = Path(root)
    run_py = str(HERE / "run.py")
    jobs = []
    for dataset in datasets:
        entry = manifest[dataset]
        base = [sys.executable, run_py, "--dataset", dataset,
                "--root", str(root)]
        if quick:
            base.append("--quick")
        tasks = pending_configs(root, dataset, config_names, quick)
        if tasks and job_name(dataset) not in busy:
            workers, mem = resources(entry["n_classes"], entry["n_train"],
                                     quick, max_workers, max_mem_mb)
            task_file = root / "logs" / (dataset + ".tasks.json")
            array = "0-%d" % (len(tasks) - 1)
            if max_parallel:
                array += "%%%d" % max_parallel
            wrap = base + ["--task-file", str(task_file),
                           "--workers", str(workers)]
            jobs.append({
                "dataset": dataset, "kind": "evodag", "tasks": tasks,
                "task_file": task_file, "workers": workers, "cpus": workers,
                "mem_mb": mem,
                "argv": ["sbatch", "--parsable",
                         "--job-name=" + job_name(dataset),
                         "--array=" + array,
                         "--cpus-per-task=%d" % workers,
                         "--mem=%dM" % mem,
                         "--output=%s" % (root / "logs" / "%x-%A_%a.out"),
                         "--wrap", shlex.join(wrap)]})
        if (with_sklearn and not sklearn_done(root, dataset, quick)
                and job_name(dataset, True) not in busy):
            jobs.append({
                "dataset": dataset, "kind": "sklearn", "tasks": ["sklearn"],
                "task_file": None, "workers": 1, "cpus": 1,
                "mem_mb": SKLEARN_MB,
                "argv": ["sbatch", "--parsable",
                         "--job-name=" + job_name(dataset, True),
                         "--cpus-per-task=1", "--mem=%dM" % SKLEARN_MB,
                         "--output=%s" % (root / "logs" / "%x-%j.out"),
                         "--wrap", shlex.join(base + ["--sklearn"])]})
    return jobs


def submit(jobs, root, dry_run=False, log=print):
    """Write the task files and submit (or just print, with ``dry_run``) the
    jobs; every submission is appended to ``logs/submissions.jsonl``."""
    root = Path(root)
    logs = root / "logs"
    for job in jobs:
        logs.mkdir(parents=True, exist_ok=True)
        if job["task_file"] is not None:
            runner.write_json(job["task_file"], job["tasks"])
        log(shlex.join(job["argv"]))
        if dry_run:
            continue
        done = subprocess.run(job["argv"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        if done.returncode != 0:
            raise SystemExit("sbatch failed for %s: %s"
                             % (job["dataset"], done.stderr.decode().strip()))
        job["job_id"] = done.stdout.decode().strip().split(";")[0]
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "job_id": job["job_id"], "dataset": job["dataset"],
                  "kind": job["kind"], "tasks": job["tasks"],
                  "cpus": job["cpus"], "mem_mb": job["mem_mb"]}
        with open(logs / "submissions.jsonl", "a") as fpt:
            fpt.write(json.dumps(record) + "\n")
        log("submitted job %s (%s)" % (job["job_id"], job_name(
            job["dataset"], job["kind"] == "sklearn")))


# -- status ----------------------------------------------------------------

def unit_state(root, dataset, config, members, quick=False):
    "``(finished, members_done, n_errors)`` of one EvoDAG unit"
    directory = runner.unit_dir(root, dataset, config, quick)
    if (directory / "result.json").exists():
        return True, members, 0
    if not directory.exists():
        return False, 0, 0
    done = len(runner.done_members(directory, members))
    errors = len(list(directory.glob("member-*.error.txt")))
    return False, done, errors


def status(root, datasets, config_names, quick=False, log=print):
    "Print the state of every dataset; returns the totals"
    members = configs.n_members(quick)
    queue = running_jobs()
    totals = {"units": 0, "finished": 0, "partial": 0, "failed": 0}
    for dataset in datasets:
        finished = partial = failed = 0
        notes = []
        for config in config_names:
            ok, done, errors = unit_state(root, dataset, config, members,
                                          quick)
            if ok:
                finished += 1
            else:
                if errors:
                    failed += 1
                    notes.append("%s: %d members with errors"
                                 % (config, errors))
                elif done:
                    partial += 1
                    notes.append("%s: %d/%d members" % (config, done,
                                                        members))
        n_sklearn = sum((runner.unit_dir(root, dataset, "sklearn", quick)
                         / (n + ".json")).exists()
                        for n in configs.SKLEARN_NAMES)
        units = len(config_names) + 1
        if n_sklearn == len(configs.SKLEARN_NAMES):
            finished += 1
        elif n_sklearn:
            partial += 1
            notes.append("sklearn: %d/%d classifiers"
                         % (n_sklearn, len(configs.SKLEARN_NAMES)))
        jobs = [j for j in queue if j in (job_name(dataset),
                                          job_name(dataset, True))]
        log("%-20s %2d/%d finished, %d partial, %d failed%s%s"
            % (dataset, finished, units, partial, failed,
               "; in the queue: " + ", ".join(jobs) if jobs else "",
               "".join("\n    " + n for n in notes)))
        for key, value in (("units", units), ("finished", finished),
                           ("partial", partial), ("failed", failed)):
            totals[key] += value
    log("total: %(finished)d/%(units)d units finished, %(partial)d partial, "
        "%(failed)d failed; %(jobs)d jobs of the campaign in the queue"
        % dict(totals, jobs=len(queue)))
    return totals


# -- command line ----------------------------------------------------------

def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Submit the selection heuristics experiments to Slurm.")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="only these datasets (default: all prepared)")
    parser.add_argument("--configs", nargs="+", metavar="NAME",
                        help="only these configurations (default: the 22)")
    parser.add_argument("--smoke", action="store_true",
                        help="the five smallest datasets: %s"
                             % ", ".join(SMOKE_DATASETS))
    parser.add_argument("--quick", action="store_true",
                        help="tiny parameters (see run.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the sbatch commands and submit nothing")
    parser.add_argument("--status", action="store_true",
                        help="print what is done, running and missing")
    parser.add_argument("--no-sklearn", action="store_true",
                        help="do not submit the scikit-learn jobs")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="members trained at the same time in a task")
    parser.add_argument("--max-mem-mb", type=int, default=120000,
                        help="memory of a task in megabytes")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="tasks of an array that run at the same time")
    parser.add_argument("--root", help="experiments directory (default: "
                        "$%s or <repository>/experiments)" % runner.ENV_ROOT)
    args = parser.parse_args(argv)
    root = runner.resolve_root(args.root)
    config_names = args.configs or list(configs.CONFIGS)
    unknown = [c for c in config_names if c not in configs.CONFIGS]
    if unknown:
        parser.error("unknown configurations: %s" % ", ".join(unknown))
    manifest = read_manifest(root)
    names = args.datasets or (SMOKE_DATASETS if args.smoke else None)
    datasets = select_datasets(manifest, names)
    if args.status:
        status(root, datasets, config_names, args.quick)
        return 0
    if not args.dry_run and shutil.which("sbatch") is None:
        raise SystemExit("sbatch is not on the PATH: this needs a Slurm "
                         "login node (use --dry-run to see the commands)")
    jobs = plan(root, manifest, datasets, config_names, args.quick,
                args.max_workers, args.max_mem_mb, args.max_parallel,
                not args.no_sklearn, busy=set(running_jobs()))
    if not jobs:
        print("nothing to submit")
        return 0
    submit(jobs, root, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
