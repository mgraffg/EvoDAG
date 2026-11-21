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
"""Profile the fit of one member of an experiment.

    python benchmarks/ecj2022/profile_fit.py --dataset optdigits --config agr-rnd
    python benchmarks/ecj2022/profile_fit.py --dataset adult --config agr-rnd --max-seconds 900

One member (``EvoDAG.model.EvoDAG(seed=seed, **params)``, with the parameters
that ``run.py`` uses for the configuration, without ``--quick``) is fitted on
a prepared dataset under ``cProfile`` and the script prints the shape of the
data, the fit time, the number of individuals created and the functions that
take the most time, by own time and by cumulative time, restricted to the
package and the NumPy calls. ``--max-seconds`` is passed to the engine as its
``time_limit``: a profile of a large dataset is cut after that long, which
is still a valid profile (the output says so). ``--output`` also saves the raw
statistics (a ``pstats`` dump). Every fit uses one core.

cProfile sees Python-level calls and the entry points of the Cython kernels,
which is enough to rank the functions of the engine.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import cProfile  # noqa: E402
import io  # noqa: E402
import pstats  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configs  # noqa: E402
import run as runner  # noqa: E402

# the package and the NumPy calls
RESTRICT = r"EvoDAG|numpy"


def profile_member(X, y, config, seed=0, max_seconds=None):
    """Fit one member under cProfile. Returns a dictionary with the profiler
    (``profile``), the fit time (``seconds``), the individuals created
    (``individuals``) and whether the fit was cut (``truncated``)."""
    from EvoDAG.base import EvoDAG as Engine
    params = configs.params_for(config)
    if max_seconds is not None:
        params["time_limit"] = max_seconds
    engine = Engine.init(seed=seed, **params)
    profile = cProfile.Profile()
    t0 = time.perf_counter()
    profile.enable()
    try:
        engine.fit(X, y)
    finally:
        profile.disable()
    seconds = time.perf_counter() - t0
    return {"profile": profile, "seconds": seconds,
            "individuals": len(engine.population.hist),
            "truncated": max_seconds is not None and seconds >= max_seconds,
            "params": params}


def format_stats(profile, top=30):
    "The tables of own time and of cumulative time as text"
    out = []
    for key, title in (("tottime", "own time"),
                       ("cumulative", "cumulative time")):
        stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stream)
        stats.sort_stats(key).print_stats(RESTRICT, top)
        out.append("Top %d functions by %s:\n%s" % (top, title,
                                                    stream.getvalue()))
    return "\n".join(out)


def main(argv=None):
    "Command line; returns the exit status"
    parser = argparse.ArgumentParser(
        description="Profile the fit of one member of an experiment.")
    parser.add_argument("--dataset", required=True,
                        help="name of a prepared dataset")
    parser.add_argument("--config", required=True,
                        help="a configuration of configs.py")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float,
                        help="cut the fit after this many seconds")
    parser.add_argument("--top", type=int, default=30,
                        help="functions shown in each table")
    parser.add_argument("--output",
                        help="save the raw pstats statistics to this file")
    parser.add_argument("--root", help="experiments directory (default: "
                        "$%s or <repository>/experiments)" % runner.ENV_ROOT)
    args = parser.parse_args(argv)
    if args.config not in configs.CONFIGS:
        parser.error("unknown configuration %r" % args.config)
    root = runner.resolve_root(args.root)
    X, y, _, _ = runner.load_data(root, args.dataset)
    result = profile_member(X, y, args.config, args.seed, args.max_seconds)
    print("dataset %s: %d examples, %d variables, %d classes"
          % (args.dataset, X.shape[0], X.shape[1], len(set(y.tolist()))))
    print("configuration %s, seed %d" % (args.config, args.seed))
    print("fit time: %.2f s (under cProfile)" % result["seconds"])
    print("individuals created: %d" % result["individuals"])
    if result["truncated"]:
        print("TRUNCATED: the fit was cut after --max-seconds %g, compare "
              "the individuals created, not the time" % args.max_seconds)
    print()
    print(format_stats(result["profile"], args.top))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result["profile"].dump_stats(str(path))
        print("statistics saved in %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
