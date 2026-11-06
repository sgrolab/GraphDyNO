#!/usr/bin/env python3
"""
Minimal build script for PolyHoop.

Usage:
    python build.py                    # auto: Intel icpx if available, else c++
    POLYHOOP_CXX=c++ python build.py   # force a specific compiler
    POLYHOOP_CXX=mpiicpx python build.py

Then import:
    import polyhoop

Toolchains are tried in order and the first one that actually produces a working
.so wins, so a missing or broken Intel install silently falls back to the stock
c++ build that has always been used here.

Why Intel is preferred when present: the hot loops are OpenMP and
memory-bandwidth bound, and libgomp's thread placement is fixed per process and
effectively arbitrary — measured 2.04x spread in end-to-end s/frame across
identical nodes. Intel's libiomp5 places threads far better.
icpx also emits an AVX-512 path for emeraldrapids.

Note on MPI: `mpiicpx` is accepted, but it is only icpx plus -I/-lmpi. Nothing
in polyhoop_bindings.cpp calls MPI (0 MPI symbols; the parallelism is 6 OpenMP
pragmas), so an MPI-wrapped build is functionally identical to the icpx build.
Running it under `mpirun -n N` would start N independent copies of the whole
simulation, not one distributed simulation.
"""

import os
import subprocess
import sys
import sysconfig


def _probe(cmd):
    """True if `cmd` can compile a trivial OpenMP translation unit."""
    try:
        r = subprocess.run(
            cmd + ["-E", "-x", "c++", "-"],
            input="", capture_output=True, text=True, timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False


def _toolchains():
    """(name, compiler, extra_flags) candidates, best first.

    Intel notes:
      -fiopenmp            Intel's OpenMP (libiomp5), not libgomp
      -fp-model=precise    icpx defaults to fast math, which reorders FP ops;
                           keep IEEE semantics so results stay comparable to
                           the gcc build
      -axCORE-AVX512       add a runtime-dispatched AVX-512 path while keeping a
                           portable baseline, so the .so still loads on hosts
                           without AVX-512 (do NOT use -march/-x native: this is
                           often built on a login node and run on compute nodes)
      -qopenmp-link=static keep libiomp5 inside the .so so LSF jobs don't need
                           `module load oneapi` at runtime
    """
    forced = os.environ.get("POLYHOOP_CXX")
    if forced:
        intel = os.path.basename(forced).startswith(("icpx", "icpc", "mpiicp"))
        if intel:
            return [(f"forced {forced} (Intel)", forced,
                     ["-fiopenmp", "-fp-model=precise", "-axCORE-AVX512",
                      "-static-intel", "-qopenmp-link=static"])]
        return [(f"forced {forced}", forced, ["-fopenmp"])]

    return [
        ("Intel icpx + libiomp5", "icpx",
         ["-fiopenmp", "-fp-model=precise", "-axCORE-AVX512",
          "-static-intel", "-qopenmp-link=static"]),
        ("c++ + libgomp", "c++", ["-fopenmp"]),
        ("c++ (no OpenMP)", "c++", []),
    ]


def _try_build(name, compiler, extra, source, output, pybind_includes):
    cmd = [compiler, "-O3", "-shared", "-std=c++17", "-fPIC",
           "-fvisibility=hidden", *extra,
           *pybind_includes, source, "-o", output]
    print(f"\n--- trying: {name}")
    print(f"    {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except FileNotFoundError:
        print(f"    skipped: {compiler} not on PATH")
        return False
    except Exception as e:
        print(f"    skipped: {e}")
        return False
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-6:]
        print("    FAILED:\n      " + "\n      ".join(tail))
        return False
    if not os.path.exists(output):
        print("    FAILED: compiler reported success but produced no output")
        return False
    print(f"    built: {output}")
    return True


def _verify(script_dir):
    """Import the freshly built module in a subprocess with a SANITIZED env.

    A build can link fine and still fail at import, which is what the fallback
    exists for. The env must be sanitized because build.py is typically run
    after `module load oneapi`, which puts the Intel libs on LD_LIBRARY_PATH —
    so an unsanitized check happily accepts a .so that then dies with
    "libsvml.so: cannot open shared object file" on a compute node that has no
    module loaded. Strip the compiler's own library paths and require the .so to
    stand on its own.
    """
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        kept = [p for p in env.get(var, "").split(":")
                if p and "INTEL" not in p.upper() and "oneapi" not in p.lower()]
        if kept:
            env[var] = ":".join(kept)
        else:
            env.pop(var, None)

    probe = (
        "import polyhoop; "
        "polyhoop.set_num_threads(2); "
        "e = polyhoop.Ensemble(); "
        "print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", probe], cwd=script_dir, env=env,
                       capture_output=True, text=True, timeout=300)
    if r.returncode == 0 and "OK" in r.stdout:
        return True
    tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-5:]
    print("    import check FAILED:\n      " + "\n      ".join(tail))
    return False


def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(script_dir, "polyhoop_bindings.cpp")

    if not os.path.exists(source):
        print(f"Error: {source} not found")
        sys.exit(1)

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = os.path.join(script_dir, f"polyhoop{ext_suffix}")

    try:
        import pybind11  # noqa: F401
    except ImportError:
        print("Installing pybind11...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])

    pybind_includes = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--includes"], text=True
    ).strip().split()

    # Keep the working .so until a replacement is proven good, so a failed
    # experiment never leaves the repo without an importable module.
    backup = output + ".prev"
    had_previous = os.path.exists(output)
    if had_previous:
        os.replace(output, backup)

    try:
        for name, compiler, extra in _toolchains():
            if not _probe([compiler] + [f for f in extra if f.startswith("-f")]):
                print(f"\n--- trying: {name}\n    skipped: {compiler} cannot compile "
                      f"with {' '.join(f for f in extra if f.startswith('-f')) or '(no flags)'}")
                continue
            if not _try_build(name, compiler, extra, source, output, pybind_includes):
                continue
            if not _verify(script_dir):
                os.remove(output)
                continue
            print(f"\nBuilt with: {name}")
            print(f"Output: {output}")
            if had_previous:
                os.remove(backup)
            return

        # Nothing worked — restore whatever was there before.
        if had_previous:
            os.replace(backup, output)
            print("\nAll toolchains failed; restored the previous .so.")
        sys.exit("No working toolchain found.")
    except BaseException:
        if had_previous and not os.path.exists(output):
            os.replace(backup, output)
        raise


if __name__ == "__main__":
    build()
