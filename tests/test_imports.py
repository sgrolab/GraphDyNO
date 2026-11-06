"""Import smoke test for the graph_dyno + dino packages.

Walks every submodule and imports it, surfacing dotted-path typos and
circular-import regressions in <1s. After any restructure or rename, this
should remain the cheapest signal that the package layout is intact.
"""
import importlib
import pkgutil

import pytest

import dino
import graph_dyno


def _all_submodules(pkg):
    yield pkg.__name__
    for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        yield m.name


@pytest.mark.parametrize("module_name", sorted(set(_all_submodules(graph_dyno)) | set(_all_submodules(dino))))
def test_module_importable(module_name):
    importlib.import_module(module_name)
