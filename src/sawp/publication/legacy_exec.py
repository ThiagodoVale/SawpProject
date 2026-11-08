from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from pathlib import Path

import numpy as np

from sawp.io import inspect_npz, open_npz_memmap


class MemmapArchive:
    def __init__(self, path: Path, original_load) -> None:
        self.path = path
        self.original_load = original_load
        self.files = list(inspect_npz(path))

    def __getitem__(self, name: str):
        if name == "data":
            return open_npz_memmap(self.path, name)
        with self.original_load(self.path, allow_pickle=False) as archive:
            return archive[name]

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def close(self) -> None:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("script")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    script = workspace / args.script
    original_load = np.load

    def compatible_load(file, *load_args, **kwargs):
        path = Path(file)
        if path.suffix.lower() == ".npz" and kwargs.get("mmap_mode") is not None:
            return MemmapArchive(path, original_load)
        return original_load(file, *load_args, **kwargs)

    np.load = compatible_load
    os.chdir(workspace)
    for name in list(sys.modules):
        if name == "sawp" or name.startswith("sawp."):
            del sys.modules[name]
    sys.path.insert(0, str(workspace))
    importlib.import_module("sawp")
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
