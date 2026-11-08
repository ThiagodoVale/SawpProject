from __future__ import annotations

import argparse
import ast
import importlib
import os
import runpy
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path


def _symbol_index(package_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for source in sorted(package_dir.glob("*.py")):
        if source.name == "__init__.py":
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index.setdefault(node.name, source.stem)
    return index


def _legacy_package(package_dir: Path) -> types.ModuleType:
    index = _symbol_index(package_dir)
    package = types.ModuleType("sawp")
    package.__file__ = str(package_dir / "__init__.py")
    package.__path__ = [str(package_dir)]
    package.__package__ = "sawp"
    package.__spec__ = ModuleSpec("sawp", loader=None, is_package=True)
    package.__version__ = "0.1.0-compat"

    def resolve(name: str):
        module_name = index.get(name)
        if module_name is None:
            raise AttributeError(f"legacy sawp package has no public symbol {name!r}")
        module = importlib.import_module(f"sawp.{module_name}")
        value = getattr(module, name)
        setattr(package, name, value)
        return value

    package.__getattr__ = resolve
    package.__all__ = sorted(index)
    return package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a historical SAWP producer lazily")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    script = args.script if args.script.is_absolute() else workspace / args.script
    script = script.resolve()
    package_dir = workspace / "sawp"
    if not script.is_file():
        raise FileNotFoundError(script)
    if not package_dir.is_dir():
        raise FileNotFoundError(package_dir)

    sys.modules["sawp"] = _legacy_package(package_dir)
    if str(workspace) not in sys.path:
        sys.path.append(str(workspace))
    os.chdir(workspace)
    forwarded = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
