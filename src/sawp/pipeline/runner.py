from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..config import ProjectConfig
from .manifest import Manifest, Stage
from .provenance import runtime_identity, sha256, write_json


class PipelineRunner:
    def __init__(self, config: ProjectConfig, manifest: Manifest) -> None:
        self.config = config
        self.manifest = manifest

    def _expected_paths(self, stage: Stage) -> list[Path]:
        base = (
            self.config.source.parent.parent
            if stage.executor == "native"
            else self.config.paths.workspace
        )
        return [base / relative for relative in stage.expected]

    def _order(self, target: str) -> list[Stage]:
        requested = (
            [stage.id for stage in self.manifest.stages] if target == "publication" else [target]
        )
        order: list[Stage] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visited:
                return
            if identifier in visiting:
                raise ValueError(f"cyclic pipeline dependency at {identifier}")
            visiting.add(identifier)
            stage = self.manifest.stage(identifier)
            for dependency in stage.depends_on:
                visit(dependency)
            visiting.remove(identifier)
            visited.add(identifier)
            order.append(stage)

        for identifier in requested:
            visit(identifier)
        return order

    def _command(self, stage: Stage) -> list[str]:
        python = (
            sys.executable if self.config.runtime.python == "auto" else self.config.runtime.python
        )
        if stage.executor == "native":
            project_root = self.config.source.parent.parent
            return [python, str(project_root / stage.script), *stage.args]
        if stage.executor != "legacy":
            raise ValueError(f"unsupported stage executor: {stage.executor}")
        return [
            python,
            "-m",
            "sawp.pipeline.legacy_exec",
            str(self.config.paths.workspace),
            stage.script,
            "--",
            *stage.args,
        ]

    def run(self, target: str, *, resume: bool = False, dry_run: bool = False) -> list[dict]:
        reports: list[dict] = []
        for stage in self._order(target):
            expected = self._expected_paths(stage)
            command = self._command(stage)
            if resume and expected and all(path.is_file() for path in expected):
                reports.append({"stage": stage.id, "status": "skipped", "reason": "complete"})
                continue
            if dry_run:
                reports.append(
                    {
                        "stage": stage.id,
                        "status": "dry-run",
                        "command": command,
                        "expected": [str(path) for path in expected],
                    }
                )
                continue
            script_base = (
                self.config.source.parent.parent
                if stage.executor == "native"
                else self.config.paths.workspace
            )
            missing_script = script_base / stage.script
            if not missing_script.is_file():
                raise FileNotFoundError(missing_script)
            environment = os.environ.copy()
            environment.update({"MPLBACKEND": "Agg", "PYTHONUTF8": "1"})
            existing_pythonpath = environment.get("PYTHONPATH", "")
            project_source = self.config.source.parent.parent / "src"
            environment["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (
                    str(project_source),
                    str(self.config.paths.workspace),
                    existing_pythonpath,
                )
                if value
            )
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=self.config.source.parent.parent,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            duration = time.perf_counter() - started
            report = {
                **runtime_identity(),
                "stage": stage.id,
                "description": stage.description,
                "command": command,
                "duration_seconds": round(duration, 3),
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "artifacts": [],
            }
            for path in expected:
                artifact = {"path": str(path), "exists": path.is_file()}
                if path.is_file():
                    artifact.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
                report["artifacts"].append(artifact)
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            write_json(
                self.config.paths.results / "logs" / f"{timestamp}_{stage.id}.json",
                report,
            )
            reports.append(report)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"stage {stage.id} failed with code {completed.returncode}; "
                    f"see results/logs for details"
                )
            if self.config.runtime.strict:
                missing = [path for path in expected if not path.is_file()]
                if missing:
                    raise RuntimeError(f"stage {stage.id} did not create: {missing}")
        if (
            target == "publication"
            and not dry_run
            and any(stage.executor == "legacy" for stage in self.manifest.stages)
        ):
            self.collect_products()
        return reports

    def collect_products(self, *, use_reference: bool = False) -> list[Path]:
        destination = self.config.paths.results / "figures"
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for product in self.manifest.products:
            source = (
                self.config.paths.reference_figures / product.reference
                if use_reference
                else self.config.paths.workspace / product.source
            )
            if not source.is_file():
                if self.config.runtime.strict:
                    raise FileNotFoundError(source)
                continue
            suffix = source.suffix if source.suffix else Path(product.reference).suffix
            target = destination / f"{Path(product.reference).stem}{suffix}"
            shutil.copy2(source, target)
            outputs.append(target)
        return outputs
