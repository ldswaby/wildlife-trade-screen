"""Translate frozen campaign settings into individual scoring commands."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig


@dataclass(frozen=True)
class CampaignCondition:
    """Describe one subprocess invocation and the evidence expected from its audited arms."""

    task: str
    input_mode: str
    shot: str
    prompts: dict[str, Path]
    models: dict[str, Path]
    label_review: str | None
    concurrency: dict[str, int] | None
    output: Path
    ceiling: float
    dry_run: bool

    @classmethod
    def from_audit(
        cls,
        audit: dict[str, Any],
        input_mode: str,
        shot: str,
        root: Path,
        ceiling: float,
        dry_run: bool,
    ) -> "CampaignCondition":
        """Resolve one condition from the frozen, backwards-compatible audit."""
        return cls(
            task=audit["task"],
            input_mode=input_mode,
            shot=shot,
            prompts={
                family: Path(info["path"]) for family, info in audit["prompt_files"][shot].items()
            },
            # The frozen audit, not the current MODEL_PATHS, defines a campaign's models.
            models={name: Path(info["path"]) for name, info in audit["model_files"].items()},
            label_review=audit["label_review"],
            concurrency=audit.get("runtime_concurrency"),
            output=root / input_mode,
            ceiling=ceiling,
            dry_run=dry_run,
        )

    def command(self) -> list[str]:
        """Render the scoring CLI arguments in their established order."""
        args = [
            sys.executable,
            "-m",
            "wildlife_screening.cli",
            "run",
            "--task",
            self.task,
            "--llm-prompt",
            str(self.prompts["llm"]),
            "--decision-prompt",
            str(self.prompts["decision"]),
            "--output",
            str(self.output),
            "--cache",
            "outputs/cache",
            "--concurrency",
            "1",
            "--spend-ceiling",
            str(self.ceiling),
        ]
        if self.label_review:
            args.extend(["--label-review", self.label_review])
        if self.input_mode == "title_and_desc":
            args.append("--include-description")
        if self.input_mode == "combined_text":
            args.append("--combined-text")
        if self.shot == "zero-shot":
            args.append("--zero-shot")
        if self.dry_run:
            args.append("--dry-run")
        for name, path in self.models.items():
            args.extend(["--model", str(path)])
            if self.concurrency is not None:
                arm = ModelConfig.load(path).arm
                args.extend(["--model-concurrency", arm, str(self.concurrency[name])])
        return args
