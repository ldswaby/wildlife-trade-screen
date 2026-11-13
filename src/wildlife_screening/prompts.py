"""Load independently versioned LLM text and native decision questions."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import Advert
from .utils import file_hash

TARGET = "<advertisement_title>"


@dataclass(frozen=True)
class Prompt:
    """Keep one family-specific prompt and its original file hash together."""

    sha256: str
    definition: dict[str, Any]
    family: str = "decision"
    template: str | None = None

    @classmethod
    def load(cls, path: Path) -> "Prompt":
        """Load a fixed prompt without deriving content from evaluation data.

        Args:
            path: Versioned LLM text or native decision YAML file.

        Returns:
            Validated prompt with its file hash.

        Raises:
            ValueError: The file format, condition or content is invalid.
        """
        if path.suffix == ".txt":
            match = re.fullmatch(r"llm_(fewshot|zeroshot)_(v\d+)", path.stem)
            if match is None:
                raise ValueError("LLM prompt filename must identify shot condition and version")
            template = path.read_text(encoding="utf-8").rstrip("\n")
            condition = "few-shot" if match.group(1) == "fewshot" else "zero-shot"
            if not template.strip() or TARGET in template or template.endswith("\nLabel:"):
                raise ValueError(
                    "LLM instructions must be nonempty and contain no target placeholder"
                )
            # Label definitions establish the output vocabulary. The paper need not
            # repeat it in a particular final instruction sentence.
            if "Label 1: relevant -" not in template or "Label 2: not relevant -" not in template:
                raise ValueError("LLM prompt must request relevant/not relevant labels")
            examples = re.findall(r"(?m)^\d+\. Advertisement: (.+)$", template)
            labels = re.findall(r"(?m)^Label: (.+)$", template)
            if condition == "few-shot" and (
                "\nExamples:" not in template
                or not examples
                or len(examples) != len(labels)
                or set(labels) != {"relevant", "not relevant"}
            ):
                raise ValueError("Few-shot LLM template needs relevant/not relevant demonstrations")
            if condition == "zero-shot" and ("\nExamples:" in template or examples or labels):
                raise ValueError("Zero-shot LLM template cannot contain demonstrations")
            return cls(
                file_hash(path),
                {
                    "version": path.stem,
                    "condition": condition,
                    "rendering": f"developer-user-{match.group(2)}",
                },
                "llm",
                template,
            )
        if path.suffix != ".yaml":
            raise ValueError("Decision prompts must be YAML; LLM prompts must be text")
        definition = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(definition, dict) or definition.get("family") != "decision":
            raise ValueError("Decision prompt must declare family: decision")
        if set(definition.get("criteria", {})) != {"relevant", "not relevant"}:
            raise ValueError("Decision prompt must define exactly relevant and not relevant")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in definition["criteria"].values()
        ):
            raise ValueError("Decision choice descriptions must be nonempty text")
        if (
            not isinstance(definition.get("instructions"), str)
            or not definition["instructions"].strip()
        ):
            raise ValueError("Decision prompt needs nonempty instructions")
        examples = definition.get("examples", [])
        if not isinstance(examples, list) or any(
            not isinstance(example, dict)
            or example.get("label") not in definition["criteria"]
            or not isinstance(example.get("text"), str)
            or not example["text"].strip()
            or not isinstance(example.get("explanation"), str)
            for example in examples
        ):
            raise ValueError("Examples require text, a permitted label and an explanation field")
        expected = "few-shot" if examples else "zero-shot"
        if definition.get("condition") != expected:
            raise ValueError("Decision prompt condition must agree with its examples")
        if not isinstance(definition.get("version"), str) or not definition["version"]:
            raise ValueError("Decision prompt needs a version")
        return cls(file_hash(path), definition)

    @property
    def examples(self) -> list[dict[str, str]]:
        """Return fixed native demonstrations, never evaluation labels."""
        if self.family == "llm":
            return [
                {"text": title}
                for title in re.findall(r"(?m)^\d+\. Advertisement: (.+)$", self.template or "")
            ]
        return list(self.definition.get("examples", []))

    @property
    def condition(self) -> str:
        """Identify the fixed shot condition."""
        return str(self.definition["condition"])

    def metadata(self) -> dict[str, Any]:
        """Describe the actual prompt used by an arm."""
        return {
            "condition": self.condition,
            "prompt_family": self.family,
            "example_count": len(self.examples),
            "prompt_version": str(self.definition["version"]),
            "prompt_hash": self.sha256,
            "prompt_rendering": str(self.definition.get("rendering", "unreported")),
        }

    def demonstrations(self) -> str:
        """Render published examples with native choice labels."""
        if not self.examples:
            return ""
        blocks = [
            f"{index}. Advertisement: {example['text']}\n"
            f"Label: {example['label']}"
            + (f"\n{example['explanation']}" if example["explanation"] else "")
            for index, example in enumerate(self.examples, 1)
        ]
        return "\nExamples:\n" + "\n".join(blocks)

    def question(self) -> dict[str, Any]:
        """Build the native choice question from its independent definition."""
        if self.family != "decision":
            raise ValueError("LLM text cannot be used as a native decision question")
        return {
            "type": "choice",
            "instructions": self.definition["instructions"] + self.demonstrations(),
            "criteria": self.definition["criteria"],
        }

    def chat_messages(self, advert: Advert) -> list[dict[str, str]]:
        """Send fixed instructions and allowlisted advert evidence in separate roles."""
        if self.family != "llm" or self.template is None:
            raise ValueError("Native decision prompt cannot be sent as LLM text")
        return [
            {"role": "developer", "content": self.template},
            {"role": "user", "content": advert.text()},
        ]
