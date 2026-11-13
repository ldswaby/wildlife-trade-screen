"""Published aggregates can support conditional points, never invented curves."""

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wildlife-screening-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import (  # noqa: E402
    policy_evaluate,
    policy_roc_curve,
    published_false_positive_rate,
)
from .types import ScoreResult  # noqa: E402


def plot_roc(
    labels: list[int],
    arms: Mapping[str, list[ScoreResult]],
    published: list[dict[str, Any]],
    output: Path,
) -> None:
    """Show ROC trade-offs without inventing curves for binary or published arms.

    Args:
        labels: Aligned gold labels shared by all measured arms.
        arms: Full aligned results; recorded failures are always reviewed.
            Binary models and all-failure arms have one policy point.
        published: Aggregate rows; P/R points are labelled as conditional on
            positive-class averaging and compatible prevalence.
        output: Directory receiving figure1.png and figure1.svg.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    for name, results in arms.items():
        metric = policy_evaluate(labels, results)
        if sum(labels) == 0 or sum(labels) == len(labels):
            continue
        # Mixing binary decisions with probabilities does not establish a full ranking.
        if any(r.score_type == "binary" for r in results) or not metric["valid_predictions"]:
            ax.scatter(
                metric["false_positive_rate"] * 100,
                metric["positive_recall"] * 100,
                s=65,
                clip_on=bool(metric["valid_predictions"]),
                label=(
                    f"{name} (binary point)"
                    if metric["valid_predictions"]
                    else f"{name} (all failures reviewed)"
                ),
            )
        else:
            points = policy_roc_curve(labels, results)
            ax.step(
                [x * 100 for x, _ in points],
                [y * 100 for _, y in points],
                where="post",
                linewidth=2,
                label=name,
            )
    ax.plot([0, 100], [0, 100], color="#777777", linestyle=":", label="Random review (expected)")
    ax.axhline(95, color="#999999", linestyle="--", label="95% recall reference")
    if 0 < sum(labels) < len(labels):
        for row in published:
            if row.get("published_precision") is None or row.get("published_recall") is None:
                continue
            image_input = row.get("input_modality") == "text-image"
            color = "#2367b1" if row["arm"].startswith("ACTL ") else "#e27422"
            false_positive_rate = published_false_positive_rate(
                row["published_precision"], row["published_recall"], sum(labels), len(labels)
            )
            ax.scatter(
                false_positive_rate * 100,
                row["published_recall"] * 100,
                marker="s" if image_input else "x",
                s=85,
                color=color,
                label=f"{row['arm']} (conditional published point)",
            )
    ax.set(
        xlabel="Irrelevant adverts sent for review (false-positive rate, FPR, %)",
        ylabel="Relevant adverts found (recall, TPR, %)",
        xlim=(0, 100),
        ylim=(0, 102),
        title=(
            "Figure 1. ROC: relevant adverts found (recall) versus "
            "irrelevant adverts sent for review (FPR)"
        ),
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=8)
    footnote = "Review on failure: failed adverts are always reviewed; curves are descriptive."
    if published:
        footnote = (
            "Published points require positive-class P/R and compatible prevalence.\n"
            "X = text; square = text + image (different input). F1-only rows are omitted.\n"
            "FPR is derived without rounding counts and assumes compatible prevalence.\n" + footnote
        )
    fig.text(0.09, 0.02, footnote, fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"figure1.{extension}", dpi=180)
    plt.close(fig)
