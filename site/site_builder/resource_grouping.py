"""Precompute resource means across configurations without pooling requests."""

from typing import Any

from .table_grouping import _stats


def resource_views(resources: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only the original eligible resource rows, retaining source precision."""
    by_series = {row["series"]: row for row in rows}
    originals = [
        {
            **resource,
            **{
                field: by_series[resource["series"]][field]
                for field in ("model", "input", "prompt")
            },
        }
        for resource in resources
    ]
    views = {}
    for fields in ((), ("input",), ("prompt",), ("input", "prompt")):
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in originals:
            key = (
                (row["series"],)
                if not fields
                else (row["model"],)
                + tuple(row[field] for field in ("input", "prompt") if field not in fields)
            )
            groups.setdefault(key, []).append(row)
        summaries = []
        for index, members in enumerate(groups.values()):
            summary = {
                **members[0],
                "series": f"group-{index}" if fields else members[0]["series"],
                "count": len(members),
                "inputs": list(dict.fromkeys(row["input"] for row in members)),
                "prompts": list(dict.fromkeys(row["prompt"] for row in members)),
                "local_api_free": all(row["local_api_free"] for row in members),
            }
            for metric, prefix in (("cost_per_advert", "cost"), ("latency_per_advert", "latency")):
                stats = _stats([row[metric] for row in members])
                summary[metric] = stats.mean
                summary[prefix + "_sd"] = stats.sd
                summary[prefix + "_availability"] = stats.reason
            summaries.append(summary)
        views[",".join(fields) or "none"] = {"fields": list(fields), "rows": summaries}
    return views
