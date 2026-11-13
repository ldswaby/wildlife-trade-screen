"""Check dataset summaries against independently calculated invented counts."""

from pathlib import Path

import pytest
from site_builder.content import class_distribution, data_summary_source

from wildlife_screening.data import load_data


@pytest.mark.parametrize(
    ("total", "positives", "expected"),
    [
        (10, 3, ("7", "3", "70.0%", "30.0%")),
        (10, 5, ("5", "5", "50.0%", "50.0%")),
        (10, 0, ("10", "0", "100.0%", "0.0%")),
        (10, 10, ("0", "10", "0.0%", "100.0%")),
        (0, 0, ("0", "0", "N/A", "N/A")),
    ],
)
def test_distribution_examples(total: int, positives: int, expected: tuple[str, ...]) -> None:
    page = class_distribution(total, positives)
    neg, pos, neg_pct, pos_pct = expected
    assert f'<th scope="row">Absolute count</th><td>{neg}</td><td>{pos}</td>' in page
    assert f'<th scope="row">Percentage (%)</th><td>{neg_pct}</td><td>{pos_pct}</td>' in page
    assert "Imbalance ratio" not in page
    assert page.count('scope="row"') == 2
    assert page.count('scope="col"') == 4
    assert "<caption>Evaluation class distribution</caption>" in page
    assert 'tabindex="0"' in page
    assert "normalised evaluation target" in page
    assert f"<td>{'100%' if total else 'N/A'}</td>" in page


def test_distribution_uses_normalised_positive_polarity(tmp_path: Path) -> None:
    source = tmp_path / "invented.csv"
    source.write_text("id,title,label\na,Invented alpha,0\nb,Invented beta,0\nc,Invented gamma,1\n")
    data = load_data(source, positive_label="0")
    assert data.labels == [1, 1, 0]
    page = class_distribution(len(data.labels), sum(data.labels))
    assert '<th scope="row">Absolute count</th><td>1</td><td>2</td><td>3</td>' in page
    assert "33.3%" in page and "66.7%" in page


@pytest.mark.parametrize("count", [0, 2])
def test_summary_marker_errors_are_actionable(tmp_path: Path, count: int) -> None:
    source = tmp_path / "invented.md"
    source.write_text("- **Labels/policy:** Invented.\n\n" + "{{ class_distribution }}\n" * count)
    with pytest.raises(ValueError, match="invented.md: expected exactly one") as error:
        data_summary_source(tmp_path, source.name)
    assert "between Labels/policy and Caveats" in str(error.value)


def test_summary_marker_preserves_authored_text(tmp_path: Path) -> None:
    source = "- **Labels/policy:** Invented.\n\n{{ class_distribution }}\n\n- **Caveats:** Draft.\n"
    (tmp_path / "invented.md").write_text(source)
    assert data_summary_source(tmp_path, "invented.md") == source


@pytest.mark.parametrize(("total", "positives"), [(2, 3), (2, -1), (-1, 0)])
def test_invalid_distribution_counts_are_rejected(total: int, positives: int) -> None:
    with pytest.raises(ValueError, match="Evaluation counts"):
        class_distribution(total, positives)
