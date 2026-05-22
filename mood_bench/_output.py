from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from rich.console import Console
from rich.table import Table
from rich.text import Text

_console = Console()
_verbose = False


def set_verbose(v: bool) -> None:
    """Enable or disable verbose (info-level) output."""
    global _verbose
    _verbose = v


def info(msg: str) -> None:
    """Informational message -- only printed when verbose is enabled."""
    if _verbose:
        _console.print(f"[dim]{msg}[/dim]")


def warn(msg: str) -> None:
    """Warning message -- always shown."""
    _console.print(f"[yellow]\\[warning][/yellow] {msg}")


@contextmanager
def status(msg: str) -> Generator[None, None, None]:
    with _console.status(msg):
        yield


def _fmt_pct(value: float) -> Text:
    if value != value:  # nan
        return Text("  —  ", style="dim")
    pct = value * 100
    if pct >= 80:
        style = "green"
    elif pct >= 50:
        style = "yellow"
    else:
        style = "red"
    return Text(f"{pct:6.1f}%", style=style)


def print_report_table(report: dict[str, Any], *, title: str | None = None) -> None:
    groups: dict[str, dict[str, Any]] = report.get("groups", {})
    if not groups:
        return

    sample_group = next(
        (v for k, v in groups.items() if k != "overall"), next(iter(groups.values()))
    )
    tpr_keys = sorted(k for k in sample_group if k.startswith("tpr@fpr"))

    table = Table(title=title, show_lines=False)
    table.add_column("Group", style="bold")
    table.add_column("N", justify="right")
    table.add_column("AUROC", justify="right")
    for key in tpr_keys:
        label = key.replace("tpr@fpr", "TPR@FPR")
        table.add_column(label, justify="right")

    ordered_keys = []
    if "id" in groups:
        ordered_keys.append("id")
    ordered_keys.extend(sorted(k for k in groups if k not in ("id", "overall")))
    if "overall" in groups:
        ordered_keys.append("overall")

    for key in ordered_keys:
        g = groups[key]
        is_overall = key == "overall"
        name = Text(key, style="bold" if is_overall else "")
        n_text = str(g.get("n", ""))
        auroc = _fmt_pct(g.get("auroc", float("nan")))
        tpr_cells = [_fmt_pct(g.get(tk, float("nan"))) for tk in tpr_keys]

        table.add_row(
            name,
            n_text,
            auroc,
            *tpr_cells,
            end_section=is_overall,
        )

    _console.print()
    _console.print(table)
    _console.print()
