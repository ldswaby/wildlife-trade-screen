"""Terminal-only run progress, separate from scores and experiment records."""

import shutil
import sys
import time
from typing import TextIO

from .types import ScoreResult


def _clock(seconds: float | None) -> str:
    """Format a rough duration without implying an ETA before the first result."""
    if seconds is None:
        return "--:--"
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return (
        f"{hours}:{minutes:02d}:{seconds_part:02d}"
        if hours
        else f"{minutes:02d}:{seconds_part:02d}"
    )


class TerminalProgress:
    """Show saved results on stderr without influencing scoring if display fails."""

    def __init__(self, arm: str, stream: TextIO | None = None) -> None:
        """Use an interactive terminal only; non-TTY callers retain quiet output."""
        self.arm = arm
        self.stream = stream if stream is not None else sys.stderr
        try:
            self.enabled = self.stream.isatty()
        except (OSError, ValueError):
            self.enabled = False
        self.started = False
        self.cached = 0
        self.pending = 0
        self.saved = 0
        self.errors = 0
        self.start_time = 0.0
        self.last_draw = 0.0

    def begin(self, cached: int, pending: int, cached_errors: int) -> None:
        """Start after cache inspection; cached rows do not inflate the ETA rate."""
        if not self.enabled:
            return
        self.cached = cached
        self.pending = pending
        self.errors = cached_errors
        self.start_time = time.monotonic()
        self.started = True
        self._write(f"{self.arm}\n")
        self._draw(force=True)

    def saved_result(self, result: ScoreResult) -> None:
        """Advance only after the runner has durably saved this result."""
        if not self.started or not self.enabled:
            return
        self.saved += 1
        self.errors += result.error is not None
        self._draw(force=self.saved == self.pending)

    def finish(self, status: str) -> None:
        """End the terminal line with the actual run state, including aborts."""
        if not self.started or not self.enabled:
            return
        self._draw(force=True)
        state = "stopped" if status == "aborted" else status
        self._write(
            f"\n{self.arm}: {state} "
            f"({self.cached + self.saved}/{self.cached + self.pending} processed, "
            f"{self.errors} errors)\n"
        )
        self.started = False

    def _draw(self, *, force: bool) -> None:
        """Throttle redraws; base remaining time on newly saved rows only."""
        moment = time.monotonic()
        if not force and moment - self.last_draw < 0.1:
            return
        self.last_draw = moment
        elapsed = moment - self.start_time
        remaining = self.pending - self.saved
        eta = elapsed * remaining / self.saved if self.saved else None
        fraction = self.saved / self.pending if self.pending else 1.0
        width = shutil.get_terminal_size((120, 20)).columns
        bar_width = 18 if width >= 100 else 8
        filled = round(bar_width * fraction)
        bar = "█" * filled + "░" * (bar_width - filled)
        if width >= 100:
            line = (
                f"[{bar}] {self.cached + self.saved}/{self.cached + self.pending} processed "
                f"| {remaining} pending | {self.cached} cached | {self.errors} errors "
                f"| {_clock(elapsed)} elapsed | ETA {_clock(eta)}"
            )
        else:
            line = (
                f"[{bar}] {self.cached + self.saved}/{self.cached + self.pending} "
                f"| {remaining} left | {self.cached} cache | {self.errors} err "
                f"| {_clock(elapsed)} | ETA {_clock(eta)}"
            )
        self._write("\r" + line[: max(1, width - 1)] + "\x1b[K")

    def _write(self, value: str) -> None:
        """Disable display after a terminal failure without affecting model calls."""
        try:
            self.stream.write(value)
            self.stream.flush()
        except (OSError, UnicodeError, ValueError):
            self.enabled = False
