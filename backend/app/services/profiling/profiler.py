import contextlib
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def stage_timer(results: dict[str, list[float]], name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        results[name].append(elapsed)


class FrameProfiler:
    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._stages: dict[str, list[float]] = defaultdict(list)
        self._frame_count = 0

    def timer(self, name: str) -> Iterator[None]:
        return stage_timer(self._stages, name)

    def log_frame(self) -> None:
        self._frame_count += 1

    @property
    def total_frames(self) -> int:
        return self._frame_count

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for stage, times in self._stages.items():
            recent = times[-self._window :]
            result[stage] = {
                "count": len(recent),
                "mean_ms": round(statistics.mean(recent), 2) if recent else 0.0,
                "median_ms": round(statistics.median(recent), 2) if recent else 0.0,
                "min_ms": round(min(recent), 2) if recent else 0.0,
                "max_ms": round(max(recent), 2) if recent else 0.0,
                "p95_ms": round(sorted(recent)[int(len(recent) * 0.95) - 1] if recent else 0.0, 2),
            }
        result["total_frames"] = self._frame_count
        return result

    def reset(self) -> None:
        self._stages.clear()
        self._frame_count = 0

    def generate_benchmark_report(self) -> str:
        s = self.summary()
        lines = [
            "=" * 60,
            "Proctor Benchmark Report",
            "=" * 60,
            f"Frames processed: {s['total_frames']}",
            "",
            "--- Per-Stage Latency (ms) ---",
            f"{'Stage':<30} {'Count':>6} {'Mean':>8} {'Median':>8}"
            f" {'Min':>8} {'Max':>8} {'P95':>8}",
            "-" * 76,
        ]
        for stage, stats in sorted(s.items()):
            if stage == "total_frames":
                continue
            lines.append(
                f"{stage:<30} {stats['count']:>6} {stats['mean_ms']:>8.1f}"
                f" {stats['median_ms']:>8.1f} {stats['min_ms']:>8.1f}"
                f" {stats['max_ms']:>8.1f} {stats['p95_ms']:>8.1f}"
            )
        total_time = sum(stats["mean_ms"] for stage, stats in s.items() if stage != "total_frames")
        lines.append("")
        lines.append(f"Estimated total pipeline latency: {total_time:.1f} ms")
        if total_time > 0:
            lines.append(f"Estimated max throughput: {1000 / total_time:.1f} FPS")
        lines.append("=" * 60)
        return "\n".join(lines)
