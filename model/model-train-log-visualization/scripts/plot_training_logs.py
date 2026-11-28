#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_train_log_metrics_module():
    module_path = Path(__file__).resolve().parent / "train_log_metrics.py"
    if not module_path.exists():
        raise RuntimeError(f"missing module file: {module_path}")
    spec = importlib.util.spec_from_file_location("train_log_metrics", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_log_metrics"] = module
    spec.loader.exec_module(module)
    return module


def _normalize_metric(metric: str) -> str:
    alias_map = {
        "memory": "memory_gib",
        "mfu": "mfu_pct",
        "elapsed": "elapsed_time_per_step",
        "indexer": "indexer_loss",
        "indexer_loss": "indexer_loss",
    }
    return alias_map.get(metric, metric)


def _parse_metric_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [_normalize_metric(item.strip()) for item in raw.split(",") if item.strip()]


PR_IMAGE_MAX_BYTES = 50_000
PR_IMAGE_INITIAL_DPI = 150.0
PR_IMAGE_MIN_DPI = 1.0
PR_IMAGE_MAX_ATTEMPTS = 12


def _default_output_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == ".agents":
            return parent.parent / "outputs"

    return Path.cwd() / "outputs"


def _default_output_path(log_a: str, log_b: str | None, output_format: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_a = Path(log_a).stem
    if log_b:
        base_b = Path(log_b).stem
        filename = f"{base_a}_vs_{base_b}_{stamp}.{output_format}"
    else:
        filename = f"{base_a}_{stamp}.{output_format}"
    return _default_output_dir() / filename


def _default_pr_image_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_pr_under_50kb.png")


def _resolve_pr_image_output(
    output: Path,
    *,
    generate_pr_image: bool,
    pr_image_output: str | None,
) -> Path | None:
    if not generate_pr_image and not pr_image_output:
        return None
    pr_output = Path(pr_image_output) if pr_image_output else _default_pr_image_output_path(output)
    if pr_output.suffix.lower() != ".png":
        pr_output = pr_output.with_suffix(".png")
    return pr_output


def _render_png_bytes(figure, *, dpi: float) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compress_level": 9},
    )
    return buffer.getvalue()


def _save_pr_image_under_limit(
    figure,
    output: Path,
    *,
    max_bytes: int = PR_IMAGE_MAX_BYTES,
) -> int:
    """Save a PNG below max_bytes without forcing a fixed aspect ratio."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    dpi = PR_IMAGE_INITIAL_DPI
    png_data = b""
    for _attempt in range(PR_IMAGE_MAX_ATTEMPTS):
        try:
            png_data = _render_png_bytes(figure, dpi=dpi)
        except OSError as error:
            raise RuntimeError(f"failed to render PR image '{output}': {error}") from error

        if len(png_data) < max_bytes:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(png_data)
            except OSError as error:
                raise RuntimeError(f"failed to save PR image to '{output}': {error}") from error
            return len(png_data)

        scale = math.sqrt((max_bytes - 1) / len(png_data)) * 0.95
        next_dpi = max(PR_IMAGE_MIN_DPI, dpi * min(0.9, scale))
        if next_dpi == dpi:
            break
        dpi = next_dpi

    raise RuntimeError(
        f"failed to compress PR image below {max_bytes / 1000:.0f} KB; "
        f"smallest render was {len(png_data) / 1000:.1f} KB"
    )


def _require_matplotlib(no_show: bool):
    try:
        import matplotlib

        if no_show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for plotting. Please install it first: pip install matplotlib"
        ) from exc


def _apply_integer_xaxis(axis) -> None:
    """Force the x-axis (training step) to display integer ticks only.

    MaxNLocator(integer=True) prefers integer ticks but still falls back to
    fractional values when the data range is narrow (e.g. a single point). Pair
    it with a formatter that hides labels for any non-integer tick so the
    rendered step axis never shows decimals.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def _format_step(value: float, _pos: int) -> str:
        rounded = round(value)
        if abs(value - rounded) > 1e-6:
            return ""
        return f"{int(rounded)}"

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_step))


def _metric_label(metric_key: str) -> str:
    labels = {
        "loss": "loss",
        "grad_norm": "grad_norm",
        "memory_gib": "memory (GiB)",
        "memory_pct": "memory (%)",
        "tps": "tps",
        "tflops": "tflops",
        "mfu_pct": "mfu (%)",
        "elapsed_time_per_step": "elapsed_time_per_step (s)",
        "indexer_loss": "indexer loss",
        "loss_abs_error": "loss abs error",
        "loss_rel_error": "loss relative error",
        "grad_norm_abs_error": "grad_norm abs error",
        "grad_norm_rel_error": "grad_norm relative error",
    }
    return labels.get(metric_key, metric_key)


def _compute_error_stats(errors: list[float]) -> dict[str, float]:
    """Compute mean/mse/min/max over finite errors only.

    Non-finite values (NaN or Inf) mean "not comparable"; an all-invalid input
    returns NaN rather than 0.0 so missing data is never shown as a match.
    """
    valid = [error for error in errors if math.isfinite(error)]
    if not valid:
        return {
            "mean": float("nan"),
            "mse": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    mean_err = sum(valid) / len(valid)
    mse = sum(error * error for error in valid) / len(valid)
    min_err = min(valid)
    max_err = max(valid)

    return {
        "mean": mean_err,
        "mse": mse,
        "min": min_err,
        "max": max_err,
    }


def _count_non_finite(values: list[float]) -> int:
    return sum(1 for value in values if not math.isfinite(value))


def _extract_paired_metric_values(
    aligned,
    metric_key: str,
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Return common-step finite metric pairs, skipping missing/non-finite steps."""
    steps: list[int] = []
    values_a: list[float] = []
    values_b: list[float] = []
    skipped_steps: list[int] = []

    for step, record_a, record_b in zip(
        aligned.steps,
        aligned.records_a,
        aligned.records_b,
        strict=True,
    ):
        value_a = record_a.get(metric_key)
        value_b = record_b.get(metric_key)
        if value_a is None or value_b is None:
            skipped_steps.append(step)
            continue

        float_a = float(value_a)
        float_b = float(value_b)
        if not math.isfinite(float_a) or not math.isfinite(float_b):
            skipped_steps.append(step)
            continue

        steps.append(step)
        values_a.append(float_a)
        values_b.append(float_b)

    return steps, values_a, values_b, skipped_steps


def _plot_single_log(
    plt,
    module,
    *,
    records,
    metrics: list[str],
    title: str,
    output: Path,
    pr_image_output: Path | None,
    no_show: bool,
):
    """Plot single log with 2-column layout."""
    selected = ["loss", "grad_norm"] + [metric for metric in metrics if metric not in {"loss", "grad_norm"}]
    n_metrics = len(selected)

    # Calculate grid: 2 columns, enough rows
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=False)
    if n_metrics == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes_flat = list(axes[0])
    else:
        axes_flat = [
            axes[row][column]
            for row in range(n_rows)
            for column in range(n_cols)
        ]

    warnings: list[str] = []
    for idx, metric in enumerate(selected):
        axis = axes_flat[idx]
        steps, values = module.extract_metric_series(records, metric)
        if not steps:
            warnings.append(f"metric '{metric}' not found in log")
            axis.text(
                0.5,
                0.5,
                f"{metric} not available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylabel(_metric_label(metric))
            axis.grid(True, linestyle="--", alpha=0.4)
            axis.set_xticks([])
            continue
        axis.plot(steps, values, label=metric, linewidth=1.2)
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="best", fontsize=8)
        _apply_integer_xaxis(axis)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error

    if pr_image_output is not None:
        try:
            _save_pr_image_under_limit(figure, pr_image_output)
        except RuntimeError:
            plt.close(figure)
            raise

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def _plot_compare(
    plt,
    module,
    *,
    records_a,
    records_b,
    metrics: list[str],
    baseline: str,
    title: str,
    output: Path,
    pr_image_output: Path | None,
    no_show: bool,
    log_a_name: str = "a",
    log_b_name: str = "b",
):
    """Plot comparison with 2-column layout and error curves under each metric."""
    aligned = module.align_by_common_steps(records_a, records_b)
    warnings: list[str] = []
    if aligned.missing_in_a:
        warnings.append(f"steps only in log-b: {len(aligned.missing_in_a)}")
    if aligned.missing_in_b:
        warnings.append(f"steps only in log-a: {len(aligned.missing_in_b)}")
    if not aligned.steps:
        warnings.append("no common steps between log-a and log-b")
        return warnings

    # Required metrics with error tracking
    required_metrics = ["loss", "grad_norm"]
    optional_metrics = [m for m in metrics if m not in required_metrics]

    # Compute absolute errors and signed relative errors for loss and grad_norm.
    # Steps with a missing or non-finite value on either side are skipped so
    # they cannot be mistaken for a perfect match.
    loss_error_steps, losses_a, losses_b, skipped_loss_steps = _extract_paired_metric_values(
        aligned,
        "loss",
    )
    if skipped_loss_steps:
        warnings.append(
            f"loss missing or non-finite in either log for {len(skipped_loss_steps)} common steps; "
            "skipped those steps in loss error curves and statistics"
        )
    if loss_error_steps:
        loss_abs_err, loss_rel_err = module.compute_errors(losses_a, losses_b, baseline=baseline)
    else:
        loss_abs_err, loss_rel_err = [], []

    grad_error_steps, grad_norms_a, grad_norms_b, skipped_grad_steps = (
        _extract_paired_metric_values(aligned, "grad_norm")
    )
    if skipped_grad_steps:
        warnings.append(
            "grad_norm missing or non-finite in either log for "
            f"{len(skipped_grad_steps)} common steps; "
            "skipped those steps in grad_norm error curves and statistics"
        )
    if grad_error_steps:
        grad_abs_err, grad_rel_err = module.compute_errors(
            grad_norms_a,
            grad_norms_b,
            baseline=baseline,
        )
    else:
        grad_abs_err, grad_rel_err = [], []

    loss_undefined_rel = _count_non_finite(loss_rel_err)
    if loss_undefined_rel:
        warnings.append(
            f"loss relative error undefined for {loss_undefined_rel} common steps "
            "(baseline is 0 or non-finite); skipped in statistics"
        )
    grad_undefined_rel = _count_non_finite(grad_rel_err)
    if grad_undefined_rel:
        warnings.append(
            f"grad_norm relative error undefined for {grad_undefined_rel} common steps "
            "(baseline is 0 or non-finite); skipped in statistics"
        )

    # Compute error statistics
    loss_abs_stats = _compute_error_stats(loss_abs_err)
    loss_rel_stats = _compute_error_stats(loss_rel_err)
    grad_abs_stats = _compute_error_stats(grad_abs_err)
    grad_rel_stats = _compute_error_stats(grad_rel_err)

    # Build subplot list: each main metric followed by its error curves
    # Layout: 2 columns
    # Row 0: loss, grad_norm
    # Row 1: loss_abs_error, grad_norm_abs_error
    # Row 2: loss_rel_error, grad_norm_rel_error
    # Then optional metrics in subsequent rows

    n_optional_rows = (len(optional_metrics) + 1) // 2
    n_rows = 3 + n_optional_rows  # 3 rows for loss+grad_norm with their errors

    figure, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Row 0: loss and grad_norm
    # Loss subplot
    ax_loss = axes[0][0]
    loss_a_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_a]
    loss_b_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_b]
    ax_loss.plot(aligned.steps, loss_a_vals, label=log_a_name, linewidth=1.2)
    ax_loss.plot(aligned.steps, loss_b_vals, label=log_b_name, linewidth=1.2)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss)

    # Grad norm subplot
    ax_grad = axes[0][1]
    grad_a_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_a]
    grad_b_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_b]
    ax_grad.plot(aligned.steps, grad_a_vals, label=log_a_name, linewidth=1.2)
    ax_grad.plot(aligned.steps, grad_b_vals, label=log_b_name, linewidth=1.2)
    ax_grad.set_ylabel("grad_norm")
    ax_grad.grid(True, linestyle="--", alpha=0.4)
    ax_grad.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad)

    # Row 1: absolute errors (no threshold lines)
    # Loss abs error
    ax_loss_abs = axes[1][0]
    ax_loss_abs.plot(loss_error_steps, loss_abs_err, label="loss abs error", color="blue", linewidth=1.0)
    ax_loss_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_abs.set_ylabel("loss abs error")
    ax_loss_abs.grid(True, linestyle="--", alpha=0.4)
    ax_loss_abs.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss_abs)

    # Add statistics text below the plot (larger font)
    stats_text = "no valid paired loss values"
    if loss_abs_err:
        stats_text = (
            f"mean={loss_abs_stats['mean']:.5f}, mse={loss_abs_stats['mse']:.5f}, "
            f"min={loss_abs_stats['min']:.5f}, max={loss_abs_stats['max']:.5f}"
        )
    ax_loss_abs.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Grad norm abs error
    ax_grad_abs = axes[1][1]
    ax_grad_abs.plot(
        grad_error_steps,
        grad_abs_err,
        label="grad_norm abs error",
        color="blue",
        linewidth=1.0,
    )
    ax_grad_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_abs.set_ylabel("grad_norm abs error")
    ax_grad_abs.grid(True, linestyle="--", alpha=0.4)
    ax_grad_abs.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad_abs)

    # Add statistics text (larger font)
    stats_text = "no valid paired grad_norm values"
    if grad_abs_err:
        stats_text = (
            f"mean={grad_abs_stats['mean']:.5f}, mse={grad_abs_stats['mse']:.5f}, "
            f"min={grad_abs_stats['min']:.5f}, max={grad_abs_stats['max']:.5f}"
        )
    ax_grad_abs.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Row 2: relative errors (with threshold lines)
    # Loss rel error with threshold lines
    ax_loss_rel = axes[2][0]
    baseline_name = log_a_name if baseline == "a" else log_b_name
    ax_loss_rel.plot(
        loss_error_steps,
        loss_rel_err,
        label=f"loss relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_loss_rel.axhline(y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02")
    ax_loss_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_loss_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_rel.set_ylabel("loss relative error")
    ax_loss_rel.grid(True, linestyle="--", alpha=0.4)
    ax_loss_rel.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss_rel)

    # Add statistics text (larger font)
    stats_text = "no valid paired loss values"
    if loss_rel_err:
        stats_text = (
            f"mean={loss_rel_stats['mean']:.5f}, mse={loss_rel_stats['mse']:.5f}, "
            f"min={loss_rel_stats['min']:.5f}, max={loss_rel_stats['max']:.5f}"
        )
    ax_loss_rel.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Grad norm rel error with threshold lines
    ax_grad_rel = axes[2][1]
    ax_grad_rel.plot(
        grad_error_steps,
        grad_rel_err,
        label=f"grad_norm relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_grad_rel.axhline(y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02")
    ax_grad_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_grad_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_rel.set_ylabel("grad_norm relative error")
    ax_grad_rel.grid(True, linestyle="--", alpha=0.4)
    ax_grad_rel.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad_rel)

    # Add statistics text (larger font)
    stats_text = "no valid paired grad_norm values"
    if grad_rel_err:
        stats_text = (
            f"mean={grad_rel_stats['mean']:.5f}, mse={grad_rel_stats['mse']:.5f}, "
            f"min={grad_rel_stats['min']:.5f}, max={grad_rel_stats['max']:.5f}"
        )
    ax_grad_rel.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )

    # Remaining rows: optional metrics
    for idx, metric in enumerate(optional_metrics):
        row = 3 + idx // 2
        col = idx % 2
        axis = axes[row][col]

        values_a = []
        values_b = []
        for record_a, record_b in zip(aligned.records_a, aligned.records_b, strict=True):
            value_a = record_a.get(metric)
            value_b = record_b.get(metric)
            values_a.append(float(value_a) if value_a is not None else float("nan"))
            values_b.append(float(value_b) if value_b is not None else float("nan"))

        has_valid_a = any(not math.isnan(value) for value in values_a)
        has_valid_b = any(not math.isnan(value) for value in values_b)
        if not has_valid_a and not has_valid_b:
            warnings.append(f"metric '{metric}' not found in either log")
            axis.text(
                0.5,
                0.5,
                f"{metric} not available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylabel(_metric_label(metric))
            axis.grid(True, linestyle="--", alpha=0.4)
            axis.set_xticks([])
            continue

        axis.plot(aligned.steps, values_a, label=log_a_name, linewidth=1.2)
        axis.plot(aligned.steps, values_b, label=log_b_name, linewidth=1.2)
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend(loc="best", fontsize=8)
        _apply_integer_xaxis(axis)

    # Hide unused subplots in optional metrics section
    n_optional_plotted = len(optional_metrics)
    for idx in range(n_optional_plotted, n_optional_rows * 2):
        row = 3 + idx // 2
        col = idx % 2
        if row < n_rows:
            axes[row][col].axis("off")

    # Set xlabel for bottom row
    for col in range(2):
        axes[-1][col].set_xlabel("step")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
        if pr_image_output is not None:
            _save_pr_image_under_limit(figure, pr_image_output)
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error
    except RuntimeError:
        plt.close(figure)
        raise

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Plot training metrics from torchtitan-npu logs")
    parser.add_argument("--log-a", required=True, help="path to primary log")
    parser.add_argument("--log-b", default=None, help="path to comparison log")
    parser.add_argument(
        "--metrics",
        default="",
        help="optional metrics: memory_gib,memory_pct,tps,tflops,mfu_pct,elapsed_time_per_step,indexer_loss",
    )
    parser.add_argument("--output", default=None, help="output image path")
    parser.add_argument(
        "--generate-pr-image",
        action="store_true",
        help="also export an extra PNG below 50 KB for PR embedding",
    )
    parser.add_argument(
        "--pr-image-output",
        default=None,
        help="optional path for the PR image below 50 KB (.png)",
    )
    parser.add_argument("--title", default=None, help="figure title")
    parser.add_argument(
        "--baseline",
        choices=["a", "b"],
        default="a",
        help="baseline for relative error (default: a, i.e., log-a)",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png", help="output format")
    parser.add_argument("--no-show", action="store_true", help="disable interactive display")
    args = parser.parse_args()

    module = _load_train_log_metrics_module()

    try:
        records_a, warnings_a = module.read_training_metrics(args.log_a)
    except OSError as error:
        logger.error(f"[error] failed to read log-a '{args.log_a}': {error}")
        return 1
    if not records_a:
        logger.error(f"[error] no valid training metrics found in log-a: {args.log_a}")
        return 1

    records_b = None
    warnings_b: list[str] = []
    if args.log_b:
        try:
            records_b, warnings_b = module.read_training_metrics(args.log_b)
        except OSError as error:
            logger.error(f"[error] failed to read log-b '{args.log_b}': {error}")
            return 1
        if not records_b:
            logger.error(f"[error] no valid training metrics found in log-b: {args.log_b}")
            return 1

    output = Path(args.output) if args.output else _default_output_path(args.log_a, args.log_b, args.format)
    if output.suffix.lower() != f".{args.format}":
        output = output.with_suffix(f".{args.format}")

    pr_image_output = _resolve_pr_image_output(
        output,
        generate_pr_image=args.generate_pr_image,
        pr_image_output=args.pr_image_output,
    )

    metrics = _parse_metric_list(args.metrics)

    # Get log file names for legend
    log_a_name = Path(args.log_a).stem
    log_b_name = Path(args.log_b).stem if args.log_b else None

    title = args.title or (
        f"Training Metrics: {log_a_name} vs {log_b_name}" if args.log_b else f"Training Metrics: {log_a_name}"
    )

    try:
        plt = _require_matplotlib(args.no_show)
    except RuntimeError as error:
        logger.error(f"[error] {error}")
        return 1

    warnings = [*warnings_a, *warnings_b]
    try:
        if records_b is None:
            warnings.extend(
                _plot_single_log(
                    plt,
                    module,
                    records=records_a,
                    metrics=metrics,
                    title=title,
                    output=output,
                    pr_image_output=pr_image_output,
                    no_show=args.no_show,
                )
            )
        else:
            warnings.extend(
                _plot_compare(
                    plt,
                    module,
                    records_a=records_a,
                    records_b=records_b,
                    metrics=metrics,
                    baseline=args.baseline,
                    title=title,
                    output=output,
                    pr_image_output=pr_image_output,
                    no_show=args.no_show,
                    log_a_name=log_a_name,
                    log_b_name=log_b_name or "b",
                )
            )
    except RuntimeError as error:
        logger.error(f"[error] {error}")
        return 1

    if records_b is not None and "no common steps between log-a and log-b" in warnings:
        logger.error("[error] no common steps between log-a and log-b")
        return 1

    logger.info(f"[ok] plot saved to: {output}")
    if pr_image_output is not None:
        logger.info(f"[ok] pr image saved to: {pr_image_output}")
    summary_a = module.summarize_records(records_a)
    logger.info(
        f"[summary-a] steps={summary_a['num_steps']} "
        f"range=[{summary_a['first_step']},{summary_a['last_step']}] "
        f"loss={summary_a['first_loss']:.5f}->{summary_a['last_loss']:.5f} "
        f"grad_norm={summary_a['first_grad_norm']:.4f}->{summary_a['last_grad_norm']:.4f}"
    )
    if records_b is not None:
        summary_b = module.summarize_records(records_b)
        logger.info(
            f"[summary-b] steps={summary_b['num_steps']} "
            f"range=[{summary_b['first_step']},{summary_b['last_step']}] "
            f"loss={summary_b['first_loss']:.5f}->{summary_b['last_loss']:.5f} "
            f"grad_norm={summary_b['first_grad_norm']:.4f}->{summary_b['last_grad_norm']:.4f}"
        )

    for warning in warnings:
        logger.warning(f"[warning] {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
