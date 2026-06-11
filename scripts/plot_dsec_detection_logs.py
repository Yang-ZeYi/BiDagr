#!/usr/bin/env python3
"""
Generate paper-ready SVG plots from DSEC detection logs.

Inputs (default):
  logs/dsec/detection/<run>/{train_metrics.jsonl,eval_metrics.jsonl}

Outputs (default):
  outputs/paper_plots/dsec_detection/
    - summary_metrics.csv
    - comparisons/*.svg
    - per_run/<run>/*.svg

Notes:
  - Designed to work in minimal Python environments (no matplotlib/pandas).
  - Uses SVG line styles + markers + grayscale to remain distinguishable in B/W prints.
  - "combine" curves are drawn thicker.
  - Generates extra zoomed plots for dense regions (tail window).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return float(v)
        return float(v)
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
    s = re.sub(r"[^a-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _human_metric_label(key: str) -> str:
    mapping = {
        "validation/metric/mAP": "mAP",
        "validation/metric/AP": "AP",
        "validation/metric/AP_50": "AP@0.50",
        "validation/metric/AP_75": "AP@0.75",
        "validation/metric/AP_90": "AP@0.90",
        "validation/metric/F1": "F1",
        "validation/metric/precision": "Precision",
        "validation/metric/recall": "Recall",
        "validation/metric/AR": "AR",
        "validation/metric/AP_S": "AP (S)",
        "validation/metric/AP_M": "AP (M)",
        "validation/metric/AP_L": "AP (L)",
        "validation/metric/AP_class/car": "AP (car)",
        "validation/metric/AP_class/pedestrian": "AP (pedestrian)",
        "training/loss": "Training Loss",
        "training/lr": "Learning Rate",
        "training/loss/iou_loss": "IoU Loss",
        "training/loss/conf_loss": "Conf Loss",
        "training/loss/cls_loss": "Cls Loss",
        "training/loss/l1_loss": "L1 Loss",
        "training/feedback/strength": "Feedback strength",
        "training/feedback/active_ratio": "Feedback active ratio",
        "training/feedback/active_cells": "Feedback active cells",
        "training/feedback/num_calls": "Feedback num calls",
        "training/feedback/num_nodes": "Feedback num nodes",
        "training/feedback/current_norm": "Feedback current norm",
        "training/feedback/avg_norm": "Feedback avg norm",
        "training/gate_scale1/avg_value": "Gate s1 avg",
        "training/gate_scale1/gate_min": "Gate s1 min",
        "training/gate_scale1/gate_max": "Gate s1 max",
        "training/gate_scale1/gate_std": "Gate s1 std",
        "training/gate_scale1/strength": "Gate s1 strength",
        "training/gate_scale1/num_nodes": "Gate s1 nodes",
        "training/gate_scale2/avg_value": "Gate s2 avg",
        "training/gate_scale2/strength": "Gate s2 strength",
        "training/gate_scale2/num_nodes": "Gate s2 nodes",
    }
    return mapping.get(key, key)


def _nice_number(x: float, round_: bool) -> float:
    if x == 0:
        return 0.0
    exp = math.floor(math.log10(abs(x)))
    f = abs(x) / (10**exp)
    if round_:
        if f < 1.5:
            nf = 1.0
        elif f < 3:
            nf = 2.0
        elif f < 7:
            nf = 5.0
        else:
            nf = 10.0
    else:
        if f <= 1:
            nf = 1.0
        elif f <= 2:
            nf = 2.0
        elif f <= 5:
            nf = 5.0
        else:
            nf = 10.0
    return math.copysign(nf * (10**exp), x)


def _nice_ticks(min_v: float, max_v: float, target: int = 6) -> List[float]:
    if not math.isfinite(min_v) or not math.isfinite(max_v):
        return []
    if min_v == max_v:
        if min_v == 0:
            return [0, 1]
        d = abs(min_v) * 0.1
        return [min_v - d, min_v, min_v + d]

    rng = _nice_number(max_v - min_v, round_=False)
    step = _nice_number(rng / max(1, (target - 1)), round_=True)
    graph_min = math.floor(min_v / step) * step
    graph_max = math.ceil(max_v / step) * step
    ticks = []
    v = graph_min
    for _ in range(1000):
        ticks.append(v)
        v += step
        if v > graph_max + step * 0.5:
            break
    return ticks


def _svg_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if av >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _downsample_xy(x: Sequence[float], y: Sequence[float], max_points: int) -> Tuple[List[float], List[float]]:
    n = min(len(x), len(y))
    if n <= max_points:
        return list(x[:n]), list(y[:n])
    stride = max(1, n // max_points)
    xs = [x[i] for i in range(0, n, stride)]
    ys = [y[i] for i in range(0, n, stride)]
    if xs[-1] != x[n - 1]:
        xs.append(x[n - 1])
        ys.append(y[n - 1])
    return xs, ys


def _pad_limits(ymin: float, ymax: float, pad_frac: float = 0.08, min_pad: float = 0.0) -> Tuple[float, float]:
    if not math.isfinite(ymin) or not math.isfinite(ymax):
        return ymin, ymax
    if ymin == ymax:
        pad = max(min_pad, abs(ymin) * 0.1 if ymin != 0 else 0.1)
        return ymin - pad, ymax + pad
    pad = max(min_pad, (ymax - ymin) * pad_frac)
    return ymin - pad, ymax + pad


@dataclass(frozen=True)
class SeriesStyle:
    stroke_gray: float  # 0..1
    stroke_width: float
    dasharray: Optional[str]
    marker: str  # circle/square/triangle/diamond/x


def _variant_style(variant: str) -> SeriesStyle:
    v = variant.lower()
    # Grayscale: 0=black, 1=white.
    if v == "combine":
        return SeriesStyle(stroke_gray=0.05, stroke_width=3.0, dasharray=None, marker="diamond")
    if v == "dsec":
        return SeriesStyle(stroke_gray=0.20, stroke_width=1.9, dasharray=None, marker="circle")
    if v == "feedback":
        return SeriesStyle(stroke_gray=0.45, stroke_width=1.9, dasharray="7,4", marker="square")
    if v == "gating":
        return SeriesStyle(stroke_gray=0.70, stroke_width=2.1, dasharray="2,4", marker="triangle")
    # fallback
    return SeriesStyle(stroke_gray=0.35, stroke_width=1.8, dasharray="5,3", marker="x")


def _marker_svg(marker: str, cx: float, cy: float, size: float, stroke: str, fill: str, stroke_w: float) -> str:
    if marker == "circle":
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" stroke="{stroke}" stroke-width="{stroke_w:.2f}" fill="{fill}"/>'
    if marker == "square":
        s = size * 1.6
        return f'<rect x="{(cx - s/2):.2f}" y="{(cy - s/2):.2f}" width="{s:.2f}" height="{s:.2f}" stroke="{stroke}" stroke-width="{stroke_w:.2f}" fill="{fill}"/>'
    if marker == "triangle":
        s = size * 2.0
        p1 = (cx, cy - s * 0.65)
        p2 = (cx - s * 0.7, cy + s * 0.55)
        p3 = (cx + s * 0.7, cy + s * 0.55)
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (p1, p2, p3))
        return f'<polygon points="{pts}" stroke="{stroke}" stroke-width="{stroke_w:.2f}" fill="{fill}"/>'
    if marker == "diamond":
        s = size * 2.0
        pts = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in ((cx, cy - s * 0.75), (cx - s * 0.75, cy), (cx, cy + s * 0.75), (cx + s * 0.75, cy))
        )
        return f'<polygon points="{pts}" stroke="{stroke}" stroke-width="{stroke_w:.2f}" fill="{fill}"/>'
    if marker == "x":
        s = size * 1.7
        return (
            f'<path d="M {cx - s:.2f} {cy - s:.2f} L {cx + s:.2f} {cy + s:.2f} M {cx - s:.2f} {cy + s:.2f} L {cx + s:.2f} {cy - s:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_w:.2f}" fill="none" stroke-linecap="round"/>'
        )
    return ""


def _line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: List[Tuple[str, Sequence[float], Sequence[float], SeriesStyle]],
    out_path: Path,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    clamp_01: bool = False,
    max_points_per_series: int = 2500,
) -> None:
    width = 1200
    height = 700
    margin_left = 95
    margin_right = 320  # legend
    margin_top = 70
    margin_bottom = 85

    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_x: List[float] = []
    all_y: List[float] = []
    cleaned: List[Tuple[str, List[float], List[float], SeriesStyle]] = []
    for label, xs, ys, style in series:
        xv = []
        yv = []
        for x, y in zip(xs, ys):
            xf = _safe_float(x)
            yf = _safe_float(y)
            if xf is None or yf is None:
                continue
            if not math.isfinite(xf) or not math.isfinite(yf):
                continue
            xv.append(xf)
            yv.append(yf)
        if not xv:
            continue
        if max_points_per_series:
            xv, yv = _downsample_xy(xv, yv, max_points_per_series)
        cleaned.append((label, xv, yv, style))
        all_x.extend(xv)
        all_y.extend(yv)

    if not cleaned:
        return

    xmin = min(all_x) if xlim is None else xlim[0]
    xmax = max(all_x) if xlim is None else xlim[1]
    ymin = min(all_y) if ylim is None else ylim[0]
    ymax = max(all_y) if ylim is None else ylim[1]

    ymin, ymax = _pad_limits(ymin, ymax, pad_frac=0.10, min_pad=0.0)
    if clamp_01:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)

    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0

    def x_to_px(x: float) -> float:
        return plot_x0 + (x - xmin) / (xmax - xmin) * plot_w

    def y_to_px(y: float) -> float:
        return plot_y0 + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    # Ticks
    xticks = _nice_ticks(xmin, xmax, target=7)
    yticks = _nice_ticks(ymin, ymax, target=7)

    # SVG header + styles
    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg_parts.append("<defs>")
    svg_parts.append(
        "<style>"
        "text{font-family:Helvetica,Arial,sans-serif; fill:#111;}"
        ".title{font-size:20px;font-weight:600;}"
        ".axislabel{font-size:16px;}"
        ".tick{font-size:13px;}"
        ".grid{stroke:#ddd;stroke-width:1;}"
        ".axis{stroke:#111;stroke-width:1.5;}"
        "</style>"
    )
    svg_parts.append("</defs>")

    # Title
    svg_parts.append(f'<text x="{margin_left}" y="35" class="title">{_svg_escape(title)}</text>')

    # Grid + ticks
    for t in xticks:
        if t < xmin - 1e-9 or t > xmax + 1e-9:
            continue
        xpx = x_to_px(t)
        svg_parts.append(f'<line x1="{xpx:.2f}" y1="{plot_y0:.2f}" x2="{xpx:.2f}" y2="{(plot_y0+plot_h):.2f}" class="grid"/>')
        svg_parts.append(f'<line x1="{xpx:.2f}" y1="{(plot_y0+plot_h):.2f}" x2="{xpx:.2f}" y2="{(plot_y0+plot_h+6):.2f}" class="axis"/>')
        svg_parts.append(f'<text x="{xpx:.2f}" y="{(plot_y0+plot_h+24):.2f}" class="tick" text-anchor="middle">{_svg_escape(_fmt_tick(t))}</text>')

    for t in yticks:
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue
        ypx = y_to_px(t)
        svg_parts.append(f'<line x1="{plot_x0:.2f}" y1="{ypx:.2f}" x2="{(plot_x0+plot_w):.2f}" y2="{ypx:.2f}" class="grid"/>')
        svg_parts.append(f'<line x1="{(plot_x0-6):.2f}" y1="{ypx:.2f}" x2="{plot_x0:.2f}" y2="{ypx:.2f}" class="axis"/>')
        svg_parts.append(
            f'<text x="{(plot_x0-10):.2f}" y="{(ypx+4):.2f}" class="tick" text-anchor="end">{_svg_escape(_fmt_tick(t))}</text>'
        )

    # Axes box
    svg_parts.append(f'<rect x="{plot_x0:.2f}" y="{plot_y0:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" class="axis"/>')

    # Axis labels
    svg_parts.append(
        f'<text x="{(plot_x0+plot_w/2):.2f}" y="{(height-25):.2f}" class="axislabel" text-anchor="middle">{_svg_escape(x_label)}</text>'
    )
    # rotated y label
    svg_parts.append(
        f'<text x="28" y="{(plot_y0+plot_h/2):.2f}" class="axislabel" text-anchor="middle" transform="rotate(-90 28 {(plot_y0+plot_h/2):.2f})">{_svg_escape(y_label)}</text>'
    )

    # Series lines
    for label, xs, ys, style in cleaned:
        color = f"rgb({int(style.stroke_gray*255)},{int(style.stroke_gray*255)},{int(style.stroke_gray*255)})"
        dash = f' stroke-dasharray="{style.dasharray}"' if style.dasharray else ""
        points = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in zip(xs, ys))
        svg_parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{style.stroke_width:.2f}"{dash} stroke-linejoin="round" stroke-linecap="round"/>'
        )
        # markers
        mark_every = max(1, len(xs) // 18)
        for i in range(0, len(xs), mark_every):
            mx = x_to_px(xs[i])
            my = y_to_px(ys[i])
            svg_parts.append(_marker_svg(style.marker, mx, my, size=3.2, stroke=color, fill="white", stroke_w=max(1.0, style.stroke_width * 0.6)))

    # Legend (right side)
    legend_x0 = plot_x0 + plot_w + 25
    legend_y0 = plot_y0 + 20
    svg_parts.append(f'<text x="{legend_x0:.2f}" y="{(legend_y0-8):.2f}" class="tick" font-weight="600">Legend</text>')
    y_cursor = legend_y0 + 12
    for label, _, __, style in cleaned:
        color = f"rgb({int(style.stroke_gray*255)},{int(style.stroke_gray*255)},{int(style.stroke_gray*255)})"
        dash = f' stroke-dasharray="{style.dasharray}"' if style.dasharray else ""
        x1 = legend_x0
        x2 = legend_x0 + 55
        y = y_cursor
        svg_parts.append(
            f'<line x1="{x1:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="{style.stroke_width:.2f}"{dash} stroke-linecap="round"/>'
        )
        svg_parts.append(_marker_svg(style.marker, legend_x0 + 27.5, y, size=3.2, stroke=color, fill="white", stroke_w=max(1.0, style.stroke_width * 0.6)))
        svg_parts.append(f'<text x="{(legend_x0+70):.2f}" y="{(y+4):.2f}" class="tick">{_svg_escape(label)}</text>')
        y_cursor += 26

    svg_parts.append("</svg>")
    _ensure_dir(out_path.parent)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _bar_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    categories: List[str],
    values: List[float],
    styles: List[SeriesStyle],
    out_path: Path,
    ylim: Optional[Tuple[float, float]] = None,
    clamp_01: bool = False,
) -> None:
    width = 1200
    height = 700
    margin_left = 105
    margin_right = 60
    margin_top = 70
    margin_bottom = 165

    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    ys = [v for v in values if v is not None and math.isfinite(v)]
    if not ys:
        return

    ymin = 0.0 if ylim is None else ylim[0]
    ymax = max(ys) if ylim is None else ylim[1]
    ymin, ymax = _pad_limits(ymin, ymax, pad_frac=0.10, min_pad=0.0)
    if clamp_01:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)

    if ymin == ymax:
        ymax = ymin + 1.0

    def y_to_px(y: float) -> float:
        return plot_y0 + plot_h - (y - ymin) / (ymax - ymin) * plot_h

    yticks = _nice_ticks(ymin, ymax, target=7)

    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg_parts.append("<defs>")
    svg_parts.append(
        "<style>"
        "text{font-family:Helvetica,Arial,sans-serif; fill:#111;}"
        ".title{font-size:20px;font-weight:600;}"
        ".axislabel{font-size:16px;}"
        ".tick{font-size:13px;}"
        ".grid{stroke:#ddd;stroke-width:1;}"
        ".axis{stroke:#111;stroke-width:1.5;}"
        "</style>"
    )
    # Simple hatch patterns
    svg_parts.append(
        """
        <pattern id="hatch_diag" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="8" stroke="#222" stroke-width="1"/>
        </pattern>
        <pattern id="hatch_cross" patternUnits="userSpaceOnUse" width="8" height="8">
          <line x1="0" y1="0" x2="8" y2="0" stroke="#222" stroke-width="1"/>
          <line x1="0" y1="0" x2="0" y2="8" stroke="#222" stroke-width="1"/>
        </pattern>
        <pattern id="hatch_dot" patternUnits="userSpaceOnUse" width="10" height="10">
          <circle cx="3" cy="3" r="1.2" fill="#222"/>
        </pattern>
        """
    )
    svg_parts.append("</defs>")

    svg_parts.append(f'<text x="{margin_left}" y="35" class="title">{_svg_escape(title)}</text>')

    # Grid + y ticks
    for t in yticks:
        if t < ymin - 1e-9 or t > ymax + 1e-9:
            continue
        ypx = y_to_px(t)
        svg_parts.append(f'<line x1="{plot_x0:.2f}" y1="{ypx:.2f}" x2="{(plot_x0+plot_w):.2f}" y2="{ypx:.2f}" class="grid"/>')
        svg_parts.append(f'<line x1="{(plot_x0-6):.2f}" y1="{ypx:.2f}" x2="{plot_x0:.2f}" y2="{ypx:.2f}" class="axis"/>')
        svg_parts.append(
            f'<text x="{(plot_x0-10):.2f}" y="{(ypx+4):.2f}" class="tick" text-anchor="end">{_svg_escape(_fmt_tick(t))}</text>'
        )

    svg_parts.append(f'<rect x="{plot_x0:.2f}" y="{plot_y0:.2f}" width="{plot_w:.2f}" height="{plot_h:.2f}" fill="none" class="axis"/>')

    svg_parts.append(
        f'<text x="{(plot_x0+plot_w/2):.2f}" y="{(height-25):.2f}" class="axislabel" text-anchor="middle">{_svg_escape(x_label)}</text>'
    )
    svg_parts.append(
        f'<text x="30" y="{(plot_y0+plot_h/2):.2f}" class="axislabel" text-anchor="middle" transform="rotate(-90 30 {(plot_y0+plot_h/2):.2f})">{_svg_escape(y_label)}</text>'
    )

    n = len(categories)
    if n == 0:
        return
    gap = 18.0
    bar_w = (plot_w - gap * (n + 1)) / n
    bar_w = max(20.0, min(bar_w, 85.0))
    total_w = n * bar_w + (n + 1) * gap
    start_x = plot_x0 + (plot_w - total_w) / 2

    # pattern assignment per style/variant order
    def fill_for(i: int) -> str:
        # rotate patterns to keep B/W distinguishable
        if i % 4 == 0:
            return "url(#hatch_diag)"
        if i % 4 == 1:
            return "url(#hatch_cross)"
        if i % 4 == 2:
            return "url(#hatch_dot)"
        return "#d0d0d0"

    for i, (cat, val, style) in enumerate(zip(categories, values, styles)):
        if val is None or not math.isfinite(val):
            continue
        x = start_x + gap + i * (bar_w + gap)
        y = y_to_px(val)
        h = plot_y0 + plot_h - y
        stroke = "#111"
        fill = fill_for(i)
        # emphasize combine by thicker border
        sw = 2.6 if "combine" in cat.lower() else 1.6
        svg_parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"/>')
        svg_parts.append(f'<text x="{(x+bar_w/2):.2f}" y="{(y-8):.2f}" class="tick" text-anchor="middle">{_svg_escape(_fmt_tick(val))}</text>')

        # x tick label rotated for long names
        tx = x + bar_w / 2
        ty = plot_y0 + plot_h + 40
        svg_parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" class="tick" text-anchor="end" transform="rotate(-35 {tx:.2f} {ty:.2f})">{_svg_escape(cat)}</text>'
        )

    svg_parts.append("</svg>")
    _ensure_dir(out_path.parent)
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


@dataclass
class RunLogs:
    name: str
    backbone: str
    variant: str
    train_path: Path
    eval_path: Path
    train_rows: List[Dict[str, Any]]
    eval_rows: List[Dict[str, Any]]


def _parse_run_name(name: str) -> Tuple[str, str]:
    parts = name.split("-")
    if len(parts) < 2:
        return name, "unknown"
    backbone = parts[0]
    variant = "-".join(parts[1:])
    return backbone, variant


def _load_run(run_dir: Path) -> Optional[RunLogs]:
    train_path = run_dir / "train_metrics.jsonl"
    eval_path = run_dir / "eval_metrics.jsonl"
    if not train_path.exists() or not eval_path.exists():
        return None
    name = run_dir.name
    backbone, variant = _parse_run_name(name)
    train_rows = list(_iter_jsonl(train_path))
    eval_rows = list(_iter_jsonl(eval_path))
    return RunLogs(
        name=name,
        backbone=backbone,
        variant=variant,
        train_path=train_path,
        eval_path=eval_path,
        train_rows=train_rows,
        eval_rows=eval_rows,
    )


def _extract_series_eval(run: RunLogs, key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in run.eval_rows:
        ep = _safe_float(row.get("epoch"))
        v = _safe_float(row.get(key))
        if ep is None or v is None:
            continue
        xs.append(ep)
        ys.append(v)
    # ensure ordered by epoch
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def _extract_series_train_step(run: RunLogs, key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in run.train_rows:
        step = _safe_float(row.get("step"))
        v = _safe_float(row.get(key))
        if step is None or v is None:
            continue
        xs.append(step)
        ys.append(v)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def _extract_series_train_epoch_mean(run: RunLogs, key: str) -> Tuple[List[float], List[float]]:
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for row in run.train_rows:
        ep = row.get("epoch")
        v = _safe_float(row.get(key))
        if ep is None or v is None:
            continue
        try:
            epi = int(ep)
        except Exception:
            continue
        sums[epi] = sums.get(epi, 0.0) + v
        counts[epi] = counts.get(epi, 0) + 1
    epochs = sorted(counts.keys())
    xs = [float(ep) for ep in epochs]
    ys = [sums[ep] / counts[ep] for ep in epochs]
    return xs, ys


def _best_epoch(run: RunLogs, key: str = "validation/metric/mAP", max_epoch: Optional[int] = None) -> Optional[Tuple[int, float]]:
    best: Optional[Tuple[int, float]] = None
    for row in run.eval_rows:
        ep = row.get("epoch")
        v = _safe_float(row.get(key))
        if ep is None or v is None:
            continue
        try:
            epi = int(ep)
        except Exception:
            continue
        if max_epoch is not None and epi > max_epoch:
            continue
        if best is None or v > best[1]:
            best = (epi, v)
    return best


def _filter_epoch_window(xs: Sequence[float], ys: Sequence[float], e0: int, e1: int) -> Tuple[List[float], List[float]]:
    x2: List[float] = []
    y2: List[float] = []
    for x, y in zip(xs, ys):
        if e0 <= x <= e1:
            x2.append(x)
            y2.append(y)
    return x2, y2


def _plot_comparisons_for_backbone(
    *,
    runs: List[RunLogs],
    out_dir: Path,
    backbone: str,
    epoch_window: Optional[Tuple[int, int]] = None,
) -> None:
    comp_dir = out_dir / "comparisons"
    _ensure_dir(comp_dir)

    clamp_metrics = {
        "validation/metric/mAP",
        "validation/metric/AP",
        "validation/metric/AP_50",
        "validation/metric/AP_75",
        "validation/metric/AP_90",
        "validation/metric/F1",
        "validation/metric/precision",
        "validation/metric/recall",
        "validation/metric/AR",
        "validation/metric/AP_S",
        "validation/metric/AP_M",
        "validation/metric/AP_L",
        "validation/metric/AP_class/car",
        "validation/metric/AP_class/pedestrian",
    }

    eval_keys = [
        "validation/metric/mAP",
        "validation/metric/AP_50",
        "validation/metric/AP_75",
        "validation/metric/F1",
        "validation/metric/precision",
        "validation/metric/recall",
        "validation/metric/AP_class/car",
        "validation/metric/AP_class/pedestrian",
        "validation/metric/AP_S",
        "validation/metric/AP_M",
        "validation/metric/AP_L",
    ]

    for key in eval_keys:
        series = []
        for r in runs:
            xs, ys = _extract_series_eval(r, key)
            if epoch_window is not None:
                xs, ys = _filter_epoch_window(xs, ys, epoch_window[0], epoch_window[1])
            if not xs:
                continue
            label = f"{r.variant}"
            style = _variant_style(r.variant)
            series.append((label, xs, ys, style))

        if not series:
            continue

        window_tag = ""
        if epoch_window is not None:
            window_tag = f"__e{epoch_window[0]}-{epoch_window[1]}"

        out_path = comp_dir / f"{_slug(backbone)}{window_tag}__{_slug(key)}.svg"
        _line_chart_svg(
            title=f"{backbone}: {_human_metric_label(key)} vs Epoch{(' (window)' if epoch_window else '')}",
            x_label="Epoch",
            y_label=_human_metric_label(key),
            series=series,
            out_path=out_path,
            xlim=(epoch_window[0], epoch_window[1]) if epoch_window is not None else None,
            clamp_01=(key in clamp_metrics),
        )

        # Tail zoom (dense region)
        # Use last 30% epochs in the chosen window.
        max_ep = max(max(xs) for _, xs, __, ___ in series)
        min_ep = min(min(xs) for _, xs, __, ___ in series)
        span = max(1.0, max_ep - min_ep)
        tail_start = max_ep - span * 0.30
        tail_start_i = int(math.floor(tail_start))
        tail_series = []
        for label, xs, ys, style in series:
            xz, yz = _filter_epoch_window(xs, ys, tail_start_i, int(math.ceil(max_ep)))
            if xz:
                tail_series.append((label, xz, yz, style))
        if tail_series:
            out_path = comp_dir / f"{_slug(backbone)}{window_tag}__{_slug(key)}__zoom_tail.svg"
            _line_chart_svg(
                title=f"{backbone}: {_human_metric_label(key)} vs Epoch (tail zoom)",
                x_label="Epoch",
                y_label=_human_metric_label(key),
                series=tail_series,
                out_path=out_path,
                xlim=(tail_start_i, int(math.ceil(max_ep))),
                clamp_01=(key in clamp_metrics),
            )

    # Training comparisons (epoch-mean): loss + components + lr
    train_keys = [
        "training/loss",
        "training/loss/iou_loss",
        "training/loss/conf_loss",
        "training/loss/cls_loss",
        "training/lr",
    ]
    for key in train_keys:
        series = []
        for r in runs:
            xs, ys = _extract_series_train_epoch_mean(r, key)
            if epoch_window is not None:
                xs, ys = _filter_epoch_window(xs, ys, epoch_window[0], epoch_window[1])
            if not xs:
                continue
            series.append((r.variant, xs, ys, _variant_style(r.variant)))
        if not series:
            continue
        window_tag = ""
        if epoch_window is not None:
            window_tag = f"__e{epoch_window[0]}-{epoch_window[1]}"
        out_path = comp_dir / f"{_slug(backbone)}__training{window_tag}__{_slug(key)}.svg"
        _line_chart_svg(
            title=f"{backbone}: {_human_metric_label(key)} (train epoch-mean)",
            x_label="Epoch",
            y_label=_human_metric_label(key),
            series=series,
            out_path=out_path,
            xlim=(epoch_window[0], epoch_window[1]) if epoch_window is not None else None,
            clamp_01=False,
        )

        # Tail zoom for loss curves as well
        max_ep = max(max(xs) for _, xs, __, ___ in series)
        min_ep = min(min(xs) for _, xs, __, ___ in series)
        span = max(1.0, max_ep - min_ep)
        tail_start = max_ep - span * 0.30
        tail_start_i = int(math.floor(tail_start))
        tail_series = []
        for label, xs, ys, style in series:
            xz, yz = _filter_epoch_window(xs, ys, tail_start_i, int(math.ceil(max_ep)))
            if xz:
                tail_series.append((label, xz, yz, style))
        if tail_series:
            out_path = comp_dir / f"{_slug(backbone)}__training{window_tag}__{_slug(key)}__zoom_tail.svg"
            _line_chart_svg(
                title=f"{backbone}: {_human_metric_label(key)} (tail zoom)",
                x_label="Epoch",
                y_label=_human_metric_label(key),
                series=tail_series,
                out_path=out_path,
                xlim=(tail_start_i, int(math.ceil(max_ep))),
                clamp_01=False,
            )


def _plot_best_bars(
    *,
    runs: List[RunLogs],
    out_dir: Path,
    key: str = "validation/metric/mAP",
    by_backbone: Optional[str] = None,
    epoch_window: Optional[Tuple[int, int]] = None,
) -> None:
    comp_dir = out_dir / "comparisons"
    _ensure_dir(comp_dir)

    selected = [r for r in runs if by_backbone is None or r.backbone == by_backbone]

    cats: List[str] = []
    vals: List[float] = []
    styles: List[SeriesStyle] = []
    for r in sorted(selected, key=lambda rr: (rr.backbone, rr.variant)):
        max_ep = None
        if epoch_window is not None:
            max_ep = epoch_window[1]
        best = _best_epoch(r, key=key, max_epoch=max_ep)
        if best is None:
            continue
        cats.append(r.name)
        vals.append(best[1])
        styles.append(_variant_style(r.variant))

    if not cats:
        return

    clamp_01 = key.startswith("validation/metric/")
    window_tag = ""
    if epoch_window is not None:
        window_tag = f"__e0-{epoch_window[1]}"
    if by_backbone is None:
        out_path = comp_dir / f"best__{_slug(key)}__all_runs{window_tag}.svg"
        title = f"Best {_human_metric_label(key)} (all runs){(' window' if epoch_window else '')}"
    else:
        out_path = comp_dir / f"{_slug(by_backbone)}__best__{_slug(key)}__by_variant{window_tag}.svg"
        title = f"{by_backbone}: Best {_human_metric_label(key)} by variant{(' (window)' if epoch_window else '')}"

    _bar_chart_svg(
        title=title,
        x_label="Run",
        y_label=_human_metric_label(key),
        categories=cats,
        values=vals,
        styles=styles,
        out_path=out_path,
        clamp_01=clamp_01,
    )


def _plot_per_run(run: RunLogs, out_dir: Path) -> None:
    run_dir = out_dir / "per_run" / run.name
    _ensure_dir(run_dir)

    eval_keys = [
        "validation/metric/mAP",
        "validation/metric/AP_50",
        "validation/metric/AP_75",
        "validation/metric/F1",
        "validation/metric/precision",
        "validation/metric/recall",
        "validation/metric/AP_class/car",
        "validation/metric/AP_class/pedestrian",
        "validation/metric/AP_S",
        "validation/metric/AP_M",
        "validation/metric/AP_L",
    ]
    for key in eval_keys:
        xs, ys = _extract_series_eval(run, key)
        if not xs:
            continue
        out_path = run_dir / f"{_slug(key)}.svg"
        _line_chart_svg(
            title=f"{run.name}: {_human_metric_label(key)} vs Epoch",
            x_label="Epoch",
            y_label=_human_metric_label(key),
            series=[(run.name, xs, ys, _variant_style(run.variant))],
            out_path=out_path,
            clamp_01=True,
            max_points_per_series=2000,
        )
        # Tail zoom for dense region
        max_ep = max(xs)
        min_ep = min(xs)
        span = max(1.0, max_ep - min_ep)
        tail_start = int(math.floor(max_ep - span * 0.30))
        xz, yz = _filter_epoch_window(xs, ys, tail_start, int(math.ceil(max_ep)))
        if xz:
            out_path = run_dir / f"{_slug(key)}__zoom_tail.svg"
            _line_chart_svg(
                title=f"{run.name}: {_human_metric_label(key)} vs Epoch (tail zoom)",
                x_label="Epoch",
                y_label=_human_metric_label(key),
                series=[(run.name, xz, yz, _variant_style(run.variant))],
                out_path=out_path,
                xlim=(tail_start, int(math.ceil(max_ep))),
                clamp_01=True,
                max_points_per_series=2000,
            )

    # Train step-level curves: loss & lr
    train_step_keys = [
        "training/loss",
        "training/lr",
        "training/loss/iou_loss",
        "training/loss/conf_loss",
        "training/loss/cls_loss",
    ]
    for key in train_step_keys:
        xs, ys = _extract_series_train_step(run, key)
        if not xs:
            continue
        out_path = run_dir / f"{_slug(key)}__by_step.svg"
        _line_chart_svg(
            title=f"{run.name}: {_human_metric_label(key)} vs Step",
            x_label="Step",
            y_label=_human_metric_label(key),
            series=[(run.name, xs, ys, _variant_style(run.variant))],
            out_path=out_path,
            clamp_01=False,
            max_points_per_series=3500,
        )

    # Train epoch-mean curves: loss components + lr (cleaner)
    for key in ["training/loss", "training/lr", "training/loss/iou_loss", "training/loss/conf_loss", "training/loss/cls_loss"]:
        xs, ys = _extract_series_train_epoch_mean(run, key)
        if not xs:
            continue
        out_path = run_dir / f"{_slug(key)}__epoch_mean.svg"
        _line_chart_svg(
            title=f"{run.name}: {_human_metric_label(key)} (epoch mean)",
            x_label="Epoch",
            y_label=_human_metric_label(key),
            series=[(run.name, xs, ys, _variant_style(run.variant))],
            out_path=out_path,
            clamp_01=False,
            max_points_per_series=2000,
        )

    # Internal stats (only present in combine runs)
    internal_keys = sorted({k for row in run.train_rows for k in row.keys() if k.startswith("training/feedback/") or k.startswith("training/gate_")})
    for key in internal_keys:
        xs, ys = _extract_series_train_step(run, key)
        if not xs:
            continue
        out_path = run_dir / f"{_slug(key)}__by_step.svg"
        _line_chart_svg(
            title=f"{run.name}: {_human_metric_label(key)} vs Step",
            x_label="Step",
            y_label=_human_metric_label(key),
            series=[(run.name, xs, ys, _variant_style(run.variant))],
            out_path=out_path,
            clamp_01=False,
            max_points_per_series=3500,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=Path("logs/dsec/detection"), help="Directory containing run subfolders.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/paper_plots"), help="Output root directory.")
    args = parser.parse_args()

    logdir: Path = args.logdir
    outroot: Path = args.outdir
    out_dir = outroot / "dsec_detection"
    _ensure_dir(out_dir)

    runs: List[RunLogs] = []
    for child in sorted(logdir.iterdir()):
        if not child.is_dir():
            continue
        rl = _load_run(child)
        if rl is not None:
            runs.append(rl)

    if not runs:
        print(f"No runs found under {logdir}")
        return 2

    # Summary CSV (best mAP per run)
    summary_path = out_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "backbone", "variant", "best_epoch", "best_mAP"])
        for r in sorted(runs, key=lambda rr: rr.name):
            best = _best_epoch(r, key="validation/metric/mAP", max_epoch=None)
            if best is None:
                continue
            w.writerow([r.name, r.backbone, r.variant, best[0], f"{best[1]:.6f}"])

    # Comparisons: per backbone full window (natural), plus resnet50 fair window 0-29
    by_backbone: Dict[str, List[RunLogs]] = {}
    for r in runs:
        by_backbone.setdefault(r.backbone, []).append(r)

    for backbone, bruns in by_backbone.items():
        _plot_comparisons_for_backbone(runs=sorted(bruns, key=lambda rr: rr.variant), out_dir=out_dir, backbone=backbone, epoch_window=None)
        _plot_best_bars(runs=runs, out_dir=out_dir, key="validation/metric/mAP", by_backbone=backbone, epoch_window=None)

    # ResNet50 fair window 0-29 (all four variants comparable)
    if "resnet50" in by_backbone:
        _plot_comparisons_for_backbone(runs=sorted(by_backbone["resnet50"], key=lambda rr: rr.variant), out_dir=out_dir, backbone="resnet50", epoch_window=(0, 29))
        for k in ["validation/metric/mAP", "validation/metric/AP_50", "validation/metric/AP_75", "validation/metric/F1", "validation/metric/recall"]:
            _plot_best_bars(runs=runs, out_dir=out_dir, key=k, by_backbone="resnet50", epoch_window=(0, 29))

    # All-runs best bar
    _plot_best_bars(runs=runs, out_dir=out_dir, key="validation/metric/mAP", by_backbone=None, epoch_window=None)

    # Per-run plots
    for r in runs:
        _plot_per_run(r, out_dir=out_dir)

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

