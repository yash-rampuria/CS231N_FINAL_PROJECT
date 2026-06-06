"""
Walk data/eval/ and data/processed/ to aggregate every ablation/run we have
artifacts for into a single CSV + markdown table.

Reads:
    nav_policy/data/eval/<run>/summary.json           # offline eval
    nav_policy/data/eval/<run>/per_rollout.csv        # closed-loop eval rollouts
    nav_policy/data/processed/<run>/dagger_summary.json  # DAgger metadata

Writes:
    nav_policy/data/eval/ablation_summary.csv         # one row per run
    nav_policy/data/eval/ablation_summary.md          # markdown table for the report

Each row records: run_tag, kind (offline | closed-loop | dagger), n samples,
closed-loop success rate (if available), closed-loop position tracking RMSE,
offline linear-velocity RMSE, offline yaw-rate RMSE, training T/H, zero-goal
flag, dagger oracle (reference | mpc).

The collector is intentionally read-only and self-contained: it only depends
on the standard library + numpy.  Run it from the nav_policy/ container or
host without touching torch/FiGS.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
except ImportError:  # numpy is optional; closed-loop aggregation uses it.
    np = None  # type: ignore


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _read_per_rollout_csv(p: Path) -> List[Dict[str, str]]:
    with open(p, "r", newline="") as f:
        return list(csv.DictReader(f))


def _mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if np is None:
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / len(vals)
        return m, v ** 0.5
    return float(np.mean(vals)), float(np.std(vals))


def collect_offline(eval_root: Path) -> List[Dict[str, Any]]:
    """One row per offline eval directory."""
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        summary = run_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            data = _read_json(summary)
        except Exception:
            continue
        # offline summaries have these keys; closed-loop ones don't
        if "rmse_lin_vel" not in data:
            continue
        rows.append({
            "kind": "offline",
            "run_tag": data.get("run_tag", run_dir.name),
            "directory": str(run_dir.name),
            "n_samples": int(data.get("n_samples", 0)),
            "T": int(data.get("T", 0)),
            "H": int(data.get("H", 0)),
            "zero_goal_heading": bool(data.get("zero_goal_heading", False)),
            "rmse_lin_vel_mps": float(data["rmse_lin_vel"]),
            "rmse_psi_dot_rad_s": float(data["rmse_psi_dot"]),
            "latency_mean_ms": float(data.get("latency_per_sample_ms", {}).get("mean", 0.0)),
            "checkpoint": str(data.get("checkpoint", "")),
        })
    return rows


def collect_closed_loop(eval_root: Path) -> List[Dict[str, Any]]:
    """One row per closed-loop eval directory."""
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        summary = run_dir / "summary.json"
        per_rollout = run_dir / "per_rollout.csv"
        if not summary.exists() or not per_rollout.exists():
            continue
        try:
            s = _read_json(summary)
        except Exception:
            continue
        # closed-loop summaries have these keys
        if "success_rate" not in s:
            continue
        rows.append({
            "kind": "closed_loop",
            "run_tag": s.get("run_tag", run_dir.name),
            "directory": str(run_dir.name),
            "n_rollouts": int(s.get("n_rollouts", 0)),
            "success_rate": float(s.get("success_rate", 0.0)),
            "bbox_violation_rate": float(s.get("bbox_violation_rate", 0.0)),
            "tracking_rmse_m": float(s.get("mean_tracking_rmse_m", float("nan"))),
            "tracking_rmse_std_m": float(s.get("std_tracking_rmse_m", float("nan"))),
            "final_error_m": float(s.get("mean_final_position_error_m", float("nan"))),
            "vel_rmse_norm_mps": float(s.get("mean_vel_rmse_norm_mps", float("nan"))),
            "yaw_rmse_rad": float(s.get("mean_yaw_rmse_rad", float("nan"))),
            "latency_model_mean_ms": float(s.get("mean_latency_model_ms_mean", float("nan"))),
            "zero_goal_heading": bool(s.get("zero_goal_heading", False)),
            "checkpoint": str(s.get("checkpoint", "")),
        })
    return rows


def collect_dagger(processed_root: Path) -> List[Dict[str, Any]]:
    """One row per DAgger round."""
    rows: List[Dict[str, Any]] = []
    if not processed_root.exists():
        return rows
    for run_dir in sorted(p for p in processed_root.iterdir() if p.is_dir()):
        summary = run_dir / "dagger_summary.json"
        if not summary.exists():
            continue
        try:
            s = _read_json(summary)
        except Exception:
            continue
        rollouts = s.get("rollouts", [])
        n_frames = [r.get("n_frames", 0) for r in rollouts if "n_frames" in r]
        relabel = [r.get("relabel_wall_s", 0.0) for r in rollouts if "relabel_wall_s" in r]
        rows.append({
            "kind": "dagger",
            "run_tag": s.get("run_tag", run_dir.name),
            "directory": str(run_dir.name),
            "round": int(s.get("round", 0)),
            "oracle": str(s.get("oracle_default", "reference")),
            "mpc_policy": str(s.get("mpc_policy") or ""),
            "n_rollouts": len(rollouts),
            "total_frames_relabeled": int(sum(n_frames)),
            "mean_relabel_wall_s": float(sum(relabel) / len(relabel)) if relabel else 0.0,
            "checkpoint": str(s.get("checkpoint", "")),
        })
    return rows


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:                # NaN
            return "--"
        if abs(v) < 1e-3 and v != 0.0:
            return f"{v:.2e}"
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v)


def render_markdown_section(title: str, rows: List[Dict[str, Any]],
                            columns: List[str]) -> str:
    if not rows:
        return f"## {title}\n\n_No runs found._\n"
    out = [f"## {title}", ""]
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    nav_root = Path(__file__).resolve().parent.parent
    eval_root = nav_root / "data" / "eval"
    processed_root = nav_root / "data" / "processed"
    out_dir = eval_root
    out_dir.mkdir(parents=True, exist_ok=True)

    offline = collect_offline(eval_root) if eval_root.exists() else []
    closed_loop = collect_closed_loop(eval_root) if eval_root.exists() else []
    dagger = collect_dagger(processed_root)

    all_rows = offline + closed_loop + dagger
    write_csv(all_rows, out_dir / "ablation_summary.csv")

    md = []
    md.append("# Ablation summary\n")
    md.append("Generated by `scripts/collect_ablations.py` from artifacts in\n"
              "`data/eval/` and `data/processed/`.\n")
    md.append(render_markdown_section(
        "Closed-loop FiGS evaluations", closed_loop,
        columns=["run_tag", "directory", "n_rollouts", "success_rate",
                 "tracking_rmse_m", "final_error_m", "yaw_rmse_rad",
                 "vel_rmse_norm_mps", "latency_model_mean_ms",
                 "zero_goal_heading"],
    ))
    md.append(render_markdown_section(
        "Offline (held-out validation) evaluations", offline,
        columns=["run_tag", "directory", "n_samples", "T", "H",
                 "rmse_lin_vel_mps", "rmse_psi_dot_rad_s",
                 "latency_mean_ms", "zero_goal_heading"],
    ))
    md.append(render_markdown_section(
        "DAgger rounds", dagger,
        columns=["run_tag", "directory", "round", "oracle", "mpc_policy",
                 "n_rollouts", "total_frames_relabeled", "mean_relabel_wall_s"],
    ))
    (out_dir / "ablation_summary.md").write_text("\n".join(md))

    print(f"[collect] wrote {out_dir / 'ablation_summary.csv'}")
    print(f"[collect] wrote {out_dir / 'ablation_summary.md'}")
    print(f"[collect] {len(offline)} offline + {len(closed_loop)} closed-loop "
          f"+ {len(dagger)} dagger rows")


if __name__ == "__main__":
    main()
