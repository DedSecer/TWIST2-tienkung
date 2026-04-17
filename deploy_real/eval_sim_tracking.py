#!/usr/bin/env python
import argparse
import csv
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np
import torch
from rich import print
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import quat_diff_np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "data_utils", "eval_sim_tracking_config.json")


@dataclass
class SimTrajectory:
    time_sec: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    root_pos: Optional[np.ndarray]
    root_quat_xyzw: Optional[np.ndarray]
    root_lin_vel: Optional[np.ndarray]
    root_ang_vel: Optional[np.ndarray]


@dataclass
class EvalRunResult:
    sim_pkl: str
    run_name: str
    out_dir: str
    summary: Dict[str, object]


def load_eval_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    for section in ["weights", "thresholds"]:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Config must contain object section '{section}'")

    config.setdefault("pass_criteria", {})
    if not isinstance(config["pass_criteria"], dict):
        raise ValueError("Config section 'pass_criteria' must be an object")

    config["pass_criteria"].setdefault("success_rate", 0.8)
    config["pass_criteria"].setdefault("score_mean", 1.0)
    config.setdefault("leaderboard", {})
    config["leaderboard"].setdefault("sort_by", "frame_score_mean")
    config["leaderboard"].setdefault("ascending", True)
    return config


def _parse_kv_floats(raw_text: str, defaults: Dict[str, float]) -> Dict[str, float]:
    values = defaults.copy()
    if not raw_text.strip():
        return values
    for segment in raw_text.split(","):
        item = segment.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid key=value pair: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in values:
            raise ValueError(f"Unknown key in config: {key}")
        values[key] = float(value.strip())
    return values


def _safe_array(record: dict, key: str, expected_dim: int) -> Optional[np.ndarray]:
    value = record.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        return None
    return arr


def load_sim_trajectory(sim_pkl: str) -> SimTrajectory:
    with open(sim_pkl, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("Sim recording file is empty or has invalid format.")

    first = raw[0]
    dof_dim = len(first["dof_pos"])

    timestamps = []
    dof_pos = []
    dof_vel = []
    root_pos = []
    root_quat_xyzw = []
    root_lin_vel = []
    root_ang_vel = []
    has_root_pos = True
    has_root_quat = True
    has_root_lin_vel = True
    has_root_ang_vel = True

    for row in raw:
        timestamps.append(float(row["timestamp"]))
        dof_pos.append(np.asarray(row["dof_pos"], dtype=np.float64))
        dof_vel.append(np.asarray(row["dof_vel"], dtype=np.float64))

        pos = _safe_array(row, "root_pos", 3)
        quat_wxyz = _safe_array(row, "root_quat", 4)
        lin_vel = _safe_array(row, "root_lin_vel", 3)
        ang_vel = _safe_array(row, "root_ang_vel", 3)

        if pos is None:
            has_root_pos = False
        else:
            root_pos.append(pos)

        if quat_wxyz is None:
            has_root_quat = False
        else:
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            root_quat_xyzw.append(quat_xyzw / (np.linalg.norm(quat_xyzw) + 1e-8))

        if lin_vel is None:
            has_root_lin_vel = False
        else:
            root_lin_vel.append(lin_vel)

        if ang_vel is None:
            has_root_ang_vel = False
        else:
            root_ang_vel.append(ang_vel)

    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    dof_vel = np.asarray(dof_vel, dtype=np.float64)
    if dof_pos.ndim != 2 or dof_pos.shape[1] != dof_dim:
        raise ValueError("dof_pos has unexpected shape in sim recordings.")
    if dof_vel.ndim != 2 or dof_vel.shape[1] != dof_dim:
        raise ValueError("dof_vel has unexpected shape in sim recordings.")

    timestamps = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    dof_pos = dof_pos[order]
    dof_vel = dof_vel[order]
    time_sec = timestamps - timestamps[0]

    traj = SimTrajectory(
        time_sec=time_sec,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        root_pos=np.asarray(root_pos, dtype=np.float64)[order] if has_root_pos else None,
        root_quat_xyzw=np.asarray(root_quat_xyzw, dtype=np.float64)[order] if has_root_quat else None,
        root_lin_vel=np.asarray(root_lin_vel, dtype=np.float64)[order] if has_root_lin_vel else None,
        root_ang_vel=np.asarray(root_ang_vel, dtype=np.float64)[order] if has_root_ang_vel else None,
    )
    return traj


def sample_reference(motion_lib: MotionLib, motion_id: int, eval_times: np.ndarray, device: torch.device):
    motion_ids = torch.full((eval_times.shape[0],), motion_id, dtype=torch.long, device=device)
    motion_times = torch.as_tensor(eval_times, dtype=torch.float32, device=device)
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, _, _ = motion_lib.calc_motion_frame(
        motion_ids, motion_times
    )
    return {
        "root_pos": root_pos.detach().cpu().numpy(),
        "root_rot_xyzw": root_rot.detach().cpu().numpy(),
        "root_vel": root_vel.detach().cpu().numpy(),
        "root_ang_vel": root_ang_vel.detach().cpu().numpy(),
        "dof_pos": dof_pos.detach().cpu().numpy(),
        "dof_vel": dof_vel.detach().cpu().numpy(),
        "local_key_body_pos": local_key_body_pos.detach().cpu().numpy(),
    }


def to_yaw_local(global_pos: np.ndarray, root_pos: np.ndarray, root_quat_xyzw: np.ndarray) -> np.ndarray:
    rel = global_pos - root_pos[:, None, :]
    yaw_angles = R.from_quat(root_quat_xyzw).as_euler("xyz", degrees=False)[:, 2]
    yaw_rot = R.from_euler("z", yaw_angles, degrees=False).as_matrix()
    yaw_inv = np.transpose(yaw_rot, (0, 2, 1))
    return np.einsum("nij,nkj->nki", yaw_inv, rel)


def apply_quat_to_vectors(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    n, k, _ = vec.shape
    rot = R.from_quat(quat_xyzw)
    return rot.apply(vec.reshape(n * k, 3)).reshape(n, k, 3)


def _build_sim_body_cache(model: mujoco.MjModel) -> Dict[str, int]:
    cache: Dict[str, int] = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            cache[name] = i
    return cache


def compute_sim_key_body_global(
    xml_path: str,
    sim_root_pos: np.ndarray,
    sim_root_quat_xyzw: np.ndarray,
    sim_dof_pos: np.ndarray,
    body_names: List[str],
) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ndof = sim_dof_pos.shape[1]
    if model.nq < 7 + ndof:
        raise ValueError("MuJoCo model nq is smaller than expected qpos length.")

    body_cache = _build_sim_body_cache(model)
    missing = [name for name in body_names if name not in body_cache]
    if missing:
        raise ValueError(f"These key bodies are not found in XML: {missing}")

    body_ids = [body_cache[name] for name in body_names]
    out = np.zeros((sim_dof_pos.shape[0], len(body_ids), 3), dtype=np.float64)
    for i in range(sim_dof_pos.shape[0]):
        data.qpos[:3] = sim_root_pos[i]
        quat_xyzw = sim_root_quat_xyzw[i]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        data.qpos[3:7] = quat_wxyz
        data.qpos[7 : 7 + ndof] = sim_dof_pos[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        out[i] = data.xpos[body_ids]
    return out


def summarize_metric(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "p95": float(np.nanpercentile(values, 95)),
        "max": float(np.nanmax(values)),
    }


def sanitize_run_name(sim_pkl: str) -> str:
    stem = os.path.splitext(os.path.basename(sim_pkl))[0]
    safe = []
    for ch in stem:
        safe.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    result = "".join(safe).strip("._")
    return result or "eval_run"


def resolve_sim_pkls(single_path: str, batch_glob: str) -> List[str]:
    candidates: List[str] = []
    if batch_glob.strip():
        candidates.extend(sorted(glob.glob(batch_glob)))
    elif single_path.strip():
        candidates.append(single_path)

    normalized = []
    seen = set()
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        if not os.path.isfile(abs_path):
            continue
        seen.add(abs_path)
        normalized.append(abs_path)
    if not normalized:
        raise ValueError("No sim recording pkl files found. Check --sim_pkl or --batch_glob.")
    return normalized


def build_metrics_plot(
    time_sec: np.ndarray,
    joint_err: np.ndarray,
    root_pos_err: np.ndarray,
    root_rot_err_deg: np.ndarray,
    key_body_err: np.ndarray,
    velocity_err: np.ndarray,
    frame_score: np.ndarray,
    png_path: str,
) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(time_sec, joint_err, label="joint")
    if np.any(np.isfinite(root_pos_err)):
        plt.plot(time_sec, root_pos_err, label="root_pos")
    if np.any(np.isfinite(root_rot_err_deg)):
        plt.plot(time_sec, root_rot_err_deg, label="root_rot_deg")
    if np.any(np.isfinite(key_body_err)):
        plt.plot(time_sec, key_body_err, label="key_body")
    if np.any(np.isfinite(velocity_err)):
        plt.plot(time_sec, velocity_err, label="velocity")
    plt.plot(time_sec, frame_score, label="frame_score", linewidth=2.0, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title("Simulation vs Reference Tracking Errors")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def evaluate_single_run(
    sim_pkl: str,
    run_out_dir: str,
    args: argparse.Namespace,
    weights: Dict[str, float],
    thresholds: Dict[str, float],
    pass_success_rate: float,
    pass_score_mean: float,
    motion_lib: MotionLib,
    device: torch.device,
) -> EvalRunResult:
    os.makedirs(run_out_dir, exist_ok=True)
    print(f"[cyan]Loading sim trajectory from {sim_pkl}[/cyan]")
    sim = load_sim_trajectory(sim_pkl)

    motion_len = float(motion_lib.get_motion_length(torch.tensor([args.motion_id], device=device))[0].item())
    eval_times = sim.time_sec + args.start_time
    eval_times = np.clip(eval_times, 0.0, motion_len)
    if args.max_duration > 0:
        keep = sim.time_sec <= args.max_duration
        eval_times = eval_times[keep]
        sim = SimTrajectory(
            time_sec=sim.time_sec[keep],
            dof_pos=sim.dof_pos[keep],
            dof_vel=sim.dof_vel[keep],
            root_pos=sim.root_pos[keep] if sim.root_pos is not None else None,
            root_quat_xyzw=sim.root_quat_xyzw[keep] if sim.root_quat_xyzw is not None else None,
            root_lin_vel=sim.root_lin_vel[keep] if sim.root_lin_vel is not None else None,
            root_ang_vel=sim.root_ang_vel[keep] if sim.root_ang_vel is not None else None,
        )

    if eval_times.shape[0] < 2:
        raise ValueError("Not enough frames for evaluation after filtering.")

    ref = sample_reference(motion_lib, args.motion_id, eval_times, device)
    if sim.dof_pos.shape[1] != ref["dof_pos"].shape[1]:
        raise ValueError(
            f"DOF mismatch: sim={sim.dof_pos.shape[1]}, ref={ref['dof_pos'].shape[1]}"
        )

    joint_err = np.mean(np.abs(ref["dof_pos"] - sim.dof_pos), axis=1)
    joint_vel_err = np.mean(np.abs(ref["dof_vel"] - sim.dof_vel), axis=1)

    has_root = sim.root_pos is not None and sim.root_quat_xyzw is not None
    if has_root:
        root_pos_err = np.mean(np.abs(ref["root_pos"] - sim.root_pos), axis=1)
        rotvec = quat_diff_np(sim.root_quat_xyzw, ref["root_rot_xyzw"], scalar_first=False)
        root_rot_err_deg = np.linalg.norm(rotvec, axis=1) * 180.0 / np.pi
    else:
        root_pos_err = np.full_like(joint_err, np.nan)
        root_rot_err_deg = np.full_like(joint_err, np.nan)

    root_vel_components = [joint_vel_err]
    if sim.root_lin_vel is not None:
        root_lin_vel_err = np.mean(np.abs(ref["root_vel"] - sim.root_lin_vel), axis=1)
        root_vel_components.append(root_lin_vel_err)
    if sim.root_ang_vel is not None:
        root_ang_vel_err = np.mean(np.abs(ref["root_ang_vel"] - sim.root_ang_vel), axis=1)
        root_vel_components.append(root_ang_vel_err)
    velocity_err = np.nanmean(np.vstack(root_vel_components), axis=0)

    key_body_err = np.full_like(joint_err, np.nan)
    key_body_used: List[str] = []
    if has_root:
        if args.key_bodies.strip():
            key_body_used = [x.strip() for x in args.key_bodies.split(",") if x.strip()]
        else:
            key_body_used = list(motion_lib._body_link_list)
        if len(key_body_used) > 0:
            try:
                sim_global = compute_sim_key_body_global(
                    args.xml,
                    sim_root_pos=sim.root_pos,
                    sim_root_quat_xyzw=sim.root_quat_xyzw,
                    sim_dof_pos=sim.dof_pos,
                    body_names=key_body_used,
                )
                ref_global = ref["root_pos"][:, None, :] + apply_quat_to_vectors(ref["root_rot_xyzw"], ref["local_key_body_pos"])
                if args.disable_yaw_align:
                    sim_local = sim_global - sim.root_pos[:, None, :]
                    ref_local = ref_global - ref["root_pos"][:, None, :]
                else:
                    sim_local = to_yaw_local(sim_global, sim.root_pos, sim.root_quat_xyzw)
                    ref_local = to_yaw_local(ref_global, ref["root_pos"], ref["root_rot_xyzw"])
                key_body_err = np.mean(np.mean(np.abs(sim_local - ref_local), axis=2), axis=1)
            except Exception as e:
                print(f"[yellow]Skip key body metric for {sim_pkl}: {e}[/yellow]")

    metrics = {
        "joint": joint_err,
        "root_pos": root_pos_err,
        "root_rot": root_rot_err_deg,
        "key_body": key_body_err,
        "velocity": velocity_err,
    }

    frame_score = np.zeros_like(joint_err)
    available_metric_count = 0
    success_mask = np.ones_like(joint_err, dtype=bool)
    for metric_name, values in metrics.items():
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        available_metric_count += 1
        normalized = np.zeros_like(values)
        normalized[valid] = values[valid] / thresholds[metric_name]
        frame_score += weights[metric_name] * normalized
        success_mask = success_mask & (normalized <= 1.0)

    if available_metric_count == 0:
        raise RuntimeError("No valid metrics were computed. Check input recording fields.")

    if args.success_mode == "score":
        success_mask = frame_score <= args.score_pass

    success_rate = float(np.mean(success_mask))
    frame_score_summary = summarize_metric(frame_score)
    passed = bool(success_rate >= pass_success_rate and frame_score_summary["mean"] <= pass_score_mean)
    run_name = sanitize_run_name(sim_pkl)
    summary = {
        "run_name": run_name,
        "sim_pkl": sim_pkl,
        "motion_file": args.motion_file,
        "motion_id": args.motion_id,
        "num_frames": int(joint_err.shape[0]),
        "start_time": args.start_time,
        "max_duration": args.max_duration,
        "weights": weights,
        "thresholds": thresholds,
        "success_mode": args.success_mode,
        "score_pass": args.score_pass,
        "success_rate_pass_threshold": pass_success_rate,
        "score_mean_pass_threshold": pass_score_mean,
        "success_rate": success_rate,
        "pass": passed,
        "available_metrics": {k: bool(np.any(np.isfinite(v))) for k, v in metrics.items()},
        "metrics": {k: summarize_metric(v[np.isfinite(v)]) for k, v in metrics.items() if np.any(np.isfinite(v))},
        "frame_score": frame_score_summary,
        "key_bodies_used": key_body_used,
    }

    csv_path = os.path.join(run_out_dir, "frame_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "time_sec",
                "joint_error",
                "root_pos_error",
                "root_rot_error_deg",
                "key_body_error",
                "velocity_error",
                "frame_score",
                "success",
            ]
        )
        for i in range(joint_err.shape[0]):
            writer.writerow(
                [
                    i,
                    float(sim.time_sec[i]),
                    float(joint_err[i]),
                    float(root_pos_err[i]) if np.isfinite(root_pos_err[i]) else "",
                    float(root_rot_err_deg[i]) if np.isfinite(root_rot_err_deg[i]) else "",
                    float(key_body_err[i]) if np.isfinite(key_body_err[i]) else "",
                    float(velocity_err[i]) if np.isfinite(velocity_err[i]) else "",
                    float(frame_score[i]),
                    int(success_mask[i]),
                ]
            )

    json_path = os.path.join(run_out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    png_path = os.path.join(run_out_dir, "metrics_plot.png")
    build_metrics_plot(
        sim.time_sec,
        joint_err,
        root_pos_err,
        root_rot_err_deg,
        key_body_err,
        velocity_err,
        frame_score,
        png_path,
    )

    print(f"[green]Finished {run_name}[/green]")
    print(f"  success_rate: {summary['success_rate']:.4f}")
    print(f"  frame_score_mean: {summary['frame_score']['mean']:.4f}")
    print(f"  summary: {json_path}")
    return EvalRunResult(sim_pkl=sim_pkl, run_name=run_name, out_dir=run_out_dir, summary=summary)


def write_leaderboard(
    results: List[EvalRunResult],
    out_dir: str,
    sort_by: str,
    ascending: bool,
) -> None:
    def row_from_result(result: EvalRunResult) -> Dict[str, object]:
        summary = result.summary
        metrics = summary["metrics"]
        return {
            "rank": 0,
            "run_name": result.run_name,
            "sim_pkl": result.sim_pkl,
            "pass": summary["pass"],
            "success_rate": summary["success_rate"],
            "frame_score_mean": summary["frame_score"]["mean"],
            "frame_score_p95": summary["frame_score"]["p95"],
            "joint_mean": metrics.get("joint", {}).get("mean", None),
            "root_pos_mean": metrics.get("root_pos", {}).get("mean", None),
            "root_rot_mean": metrics.get("root_rot", {}).get("mean", None),
            "key_body_mean": metrics.get("key_body", {}).get("mean", None),
            "velocity_mean": metrics.get("velocity", {}).get("mean", None),
            "summary_json": os.path.join(result.out_dir, "summary.json"),
        }

    rows = [row_from_result(result) for result in results]
    valid_sort_keys = set(rows[0].keys()) - {"rank", "summary_json", "run_name", "sim_pkl"} if rows else set()
    if sort_by not in valid_sort_keys:
        raise ValueError(f"Unsupported leaderboard sort field: {sort_by}")

    rows.sort(
        key=lambda row: (row[sort_by] is None, row[sort_by]),
        reverse=not ascending,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    leaderboard_json = {
        "sort_by": sort_by,
        "ascending": ascending,
        "num_runs": len(rows),
        "rows": rows,
    }
    json_path = os.path.join(out_dir, "leaderboard.json")
    with open(json_path, "w") as f:
        json.dump(leaderboard_json, f, indent=2)

    csv_path = os.path.join(out_dir, "leaderboard.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[bold green]Leaderboard[/bold green]")
    for row in rows[:10]:
        print(
            f"  #{row['rank']} {row['run_name']} | score={row['frame_score_mean']:.4f} | "
            f"success={row['success_rate']:.4f} | pass={row['pass']}"
        )
    print(f"  leaderboard json: {json_path}")
    print(f"  leaderboard csv: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluator: simulated trajectory vs reference motion")
    parser.add_argument("--sim_pkl", type=str, default="twist2_proprio_recordings.pkl", help="Recorded sim trajectory pkl")
    parser.add_argument("--batch_glob", type=str, default="", help="Glob for batch evaluation, e.g. 'deploy_real/records/*.pkl'")
    parser.add_argument("--motion_file", type=str, required=True, help="MotionLib pkl/yaml path")
    parser.add_argument("--xml", type=str, default="../assets/g1/g1_sim2sim_29dof.xml", help="MuJoCo XML for key body FK")
    parser.add_argument("--out_dir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", choices=["unitree_g1", "unitree_g1_with_hands"])
    parser.add_argument("--motion_id", type=int, default=0, help="Motion id in motion library")
    parser.add_argument("--start_time", type=float, default=0.0, help="Reference motion start time in seconds")
    parser.add_argument("--max_duration", type=float, default=-1.0, help="Maximum evaluation duration, <=0 means all")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for MotionLib")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="JSON config path for default thresholds/weights")
    parser.add_argument("--weights", type=str, default="", help="Override weights like joint=0.35,key_body=0.25")
    parser.add_argument("--thresholds", type=str, default="", help="Override thresholds with same key=value format")
    parser.add_argument("--success_mode", type=str, default="all", choices=["all", "score"], help="Success rule")
    parser.add_argument("--score_pass", type=float, default=1.0, help="Pass score threshold when success_mode=score")
    parser.add_argument("--disable_yaw_align", action="store_true", help="Disable yaw-only alignment for key body")
    parser.add_argument("--key_bodies", type=str, default="", help="Comma separated key body names; default uses motion body list")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_eval_config(args.config)
    weights = _parse_kv_floats(args.weights, config["weights"])
    thresholds = _parse_kv_floats(args.thresholds, config["thresholds"])
    pass_success_rate = float(config["pass_criteria"].get("success_rate", 0.8))
    pass_score_mean = float(config["pass_criteria"].get("score_mean", 1.0))
    leaderboard_sort_by = str(config["leaderboard"].get("sort_by", "frame_score_mean"))
    leaderboard_ascending = bool(config["leaderboard"].get("ascending", True))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[cyan]Loading reference motion from {args.motion_file} on {device}[/cyan]")
    motion_lib = MotionLib(args.motion_file, device=device)
    if args.motion_id < 0 or args.motion_id >= motion_lib.num_motions():
        raise ValueError(f"motion_id out of range: {args.motion_id}, num_motions={motion_lib.num_motions()}")

    sim_pkls = resolve_sim_pkls(args.sim_pkl, args.batch_glob)
    batch_mode = len(sim_pkls) > 1 or bool(args.batch_glob.strip())
    print(f"[cyan]Running {'batch' if batch_mode else 'single'} evaluation on {len(sim_pkls)} file(s)[/cyan]")

    results: List[EvalRunResult] = []
    for sim_pkl in sim_pkls:
        run_name = sanitize_run_name(sim_pkl)
        run_out_dir = os.path.join(args.out_dir, run_name) if batch_mode else args.out_dir
        result = evaluate_single_run(
            sim_pkl=sim_pkl,
            run_out_dir=run_out_dir,
            args=args,
            weights=weights,
            thresholds=thresholds,
            pass_success_rate=pass_success_rate,
            pass_score_mean=pass_score_mean,
            motion_lib=motion_lib,
            device=device,
        )
        results.append(result)

    if batch_mode:
        write_leaderboard(results, args.out_dir, leaderboard_sort_by, leaderboard_ascending)


if __name__ == "__main__":
    main()