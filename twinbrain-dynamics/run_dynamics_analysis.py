#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinBrain — Brain Network Dynamics Testbed
===========================================

完整的动力系统分析主执行脚本。

流程：
  1  load trained model
  2  create simulator
  3  run free dynamics
  4  run attractor analysis
  5  run stimulation experiments
  6  compute response matrix
  7  run stability analysis
  8  save results & plots

用法::

    python run_dynamics_analysis.py                          # 使用默认配置
    python run_dynamics_analysis.py --config configs/dynamics_config.yaml
    python run_dynamics_analysis.py --model path/to/model.pt
    python run_dynamics_analysis.py --output results/

该脚本完全独立于训练流程，仅用于模型科学验证。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress Intel OpenMP duplicate-runtime crash (see AGENTS.md)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure the package root is on sys.path when run directly
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Configuration loading ─────────────────────────────────────────────────────

def _load_config(config_path: Path) -> dict:
    """Load YAML config file. Returns empty dict if yaml not available."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        logger.warning("PyYAML 未安装，使用内置默认配置。")
        return {}
    except FileNotFoundError:
        logger.warning("配置文件未找到: %s，使用默认配置。", config_path)
        return {}


def _merge_config(defaults: dict, overrides: dict) -> dict:
    """Recursively merge override dict into defaults."""
    result = dict(defaults)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_config(result[k], v)
        else:
            result[k] = v
    return result


# ── Default configuration ─────────────────────────────────────────────────────

_DEFAULTS = {
    "model": {"path": None, "device": "cpu"},
    "simulator": {"dt": 0.004, "fmri_subsample": 25, "n_regions": 200},
    "free_dynamics": {"n_init": 200, "steps": 1000, "seed": 42},
    "attractor_analysis": {
        "tail_steps": 100,
        "k_candidates": [2, 3, 4, 5, 6],
        "k_best": 3,
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
    },
    "virtual_stimulation": {
        "target_nodes": [0, 100],
        "frequency": 10.0,
        "amplitude": 0.5,
        "duration": 200,
        "pre_steps": 100,
        "post_steps": 200,
        "patterns": ["sine"],
    },
    "response_matrix": {
        "n_nodes": 200,
        "stim_amplitude": 0.5,
        "stim_duration": 50,
        "stim_frequency": 10.0,
        "stim_pattern": "sine",
        "measure_window": 20,
    },
    "stability_analysis": {
        "convergence_tol": 1e-4,
        "period_max_lag": 100,
    },
    "output": {
        "directory": "outputs",
        "save_trajectories": True,
        "save_attractors": True,
        "save_response_matrix": True,
        "save_stability_metrics": True,
        "save_plots": True,
    },
}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict:
    """
    Run the full dynamics analysis pipeline.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        results: Dictionary containing all computed artefacts.
    """
    from loader.load_model import load_trained_model
    from simulator.brain_dynamics_simulator import BrainDynamicsSimulator
    from experiments.free_dynamics import run_free_dynamics
    from experiments.attractor_analysis import run_attractor_analysis
    from experiments.virtual_stimulation import run_virtual_stimulation
    from analysis.response_matrix import compute_response_matrix
    from analysis.stability_analysis import run_stability_analysis

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", output_dir.resolve())

    # ── Step 1: Load trained model ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 1/7  加载训练模型")
    model_path = cfg["model"].get("path")
    device = cfg["model"].get("device", "cpu")
    model = None
    if model_path:
        model = load_trained_model(model_path, device=device)
        if model is None:
            logger.warning("模型加载失败，回退到 Wilson-Cowan 演化模式。")
    else:
        logger.info("未指定模型路径，使用 Wilson-Cowan 演化模式。")

    # ── Step 2: Create simulator ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 2/7  创建动力学模拟器")
    sim_cfg = cfg["simulator"]
    simulator = BrainDynamicsSimulator(
        model=model,
        n_regions=sim_cfg.get("n_regions", 200),
        dt=sim_cfg.get("dt", 0.004),
        fmri_subsample=sim_cfg.get("fmri_subsample", 25),
    )
    logger.info(
        "  n_regions=%d, dt=%.4f s, fmri_subsample=%d",
        simulator.n_regions,
        simulator.dt,
        simulator.fmri_subsample,
    )

    results: dict = {}

    # ── Step 3: Free dynamics ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 3/7  自由动力学实验")
    fd_cfg = cfg["free_dynamics"]
    trajectories = run_free_dynamics(
        simulator=simulator,
        n_init=fd_cfg.get("n_init", 200),
        steps=fd_cfg.get("steps", 1000),
        seed=fd_cfg.get("seed", 42),
        output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
    )
    results["trajectories"] = trajectories

    # ── Step 4: Attractor analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 4/7  吸引子分析")
    att_cfg = cfg["attractor_analysis"]
    attractor_results = run_attractor_analysis(
        trajectories=trajectories,
        tail_steps=att_cfg.get("tail_steps", 100),
        k_candidates=att_cfg.get("k_candidates", [2, 3, 4, 5, 6]),
        k_best=att_cfg.get("k_best", 3),
        dbscan_eps=att_cfg.get("dbscan_eps", 0.5),
        dbscan_min_samples=att_cfg.get("dbscan_min_samples", 5),
        output_dir=output_dir if cfg["output"].get("save_attractors") else None,
    )
    results["attractor_results"] = attractor_results

    # ── Step 5: Virtual stimulation ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 5/7  虚拟刺激实验")
    vs_cfg = cfg["virtual_stimulation"]
    stim_results = run_virtual_stimulation(
        simulator=simulator,
        target_nodes=vs_cfg.get("target_nodes", [0, 100]),
        amplitude=vs_cfg.get("amplitude", 0.5),
        frequency=vs_cfg.get("frequency", 10.0),
        stim_steps=vs_cfg.get("duration", 200),
        pre_steps=vs_cfg.get("pre_steps", 100),
        post_steps=vs_cfg.get("post_steps", 200),
        patterns=vs_cfg.get("patterns", ["sine"]),
        output_dir=output_dir,
    )
    results["stimulation_results"] = stim_results

    # ── Step 6: Response matrix ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 6/7  响应矩阵计算")
    rm_cfg = cfg["response_matrix"]
    response_matrix = compute_response_matrix(
        simulator=simulator,
        n_nodes=rm_cfg.get("n_nodes", simulator.n_regions),
        stim_amplitude=rm_cfg.get("stim_amplitude", 0.5),
        stim_duration=rm_cfg.get("stim_duration", 50),
        stim_frequency=rm_cfg.get("stim_frequency", 10.0),
        stim_pattern=rm_cfg.get("stim_pattern", "sine"),
        measure_window=rm_cfg.get("measure_window", 20),
        output_dir=output_dir if cfg["output"].get("save_response_matrix") else None,
    )
    results["response_matrix"] = response_matrix

    # ── Step 7: Stability analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 7/7  稳定性分析")
    sa_cfg = cfg["stability_analysis"]
    stability_summary = run_stability_analysis(
        trajectories=trajectories,
        convergence_tol=sa_cfg.get("convergence_tol", 1e-4),
        period_max_lag=sa_cfg.get("period_max_lag", 100),
        output_dir=output_dir if cfg["output"].get("save_stability_metrics") else None,
    )
    results["stability_summary"] = stability_summary

    # ── Visualisations ────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots"):
        _save_plots(results, output_dir, simulator)

    logger.info("=" * 60)
    logger.info("✓ 动力系统分析全部完成！结果保存至: %s", output_dir.resolve())
    return results


def _save_plots(results: dict, output_dir: Path, simulator) -> None:
    """Save all visualisations to output_dir/plots/."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        from visualization.trajectory_plot import (
            plot_pca_trajectories,
            plot_region_heatmap,
            plot_trajectory_norms,
        )
        from visualization.response_plot import (
            plot_response_matrix,
            plot_stimulation_response,
        )
    except Exception as exc:
        logger.warning("可视化模块加载失败 (%s)，跳过绘图。", exc)
        return

    trajs = results.get("trajectories")
    if trajs is not None:
        plot_trajectory_norms(
            trajs,
            save_path=plots_dir / "trajectory_norms.png",
        )
        plot_pca_trajectories(
            trajs,
            save_path=plots_dir / "pca_trajectories.png",
        )
        plot_region_heatmap(
            trajs[0],
            title="Free Dynamics — Trajectory 0",
            save_path=plots_dir / "region_heatmap.png",
        )

    R = results.get("response_matrix")
    if R is not None:
        plot_response_matrix(R, save_path=plots_dir / "response_matrix.png")

    stim_res = results.get("stimulation_results", {})
    for pattern, res_list in stim_res.items():
        for res in res_list:
            fname = plots_dir / f"stim_response_{pattern}_node{res.target_node}.png"
            plot_stimulation_response(
                pre_traj=res.pre_trajectory,
                stim_traj=res.stim_trajectory,
                post_traj=res.post_trajectory,
                target_node=res.target_node,
                dt=simulator.dt,
                title=f"Stimulation Response ({pattern}) — Node {res.target_node}",
                save_path=fname,
            )

    logger.info("  → 所有图表已保存至: %s", plots_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TwinBrain — Brain Network Dynamics Testbed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs" / "dynamics_config.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="训练好的模型文件路径 (.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件中的 output.directory）",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="自由动力学初始状态数（覆盖配置）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="每条轨迹的模拟步数（覆盖配置）",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="跳过生成可视化图表",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="计算设备",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load and merge configuration
    file_cfg = _load_config(args.config)
    cfg = _merge_config(_DEFAULTS, file_cfg)

    # Apply CLI overrides
    if args.model:
        cfg["model"]["path"] = args.model
    if args.output:
        cfg["output"]["directory"] = args.output
    if args.n_init:
        cfg["free_dynamics"]["n_init"] = args.n_init
    if args.steps:
        cfg["free_dynamics"]["steps"] = args.steps
    if args.no_plots:
        cfg["output"]["save_plots"] = False
    if args.device:
        cfg["model"]["device"] = args.device

    run(cfg)


if __name__ == "__main__":
    main()
