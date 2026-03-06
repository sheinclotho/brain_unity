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
  7  run stability analysis (improved: delay-distance method)
  8  trajectory convergence analysis
  9  Lyapunov exponent estimation
  10 random model comparison

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
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Device helpers ────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the actual device string (``"cuda"`` or ``"cpu"``)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _log_device_info(device: str) -> None:
    """Log compute device and available GPU memory (if CUDA)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        # Normalise "cuda" to an integer device index for consistent property queries
        dev_idx = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(dev_idx)
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        free_mib = free_bytes / (1024 * 1024)
        total_mib = props.total_memory / (1024 * 1024)
        logger.info(
            "GPU: %s  |  VRAM 可用 %.0f MiB / 总计 %.0f MiB",
            props.name,
            free_mib,
            total_mib,
        )
    else:
        logger.info("计算设备: CPU")


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
    "model": {
        "path": None,          # Required: path to best_model.pt
        "graph_path": None,    # Required: path to graph cache .pt file
        "config_path": None,   # Optional: training config.yaml (auto-detected if None)
        "device": "auto",      # "auto" → CUDA if available, else CPU
        "k_cross_modal": 5,    # Cross-modal edges per EEG electrode (API.md §2.5)
    },
    "simulator": {
        "modality": "fmri",    # Which modality to simulate ('fmri' or 'eeg')
        "fmri_subsample": 25,  # Kept for compatibility; model infers dt from graph
    },
    # Steps here mean fMRI prediction steps (not EEG samples).
    # With prediction_steps=50 (TR=2s), steps=50 → 100s of future prediction.
    "free_dynamics": {"n_init": 10, "steps": 50, "seed": 42},
    "attractor_analysis": {
        "tail_steps": 10,
        "k_candidates": [2, 3, 4, 5, 6],
        "k_best": 3,
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 3,
    },
    "virtual_stimulation": {
        "target_nodes": [0, 100],
        "frequency": 10.0,
        "amplitude": 0.5,
        "duration": 30,       # fMRI steps (≈ 60 s at TR=2 s)
        "pre_steps": 10,
        "post_steps": 20,
        "patterns": ["sine"],
    },
    "response_matrix": {
        "n_nodes": 10,         # Reduced default for model mode (full 200 is slow)
        "stim_amplitude": 0.5,
        "stim_duration": 20,
        "stim_frequency": 10.0,
        "stim_pattern": "sine",
        "measure_window": 10,
    },
    "stability_analysis": {
        "convergence_tol": 1e-4,
        "period_max_lag": 100,
        "delay_dt": 50,        # Delay lag for improved stability classification (method B)
    },
    "lyapunov": {
        "enabled": True,
        "epsilon": 1e-5,       # Initial perturbation magnitude
        "skip_fraction": 0.1,  # Skip initial transient fraction when fitting
    },
    "trajectory_convergence": {
        "enabled": True,
        "n_pairs": 50,         # Number of random trajectory pairs
    },
    "random_comparison": {
        "enabled": True,
        "n_init": 200,
        "steps": 1000,
        "spectral_radius": 0.9,
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

    Loads a trained TwinBrainDigitalTwin model and drives the complete
    dynamics-analysis workflow (free dynamics → attractor analysis →
    virtual stimulation → response matrix → stability analysis).

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        results: Dictionary containing all computed artefacts.

    Raises:
        ValueError:   If model_path or graph_path is not specified.
        RuntimeError: If the model fails to load or run inference.
    """
    from loader.load_model import load_trained_model, load_graph_for_inference
    from simulator.brain_dynamics_simulator import BrainDynamicsSimulator
    from experiments.free_dynamics import run_free_dynamics
    from experiments.attractor_analysis import run_attractor_analysis
    from experiments.virtual_stimulation import run_virtual_stimulation
    from analysis.response_matrix import compute_response_matrix
    from analysis.stability_analysis import run_stability_analysis
    from analysis.lyapunov import run_lyapunov_analysis
    from analysis.trajectory_convergence import run_trajectory_convergence
    from analysis.random_comparison import run_random_model_comparison

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", output_dir.resolve())

    # ── Step 1: Load trained model ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 1/7  加载训练模型")
    model_path = cfg["model"].get("path")
    graph_path = cfg["model"].get("graph_path")
    # Resolve "auto" → "cuda" / "cpu" once here; propagate to all components.
    device = _resolve_device(cfg["model"].get("device", "auto"))
    _log_device_info(device)
    # config_path: training config.yaml auto-detected from checkpoint directory,
    # but can be overridden via cfg["model"]["config_path"]
    config_path = cfg["model"].get("config_path")

    if not model_path:
        raise ValueError(
            "必须通过 --model 或配置文件 model.path 指定训练好的模型路径。\n"
            "示例: python run_dynamics_analysis.py "
            "--model outputs/twinbrain_v5_xxx/best_model.pt "
            "--graph outputs/graph_cache/sub-01_notask_xx.pt"
        )

    twin = load_trained_model(
        checkpoint_path=model_path,
        device=device,
        config_path=config_path,
    )

    # ── Step 1b: Load graph cache ─────────────────────────────────────────────
    if not graph_path:
        raise ValueError(
            "必须通过 --graph 或配置文件 model.graph_path 指定图缓存路径。\n"
            "图缓存位于 outputs/graph_cache/<subject_id>_<task>_<hash>.pt\n"
            "示例: --graph outputs/graph_cache/sub-01_notask_ff12ab34.pt"
        )

    k_cross_modal = cfg["model"].get("k_cross_modal", 5)
    base_graph = load_graph_for_inference(
        graph_path=graph_path,
        device=device,
        k_cross_modal=k_cross_modal,
    )

    # ── Step 2: Create simulator ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 2/7  创建动力学模拟器")
    modality = cfg["simulator"].get("modality", "fmri")

    simulator = BrainDynamicsSimulator(
        model=twin,
        base_graph=base_graph,
        modality=modality,
        fmri_subsample=cfg["simulator"].get("fmri_subsample", 25),
        device=device,
    )
    logger.info(
        "  模式=TwinBrainDigitalTwin, modality=%s, n_regions=%d, dt=%.4f s",
        simulator.modality,
        simulator.n_regions,
        simulator.dt,
    )

    results: dict = {}

    # ── Step 3: Free dynamics ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 3/10  自由动力学实验")
    fd_cfg = cfg["free_dynamics"]
    # In model mode, each rollout() call ignores x0 and uses base_graph context.
    # n_init > 1 produces n_init independent rollout segments (identical context
    # but the predictor may have stochastic components from dropout if enabled).
    trajectories = run_free_dynamics(
        simulator=simulator,
        n_init=fd_cfg.get("n_init", 10),
        steps=fd_cfg.get("steps", 50),
        seed=fd_cfg.get("seed", 42),
        output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
        device=device,
    )
    results["trajectories"] = trajectories

    # ── Step 4: Attractor analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 4/10  吸引子分析")
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
    logger.info("步骤 5/10  虚拟刺激实验")
    vs_cfg = cfg["virtual_stimulation"]
    # Clamp target nodes to the actual number of fMRI regions
    all_target_nodes = vs_cfg.get("target_nodes", [0, 100])
    target_nodes = [n for n in all_target_nodes if n < simulator.n_regions]
    if not target_nodes:
        logger.warning(
            "所有目标节点索引超出范围（n_regions=%d），使用节点 0。",
            simulator.n_regions,
        )
        target_nodes = [0]

    stim_results = run_virtual_stimulation(
        simulator=simulator,
        target_nodes=target_nodes,
        amplitude=vs_cfg.get("amplitude", 0.5),
        frequency=vs_cfg.get("frequency", 10.0),
        stim_steps=vs_cfg.get("duration", 30),
        pre_steps=vs_cfg.get("pre_steps", 10),
        post_steps=vs_cfg.get("post_steps", 20),
        patterns=vs_cfg.get("patterns", ["sine"]),
        output_dir=output_dir,
    )
    results["stimulation_results"] = stim_results

    # ── Step 6: Response matrix ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 6/10  响应矩阵计算")
    rm_cfg = cfg["response_matrix"]
    n_nodes_rm = min(
        rm_cfg.get("n_nodes", simulator.n_regions), simulator.n_regions
    )
    response_matrix = compute_response_matrix(
        simulator=simulator,
        n_nodes=n_nodes_rm,
        stim_amplitude=rm_cfg.get("stim_amplitude", 0.5),
        stim_duration=rm_cfg.get("stim_duration", 20),
        stim_frequency=rm_cfg.get("stim_frequency", 10.0),
        stim_pattern=rm_cfg.get("stim_pattern", "sine"),
        measure_window=rm_cfg.get("measure_window", 10),
        output_dir=output_dir if cfg["output"].get("save_response_matrix") else None,
    )
    results["response_matrix"] = response_matrix

    # ── Step 7: Stability analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 7/10  稳定性分析")
    sa_cfg = cfg["stability_analysis"]
    stability_summary = run_stability_analysis(
        trajectories=trajectories,
        convergence_tol=sa_cfg.get("convergence_tol", 1e-4),
        period_max_lag=sa_cfg.get("period_max_lag", 100),
        delay_dt=sa_cfg.get("delay_dt", 50),
        output_dir=output_dir if cfg["output"].get("save_stability_metrics") else None,
    )
    results["stability_summary"] = stability_summary

    # ── Step 8: Trajectory convergence ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 8/10  轨迹收敛分析")
    tc_cfg = cfg.get("trajectory_convergence", {})
    if tc_cfg.get("enabled", True):
        tc_results = run_trajectory_convergence(
            trajectories=trajectories,
            n_pairs=tc_cfg.get("n_pairs", 50),
            seed=cfg["free_dynamics"].get("seed", 42),
            output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
        )
        results["trajectory_convergence"] = tc_results
    else:
        logger.info("  轨迹收敛分析已禁用，跳过。")

    # ── Step 9: Lyapunov exponent ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 9/10  Lyapunov 指数估计")
    lya_cfg = cfg.get("lyapunov", {})
    if lya_cfg.get("enabled", True):
        try:
            lyapunov_results = run_lyapunov_analysis(
                trajectories=trajectories,
                simulator=simulator,
                epsilon=lya_cfg.get("epsilon", 1e-5),
                skip_fraction=lya_cfg.get("skip_fraction", 0.1),
                output_dir=output_dir,
            )
            results["lyapunov"] = lyapunov_results
        except Exception as exc:
            logger.warning("  Lyapunov 分析失败 (%s)，跳过。", exc)
    else:
        logger.info("  Lyapunov 分析已禁用，跳过。")

    # ── Step 10: Random model comparison ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 10/10  随机模型对照实验")
    rc_cfg = cfg.get("random_comparison", {})
    if rc_cfg.get("enabled", True):
        try:
            comparison = run_random_model_comparison(
                trajectories=trajectories,
                attractor_results=attractor_results,
                lyapunov_results=results.get("lyapunov"),
                response_matrix=response_matrix,
                random_n_init=rc_cfg.get("n_init", 200),
                random_steps=rc_cfg.get("steps", 1000),
                spectral_radius=rc_cfg.get("spectral_radius", 0.9),
                seed=cfg["free_dynamics"].get("seed", 42),
                output_dir=output_dir,
            )
            results["random_comparison"] = comparison
        except Exception as exc:
            logger.warning("  随机模型对照实验失败 (%s)，跳过。", exc)
    else:
        logger.info("  随机模型对照实验已禁用，跳过。")

    # ── Visualisations ────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots"):
        _save_plots(results, output_dir, simulator)

    logger.info("=" * 60)
    logger.info("✓ 动力系统分析全部完成（10 步）！结果保存至: %s", output_dir.resolve())
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
            plot_trajectory_convergence,
            plot_lyapunov_histogram,
            plot_basin_sizes,
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

    # ── New plots ──────────────────────────────────────────────────────────────
    tc_results = results.get("trajectory_convergence")
    if tc_results is not None:
        plot_trajectory_convergence(
            tc_results["mean_distances"],
            save_path=plots_dir / "trajectory_convergence.png",
        )

    lya_results = results.get("lyapunov")
    if lya_results is not None:
        plot_lyapunov_histogram(
            lya_results["lyapunov_values"],
            save_path=plots_dir / "lyapunov_histogram.png",
        )

    att_results = results.get("attractor_results")
    if att_results is not None:
        plot_basin_sizes(
            att_results["basin_distribution"],
            save_path=plots_dir / "basin_sizes.png",
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
        help="动力学分析 YAML 配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "训练好的模型检查点路径（best_model.pt）。\n"
            "示例: outputs/<experiment_name>/best_model.pt"
        ),
    )
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help=(
            "图缓存文件路径（outputs/graph_cache/*.pt）。\n"
            "示例: outputs/graph_cache/<subject_id>_<task>_<hash>.pt"
        ),
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        dest="training_config",
        help=(
            "训练时生成的 config.yaml 路径。\n"
            "省略时自动在检查点同目录查找（推荐）。\n"
            "示例: outputs/<experiment_name>/config.yaml"
        ),
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
        help="自由动力学独立预测轮次数（覆盖配置，默认 200）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="每条轨迹的预测步数（fMRI TR 数，覆盖配置）",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["fmri", "eeg"],
        help="分析模态（默认 fmri）",
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
        choices=["cpu", "cuda", "auto"],
        help="计算设备（auto 自动选择 CUDA/CPU）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load and merge dynamics configuration
    file_cfg = _load_config(args.config)
    cfg = _merge_config(_DEFAULTS, file_cfg)

    # Apply CLI overrides (--model and --graph are required by argparse)
    cfg["model"]["path"] = args.model
    cfg["model"]["graph_path"] = args.graph
    if args.training_config:
        cfg["model"]["config_path"] = args.training_config
    if args.output:
        cfg["output"]["directory"] = args.output
    if args.n_init is not None:
        cfg["free_dynamics"]["n_init"] = args.n_init
    if args.steps is not None:
        cfg["free_dynamics"]["steps"] = args.steps
    if args.modality:
        cfg["simulator"]["modality"] = args.modality
    if args.no_plots:
        cfg["output"]["save_plots"] = False
    if args.device:
        cfg["model"]["device"] = args.device

    run(cfg)


if __name__ == "__main__":
    main()
