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

# Fix joblib "Could not find the number of physical cores" warning on Windows:
# joblib queries physical cores via a subprocess call that may fail in some
# environments (e.g. Anaconda on Windows). Silencing it by pinning the count.
# Trade-off: this restricts joblib to 1 worker for parallel calls, which does
# not affect the current pipeline (all parallelism is via PyTorch/CUDA or NumPy).
# Users who need >1 joblib worker can set LOKY_MAX_CPU_COUNT in their shell.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Fix sklearn KMeans "memory leak on Windows with MKL" warning:
# Setting OMP_NUM_THREADS=1 prevents the MKL thread pool from spawning more
# threads than KMeans chunks, eliminating the warning without affecting results.
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
        # Which modality to simulate: 'fmri', 'eeg', 'both', or 'joint'.
        # - 'fmri'  : analyse the fMRI BOLD stream only.
        # - 'eeg'   : analyse the EEG stream only.
        # - 'both'  : run the full pipeline independently for each modality,
        #             outputting separate results under output_dir/fmri/ and
        #             output_dir/eeg/.  Requires both node types in the graph cache.
        # - 'joint' : single pipeline run using a COMBINED fMRI+EEG state vector.
        #             predict_future() is called once; each modality's predictions are
        #             z-score normalised per channel, then concatenated into a joint
        #             state [z_fmri | z_eeg] of dimension N_fmri+N_eeg.  Produces a
        #             single set of dynamics metrics (one Lyapunov exponent, one
        #             attractor, etc.) that reflects the joint multi-scale brain state.
        #             Requires both 'fmri' and 'eeg' nodes in the graph cache.
        "modality": "fmri",
        "fmri_subsample": 25,  # Kept for compatibility; model infers dt from graph
    },
    # Steps here mean fMRI prediction steps (not EEG samples).
    # With prediction_steps=50 (TR=2s), steps=50 → 100s of future prediction.
    "free_dynamics": {"n_init": 10, "steps": 50, "seed": 42, "n_temporal_windows": None},
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
        "n_nodes": 10,         # Number of source nodes to stimulate (full 200 is slow)
        "stim_amplitude": 0.5,
    },
    "stability_analysis": {
        "convergence_tol": 1e-4,
        "period_max_lag": 100,
        "delay_dt": 50,        # Delay lag for improved stability classification (method B)
    },
    "lyapunov": {
        "enabled": True,
        # ── Method selection ──────────────────────────────────────────────────
        # "rosenstein" (default): uses only the pre-computed free-dynamics
        #   trajectories — zero extra model calls.  Strongly recommended for
        #   TwinBrainDigitalTwin (no context-dilution bias).
        # "wolf" / "ftle" / "both": Wolf requires 2 predict_future() calls per
        #   renormalisation period × n_traj trajectories × n_segments segments.
        #   With n_traj=200, steps=1000, renorm_steps=50 (→ 20 periods/traj),
        #   n_segments=1: 200 × 20 × 2 = ~8000 calls → ~60 min on CPU.
        #   Using method="rosenstein" reduces this to 0 extra calls.
        "method": "rosenstein",    # Default: zero extra model calls, no Wolf bias
        "epsilon": 1e-6,           # Nominal perturbation magnitude
        "renorm_steps": 50,        # Steps per Wolf period
        "skip_fraction": 0.1,      # Skip initial transient fraction when fitting (FTLE)
        # Raised from 0.01 to 0.05: distance_ratio=0.010 (e.g. 0.0026/0.2547) now
        # correctly triggers Wolf skip.  The old threshold missed the boundary case
        # where ratio==0.010 due to strict `<` comparison.
        "convergence_threshold": 0.05,
        # ── n_segments ────────────────────────────────────────────────────────
        # For Rosenstein, each extra segment adds negligible cost (pure NumPy);
        # 3 segments samples the attractor at early / mid / late trajectory
        # positions, giving a more robust LLE estimate.  For Wolf/FTLE, each
        # extra segment adds 2 model calls per trajectory (keep at 1 for speed).
        "n_segments": 3,           # Changed from 1 → 3 for better Rosenstein coverage
        "rosenstein_max_lag": 50,  # Rosenstein method: max tracking lag
        "rosenstein_min_sep": 20,  # Rosenstein method: min temporal separation for NN search
        # ── Delay embedding (Takens) ──────────────────────────────────────────
        # Set delay_embed_dim >= 2 to run Rosenstein in a delay-embedded space
        # instead of the raw N-dimensional state space.  This avoids the "curse
        # of dimensionality" (N=190 → all NN distances nearly equal) and improves
        # LLE estimate quality.
        #
        # Recommended value: the FNN min_sufficient_dim from step 12 (embedding
        # dimension analysis).  Typical value: 4–9 for fMRI/EEG brain dynamics.
        # The log from step 12 suggests: FNN min dimension = 4, Takens min = 9.
        #
        # 0 = disabled (default): run Rosenstein in original N-dim space.
        # Set to 7 after running step 12 once to obtain the FNN dimension.
        "delay_embed_dim": 0,      # 0 = disabled; set to FNN dim for better NN quality
        "delay_embed_tau": 1,      # Delay embedding lag (steps)
        # ── Parallelism ───────────────────────────────────────────────────────
        # n_workers > 1 enables parallel computation of Wolf/FTLE across
        # trajectories using a ThreadPoolExecutor.  Requires thread-safe model
        # inference (CPU only; GPU may cause memory contention).
        # Rosenstein is always parallelised when n_workers > 1 (pure NumPy).
        "n_workers": 1,            # 1 = sequential; set >1 for multi-core Wolf/FTLE
    },
    "trajectory_convergence": {
        "enabled": True,
        "n_pairs": 50,         # Number of random trajectory pairs
    },
    "random_comparison": {
        "enabled": True,
        "n_init": 200,
        "steps": 1000,
        "spectral_radii": [0.9, 1.5, 2.0],  # tanh chaos boundary is ρ≈1.5 for n≈190 (not ρ=1)
        "n_seeds": 5,                           # independent W matrices per spectral radius
    },
    # ── New analysis modules (added in critical review v2) ────────────────────
    "information_flow": {
        "enabled": True,
        # Set to a small number by default for speed; increase to n_regions for
        # full N×N TE matrix.  Full matrix (N=200) takes ~5 min on CPU.
        "n_source_regions": 20,
        "n_target_regions": 20,
        "order": 1,        # Markov embedding order (1 = single-step, standard for fMRI)
        "n_bins": 16,      # Histogram bins for discrete TE estimator
    },
    "controllability": {
        "enabled": True,
        "n_communities": 6,    # Number of functional communities (cortical lobes approx.)
        "n_gramian_terms": 100, # Gramian series truncation
    },
    "critical_slowing_down": {
        "enabled": True,
        "window_fraction": 0.5,  # Rolling window as fraction of trajectory length
    },
    # ── GPT-review validation modules (steps 11–12) ───────────────────────────
    "surrogate_test": {
        "enabled": True,
        "n_surrogates": 19,          # 19 surrogates → p < 0.05 (rank test)
        "surrogate_types": ["phase_randomize", "shuffle", "ar"],
        "ar_order": 1,               # AR(p) order for ar_surrogate
        "rosenstein_max_lag": 30,
        "rosenstein_min_sep": 10,
        "n_traj_sample": 5,          # trajectories used (speed vs. precision)
    },
    "embedding_dimension": {
        "enabled": True,
        "fnn_max_dim": 8,            # max embedding dimension to test via FNN
        "fnn_tau": 1,                # delay lag for FNN (steps)
        "corr_dim": True,            # compute Grassberger-Procaccia D₂
        "check_leakage": True,       # run normalisation + PCA leakage checks
        "train_fraction": 0.7,       # fraction of trajectories treated as "train"
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

def _run_single_modality(
    cfg: dict,
    twin,
    base_graph,
    modality: str,
    output_dir: Path,
    device: str,
) -> dict:
    """
    Run the full 12-step dynamics-analysis pipeline for *one* modality.

    Steps:
      2  Create simulator
      3  Free dynamics
      4  Attractor analysis
      5  Virtual stimulation  (skipped for joint mode)
      6  Response matrix      (skipped for joint mode)
      7  Stability analysis
      8  Trajectory convergence
      9  Lyapunov exponent    (joint mode forces method='rosenstein')
      10 Random model comparison
      11 Surrogate test          (GPT-review validation)
      12 Embedding dimension     (GPT-review validation)

    Args:
        cfg:        Merged configuration dict.
        twin:       Loaded TwinBrainDigitalTwin model.
        base_graph: HeteroData graph cache.
        modality:   ``"fmri"``, ``"eeg"``, or ``"joint"``.
        output_dir: Per-modality output directory.
        device:     Compute device string.

    Returns:
        results dict containing all computed artefacts for this modality.
    """
    from simulator.brain_dynamics_simulator import BrainDynamicsSimulator
    from experiments.free_dynamics import run_free_dynamics
    from experiments.attractor_analysis import run_attractor_analysis
    from experiments.virtual_stimulation import run_virtual_stimulation
    from analysis.response_matrix import compute_response_matrix
    from analysis.stability_analysis import run_stability_analysis
    from analysis.lyapunov import run_lyapunov_analysis
    from analysis.trajectory_convergence import run_trajectory_convergence
    from analysis.random_comparison import run_random_model_comparison
    from analysis.surrogate_test import run_surrogate_test
    from analysis.embedding_dimension import run_embedding_dimension_analysis

    output_dir.mkdir(parents=True, exist_ok=True)
    n_total_steps = 12

    # ── Step 2: Create simulator ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 2/%d  创建动力学模拟器 [modality=%s]", n_total_steps, modality)

    simulator = BrainDynamicsSimulator(
        model=twin,
        base_graph=base_graph,
        modality=modality,
        fmri_subsample=cfg["simulator"].get("fmri_subsample", 25),
        device=device,
    )
    if modality == "joint":
        logger.info(
            "  joint 模态: N_fmri=%d, N_eeg=%d, N_joint=%d, dt=%.4f s/TR",
            simulator.n_fmri_regions, simulator.n_eeg_regions,
            simulator.n_regions, simulator.dt,
        )
    else:
        logger.info(
            "  模式=TwinBrainDigitalTwin, modality=%s, n_regions=%d, dt=%.4f s/TR",
            simulator.modality, simulator.n_regions, simulator.dt,
        )

    results: dict = {}

    # ── Step 3: Free dynamics ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 3/%d  自由动力学实验", n_total_steps)
    fd_cfg = cfg["free_dynamics"]
    trajectories = run_free_dynamics(
        simulator=simulator,
        n_init=fd_cfg.get("n_init", 10),
        steps=fd_cfg.get("steps", 50),
        seed=fd_cfg.get("seed", 42),
        output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
        device=device,
        n_temporal_windows=fd_cfg.get("n_temporal_windows", None),
    )
    results["trajectories"] = trajectories

    # ── Step 4: Attractor analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 4/%d  吸引子分析", n_total_steps)
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
    logger.info("步骤 5/%d  虚拟刺激实验", n_total_steps)
    vs_cfg = cfg["virtual_stimulation"]
    all_target_nodes = vs_cfg.get("target_nodes", [0, 100])
    target_nodes = [n for n in all_target_nodes if n < simulator.n_regions]
    if not target_nodes:
        logger.warning(
            "所有目标节点索引超出范围（n_regions=%d），使用节点 0。",
            simulator.n_regions,
        )
        target_nodes = [0]
    if modality == "joint":
        # In joint mode node indices are in [0, N_fmri+N_eeg).
        # Log which modality each target node falls into for transparency.
        for tn in target_nodes:
            _tmod = "fmri" if tn < simulator.n_fmri_regions else "eeg"
            _tch = tn if tn < simulator.n_fmri_regions else tn - simulator.n_fmri_regions
            logger.info(
                "  target_node=%d → %s 节点 %d（联合模态索引映射）",
                tn, _tmod, _tch,
            )

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
    logger.info("步骤 6/%d  响应矩阵计算", n_total_steps)
    rm_cfg = cfg["response_matrix"]
    n_nodes_rm = min(rm_cfg.get("n_nodes", simulator.n_regions), simulator.n_regions)
    if modality == "joint":
        # In joint mode the response matrix covers both fMRI and EEG nodes.
        # The first N_fmri rows are fMRI stimulation effects; the next N_eeg
        # rows are EEG stimulation effects.  Each row is a z-normalised joint
        # response vector of length N_fmri + N_eeg.
        logger.info(
            "  joint 模态响应矩阵：n_nodes=%d（fMRI: [0,%d), EEG: [%d,%d)），"
            "每行为 z-score 联合响应向量（长度 %d）",
            n_nodes_rm, simulator.n_fmri_regions,
            simulator.n_fmri_regions, simulator.n_regions,
            simulator.n_regions,
        )
    response_matrix = compute_response_matrix(
        simulator=simulator,
        n_nodes=n_nodes_rm,
        stim_amplitude=rm_cfg.get("stim_amplitude", 0.5),
        output_dir=output_dir if cfg["output"].get("save_response_matrix") else None,
    )
    results["response_matrix"] = response_matrix

    # ── Step 7: Stability analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 7/%d  稳定性分析", n_total_steps)
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
    logger.info("步骤 8/%d  轨迹收敛分析", n_total_steps)
    tc_cfg = cfg.get("trajectory_convergence", {})
    if tc_cfg.get("enabled", True):
        tc_results = run_trajectory_convergence(
            trajectories=trajectories,
            n_pairs=tc_cfg.get("n_pairs", 50),
            seed=fd_cfg.get("seed", 42),
            output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
        )
        results["trajectory_convergence"] = tc_results
    else:
        logger.info("  轨迹收敛分析已禁用，跳过。")

    # ── Step 9: Lyapunov exponent ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 9/%d  Lyapunov 指数估计", n_total_steps)
    lya_cfg = cfg.get("lyapunov", {})
    if lya_cfg.get("enabled", True):
        try:
            # Joint mode: z-scored unbounded state space → Wolf/FTLE clipping to
            # [0,1] is inappropriate.  Force Rosenstein method which requires
            # no state-space bounds and works directly on the trajectory.
            lya_method = lya_cfg.get("method", "rosenstein")
            if modality == "joint" and lya_method != "rosenstein":
                logger.info(
                    "  joint 模态自动切换 Lyapunov 方法: '%s' → 'rosenstein'。\n"
                    "  原因：Wolf/FTLE 依赖 [0,1] 状态空间裁剪，"
                    "联合模态的 z-score 空间无上下界，裁剪会引入虚假吸引点。\n"
                    "  Rosenstein 方法不依赖状态空间边界，适合 z-score 轨迹。",
                    lya_method,
                )
                lya_method = "rosenstein"

            lyapunov_results = run_lyapunov_analysis(
                trajectories=trajectories,
                simulator=simulator,
                epsilon=lya_cfg.get("epsilon", 1e-6),
                renorm_steps=lya_cfg.get("renorm_steps", 50),
                skip_fraction=lya_cfg.get("skip_fraction", 0.1),
                method=lya_method,
                convergence_result=results.get("trajectory_convergence"),
                convergence_threshold=lya_cfg.get("convergence_threshold", 0.05),
                n_segments=lya_cfg.get("n_segments", 3),
                rosenstein_max_lag=lya_cfg.get("rosenstein_max_lag", 50),
                rosenstein_min_sep=lya_cfg.get("rosenstein_min_sep", 20),
                rosenstein_delay_embed_dim=lya_cfg.get("delay_embed_dim", 0),
                rosenstein_delay_embed_tau=lya_cfg.get("delay_embed_tau", 1),
                n_workers=lya_cfg.get("n_workers", 1),
                output_dir=output_dir,
            )
            results["lyapunov"] = lyapunov_results
            regime = lyapunov_results["chaos_regime"]["regime"]
            interp = lyapunov_results["chaos_regime"]["interpretation_zh"]
            logger.info("  混沌评估: [%s] %s", regime.upper(), interp)
        except Exception as exc:
            logger.warning("  Lyapunov 分析失败 (%s)，跳过。", exc)
    else:
        logger.info("  Lyapunov 分析已禁用，跳过。")

    # ── Step 10: Random model comparison ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 10/%d  随机模型对照实验", n_total_steps)
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
                spectral_radii=rc_cfg.get("spectral_radii", None),
                n_seeds=rc_cfg.get("n_seeds", 5),
                seed=fd_cfg.get("seed", 42),
                output_dir=output_dir,
            )
            results["random_comparison"] = comparison
        except Exception as exc:
            logger.warning("  随机模型对照实验失败 (%s)，跳过。", exc)
    else:
        logger.info("  随机模型对照实验已禁用，跳过。")

    # ── Step 11: Surrogate test ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 11/%d  代替数据检验（Surrogate Test）", n_total_steps)
    st_cfg = cfg.get("surrogate_test", {})
    if st_cfg.get("enabled", True):
        try:
            surrogate_results = run_surrogate_test(
                trajectories=trajectories,
                n_surrogates=st_cfg.get("n_surrogates", 19),
                surrogate_types=st_cfg.get("surrogate_types", None),
                ar_order=st_cfg.get("ar_order", 1),
                rosenstein_max_lag=st_cfg.get("rosenstein_max_lag", 30),
                rosenstein_min_sep=st_cfg.get("rosenstein_min_sep", 10),
                n_traj_sample=st_cfg.get("n_traj_sample", 5),
                seed=fd_cfg.get("seed", 42),
                output_dir=output_dir,
            )
            results["surrogate_test"] = surrogate_results
        except Exception as exc:
            logger.warning("  代替数据检验失败 (%s)，跳过。", exc)
    else:
        logger.info("  代替数据检验已禁用，跳过。")

    # ── Step 12: Embedding dimension & leakage check ──────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 12/%d  嵌入维度验证 & 数据泄漏检查", n_total_steps)
    ed_cfg = cfg.get("embedding_dimension", {})
    if ed_cfg.get("enabled", True):
        try:
            embed_results = run_embedding_dimension_analysis(
                trajectories=trajectories,
                fnn_max_dim=ed_cfg.get("fnn_max_dim", 8),
                fnn_tau=ed_cfg.get("fnn_tau", 1),
                corr_dim=ed_cfg.get("corr_dim", True),
                check_leakage=ed_cfg.get("check_leakage", True),
                train_fraction=ed_cfg.get("train_fraction", 0.7),
                output_dir=output_dir,
            )
            results["embedding_dimension"] = embed_results
        except Exception as exc:
            logger.warning("  嵌入维度分析失败 (%s)，跳过。", exc)
    else:
        logger.info("  嵌入维度分析已禁用，跳过。")

    # ── Visualisations ────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots"):
        _save_plots(results, output_dir, simulator)

    logger.info("=" * 60)
    logger.info(
        "✓ [%s] 动力系统分析全部完成（%d 步）！结果保存至: %s",
        modality.upper(), n_total_steps, output_dir.resolve(),
    )
    return results


def run(cfg: dict) -> dict:
    """
    Run the full dynamics analysis pipeline.

    Loads a trained TwinBrainDigitalTwin model and drives the complete
    dynamics-analysis workflow for one or both modalities.

    When ``cfg["simulator"]["modality"]`` is ``"both"``, the pipeline runs
    twice (once per modality available in the graph cache) and results are
    stored in per-modality subdirectories::

        output_dir/fmri/   ← fMRI results
        output_dir/eeg/    ← EEG results

    The returned dict has the same shape in all three cases:

    * ``modality="fmri"`` → ``{"trajectories": ..., "lyapunov": ..., ...}``
    * ``modality="eeg"``  → same flat dict for EEG
    * ``modality="both"`` → ``{"fmri": {...}, "eeg": {...}}``

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        results: Dictionary containing all computed artefacts.

    Raises:
        ValueError:   If model_path or graph_path is not specified, or if
                      no requested modality is present in the graph cache.
        RuntimeError: If the model fails to load or run inference.
    """
    from loader.load_model import load_trained_model, load_graph_for_inference

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", output_dir.resolve())

    # ── Step 1: Load trained model ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("步骤 1/12  加载训练模型")
    model_path = cfg["model"].get("path")
    graph_path = cfg["model"].get("graph_path")
    # Resolve "auto" → "cuda" / "cpu" once here; propagate to all components.
    device = _resolve_device(cfg["model"].get("device", "auto"))
    _log_device_info(device)
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

    # ── Determine which modalities to run ─────────────────────────────────────
    modality_cfg = cfg["simulator"].get("modality", "fmri")
    available_modalities = list(base_graph.node_types)

    if modality_cfg == "joint":
        # Joint mode: single pipeline run with a combined fMRI+EEG state vector.
        # Requires BOTH fmri and eeg to be present in the graph cache.
        if "fmri" not in available_modalities or "eeg" not in available_modalities:
            raise ValueError(
                f"modality='joint' 需要图缓存同时包含 'fmri' 和 'eeg' 节点。\n"
                f"图缓存节点类型: {available_modalities}\n"
                "scientific note: joint mode 使用单次 predict_future() 调用同时"
                "获取 fMRI+EEG 预测，各自 z-score 归一化后拼接为联合状态向量，"
                "计算统一的混沌/动力学指标。"
            )
        logger.info(
            "modality='joint'：单次模型调用联合处理 fMRI(%d) + EEG(%d) → "
            "拼接 z-score 状态向量，输出单一动力学指标。",
            base_graph["fmri"].x.shape[0],
            base_graph["eeg"].x.shape[0],
        )
        return _run_single_modality(
            cfg=cfg,
            twin=twin,
            base_graph=base_graph,
            modality="joint",
            output_dir=output_dir,
            device=device,
        )

    if modality_cfg == "both":
        modalities_to_run = [m for m in ["fmri", "eeg"] if m in available_modalities]
        if not modalities_to_run:
            raise ValueError(
                f"modality='both' 指定，但图缓存中不含 fmri 或 eeg 节点。\n"
                f"图缓存节点类型: {available_modalities}"
            )
        logger.info(
            "modality='both'：将依次运行 %s（各自输出至 output_dir/<modality>/）",
            modalities_to_run,
        )
    else:
        if modality_cfg not in available_modalities:
            raise ValueError(
                f"请求的 modality='{modality_cfg}' 不在图缓存节点类型中。\n"
                f"图缓存节点类型: {available_modalities}\n"
                f"本模块支持 'fmri' 和 'eeg' 两种模态（'both' 同时运行，'joint' 联合分析）。\n"
                f"其他节点类型（如 {[m for m in available_modalities if m not in ('fmri','eeg')]}）"
                f"暂不支持作为分析模态，请检查 --graph 文件是否正确。"
            )
        modalities_to_run = [modality_cfg]

    # ── Dispatch: single modality → flat result dict; both → nested dict ──────
    if len(modalities_to_run) == 1:
        # Backward-compatible: single flat results dict (no modality wrapper)
        return _run_single_modality(
            cfg=cfg,
            twin=twin,
            base_graph=base_graph,
            modality=modalities_to_run[0],
            output_dir=output_dir,
            device=device,
        )

    # Both modalities: run sequentially, store under output_dir/{modality}/
    all_results: dict = {}
    for modality in modalities_to_run:
        logger.info("=" * 60)
        logger.info("◆ 开始 [%s] 模态完整分析", modality.upper())
        mod_output_dir = output_dir / modality
        mod_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            all_results[modality] = _run_single_modality(
                cfg=cfg,
                twin=twin,
                base_graph=base_graph,
                modality=modality,
                output_dir=mod_output_dir,
                device=device,
            )
        except Exception as exc:
            logger.error("  [%s] 模态分析异常终止: %s", modality.upper(), exc)
            all_results[modality] = {"error": str(exc)}

    logger.info("=" * 60)
    logger.info(
        "✓ 双模态分析全部完成！结果保存至: %s/{%s}",
        output_dir.resolve(),
        ", ".join(modalities_to_run),
    )
    return all_results


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
            plot_lyapunov_growth,
            plot_basin_sizes,
        )
        from visualization.response_plot import (
            plot_response_matrix,
            plot_stimulation_response,
            plot_response_column_stats,
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
        plot_response_column_stats(R, save_path=plots_dir / "response_column_stats.png")

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
        if len(lya_results.get("log_growth_curve", [])) > 0:
            # Use the best (unbiased) LLE estimate for the plot annotation.
            # When Wolf bias is detected, Rosenstein is the primary estimate;
            # using the biased Wolf mean would mis-label the chart.
            wolf_biased = lya_results.get("wolf_bias_warning", False)
            rosen_val   = lya_results.get("mean_rosenstein")
            best_lle = (
                float(rosen_val)
                if (wolf_biased and rosen_val is not None and np.isfinite(float(rosen_val)))
                else float(lya_results["mean_lyapunov"])
            )
            plot_lyapunov_growth(
                lya_results["log_growth_curve"],
                renorm_steps=lya_results.get("renorm_steps", 20),
                mean_lle=best_lle,
                chaos_regime=lya_results["chaos_regime"]["regime"],
                save_path=plots_dir / "lyapunov_growth.png",
            )

    att_results = results.get("attractor_results")
    if att_results is not None:
        plot_basin_sizes(
            att_results["basin_distribution"],
            save_path=plots_dir / "basin_sizes.png",
        )

    logger.info("  → 所有图表已保存至: %s", plots_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

# Quick-mode presets: drastically reduced defaults for fast pre-experiments.
# All values can be further overridden by explicit --n-init / --steps flags.
_QUICK_OVERRIDES: dict = {
    "free_dynamics": {
        "n_init": 20,    # 20 trajectories instead of 200 (10× faster)
        "steps": 200,    # 200 steps instead of 1000 (5× faster)
    },
    "response_matrix": {
        "n_nodes": 10,   # 10 nodes instead of all 190+
    },
    "random_comparison": {
        "n_init": 20,
        "steps": 200,
        "n_seeds": 2,    # fewer seeds per spectral radius
    },
    "lyapunov": {
        "method": "rosenstein",  # fastest method (zero extra model calls)
        "n_segments": 1,
    },
    "surrogate_test": {
        "n_surrogates": 9,       # 9 surrogates (p < 0.10) instead of 19
        "n_traj_sample": 3,
    },
    "embedding_dimension": {
        "fnn_max_dim": 4,        # fewer dimensions to test
        "corr_dim": False,       # skip correlation dimension (slow)
    },
}


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
        choices=["fmri", "eeg", "both", "joint"],
        help=(
            "分析模态：\n"
            "  fmri  — 仅分析 fMRI BOLD 流\n"
            "  eeg   — 仅分析 EEG 流\n"
            "  both  — 对两种模态分别独立运行完整管线，输出至各自子目录\n"
            "  joint — 单次 predict_future() 联合预测 fMRI+EEG，\n"
            "          z-score 归一化后拼接为联合状态向量，输出单一动力学指标"
        ),
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "快速预实验模式：大幅缩减轨迹数量与步数，跳过耗时分析，适合快速验证。\n"
            "  n_init: 200 → 20，steps: 1000 → 200，response_matrix n_nodes: 10，\n"
            "  Lyapunov 方法: rosenstein（零额外模型调用），surrogate: 9 个。\n"
            "  完整分析请去掉此标志，或在 --n-init / --steps 中明确指定数量。"
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        help=(
            "Lyapunov Wolf/FTLE 分析的并行 worker 数（覆盖配置，默认 1）。\n"
            "  CPU 推断：设 4–8 可显著加速 Wolf/FTLE。\n"
            "  GPU 推断：保持 1（多线程 GPU 调用会产生显存竞争）。\n"
            "  Rosenstein 方法（推荐）不受此参数影响（始终并行）。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load and merge dynamics configuration
    file_cfg = _load_config(args.config)
    cfg = _merge_config(_DEFAULTS, file_cfg)

    # Apply --quick preset BEFORE explicit overrides so --n-init/--steps can
    # still override the quick defaults.
    if args.quick:
        cfg = _merge_config(cfg, _QUICK_OVERRIDES)
        logger.info(
            "⚡ 快速预实验模式（--quick）：\n"
            "   n_init=%d, steps=%d, response_matrix n_nodes=%d, "
            "Lyapunov=%s, surrogate=%d 个。\n"
            "   完整分析请去掉 --quick 标志。",
            cfg["free_dynamics"]["n_init"],
            cfg["free_dynamics"]["steps"],
            cfg["response_matrix"]["n_nodes"],
            cfg["lyapunov"]["method"],
            cfg["surrogate_test"]["n_surrogates"],
        )

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
    if args.n_workers is not None:
        cfg["lyapunov"]["n_workers"] = args.n_workers

    run(cfg)


if __name__ == "__main__":
    main()
