#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physiologically-Constrained Synthetic Brain Data Generator
===========================================================

**目的**：生成具有生理约束的合成 fMRI / EEG 时序数据，作为 GNN 模型的输入，
以在可控条件下探索模型的动力学行为。

**为什么需要合成数据？**

真实 fMRI 数据受到被试特异性噪声、扫描仪漂移、运动伪影等干扰，难以独立控制
每个生理参数。合成数据允许：

1. **流形维度控制**：精确设置内在维度（PR），验证模型在已知维度流形上的行为
2. **信噪比扫描**：系统地改变 network_snr，研究模型对噪声的鲁棒性
3. **网络结构变体**：注入已知的网络模式（DMN 主导 vs 视觉网络主导等）
4. **消融实验**：关闭某项生理约束（如取消时域自相关），观察对模型动力学的影响

**生理约束（五层）**

1. **低维流形**（PR ≈ manifold_dim）：
   信号由 ``manifold_dim`` 个主成分生成，主成分方向从真实数据的 FC 矩阵提取
   （或在无真实数据时使用随机正交基）。这保证合成数据的内在维度匹配神经数据。

2. **时域 AR(1) 自相关**（ρ = autocorr_rho）：
   每个潜变量方向独立遵循 AR(1) 过程：
       h[k, t] = ρ · h[k, t-1] + √(1-ρ²) · ε[k, t]
   ρ=0.6–0.8 对应典型 BOLD 时域相关结构。

3. **空间网络结构**（来自真实 FC 或解剖先验）：
   观测信号 x[:, t] = U @ (λ ⊙ h[:, t]) + σ_noise · η
   U 为 FC 矩阵前 manifold_dim 个特征向量，λ 为对应特征值（控制各方向的"信号强度"）。
   如不提供真实数据，则使用指数衰减特征值 + 随机正交 U（模拟广义神经谱）。

4. **BOLD 带通滤波**（0.01–0.1 Hz，可选）：
   应用 4 阶 Butterworth 带通滤波器，模拟血氧神经血管耦合引起的低频偏好，
   同时去除扫描仪漂移（< 0.01 Hz）。若 scipy 不可用，使用均值滤波器替代。

5. **Z-score 归一化**（匹配 V5 格式）：
   每个 ROI 的时序独立 z-score 归一化，与真实数据存储格式完全一致，
   保证合成数据可直接输入模型。

**输出**

- ``synth_NNNNN.pt`` 文件（V5 HeteroData 格式，与真实 graph_cache 兼容）
- ``physiological_synth_comparison.png``：真实 vs 合成数据的 PCA 流形对比图
- ``physiological_synth_results.json``：流形指标（PR、FC 相似度、knn 纯度）

**用法**::

    python -m analysis.physiological_synth \\
        --graphs outputs/graph_cache/ \\
        --output outputs/synthetic/ \\
        [--n-subjects 10] \\
        [--n-timepoints 300] \\
        [--manifold-dim 10] \\
        [--autocorr-rho 0.7] \\
        [--network-snr 2.0] \\
        [--no-hrf-filter] \\
        [--seed 42] \\
        [--modality fmri]

**参考文献**：

  Stringer et al. (2019) Science — neural manifold dimensionality ~10.
  Bhatt et al. (2020) NeuroImage — synthetic fMRI benchmark.
  Laumann et al. (2015) Cerebral Cortex — fMRI noise structure.
  Welvaert & Rosseel (2014) PLoS ONE — neuRosim: fMRI simulation.
  Eklund et al. (2016) PNAS — fMRI statistical validity.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Minimum eigenvalue floor for degenerate FC directions.
# Prevents negative (noise-induced) FC eigenvalues from collapsing
# to zero variance in the signal basis.
_MIN_EIG_FLOOR: float = 0.01

# Training checkpoint stems that should NOT be treated as graph-cache files.
_CHECKPOINT_STEMS: frozenset = frozenset({"best_model", "swa_model"})
_CHECKPOINT_PREFIX: str = "checkpoint_epoch_"


def _find_graph_pts(folder: Path) -> List[Path]:
    """Return sorted list of graph-cache ``.pt`` files (training checkpoints excluded)."""
    result: List[Path] = []
    for f in sorted(folder.glob("*.pt")):
        stem = f.stem
        if stem not in _CHECKPOINT_STEMS and not stem.startswith(_CHECKPOINT_PREFIX):
            result.append(f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reference-data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_checkpoint_file(path: Path) -> bool:
    """Return True if *path* looks like a training checkpoint (not a graph cache)."""
    name = path.stem.lower()
    return any(kw in name for kw in (
        "best_model", "swa_model", "checkpoint", "epoch", "last", "final",
    ))


def _load_reference_statistics(
    graph_paths: List[Path],
    modality: str = "fmri",
    max_files: int = 10,
) -> Dict:
    """Load n_regions, mean FC matrix, and data statistics from real graph caches.

    Parameters
    ----------
    graph_paths:  list of .pt file paths to scan.
    modality:     ``'fmri'`` or ``'eeg'``.
    max_files:    maximum number of files to scan (for speed).

    Returns
    -------
    dict with keys:
        n_regions   – number of ROIs / channels
        fc_mean     – (N, N) mean Pearson FC matrix across loaded files
        n_loaded    – how many files were successfully loaded
        data_mean   – per-region mean (used only as sanity check; z-scored data ≈ 0)
        data_std    – per-region std
        T_ref       – typical T from the first file
        pos         – (N, 3) MNI coordinates, or None
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to load real graph-cache files.  "
            "Install with: pip install torch"
        )

    stats: Dict = {"n_loaded": 0}
    checked = 0

    for p in graph_paths[:max_files]:
        if _is_checkpoint_file(p):
            continue
        try:
            g = torch.load(str(p), map_location="cpu", weights_only=False)
            nt = modality
            if not (hasattr(g, "node_types") and nt in g.node_types):
                continue
            x = g[nt].x  # (N, T, 1)
            if x.ndim != 3:
                continue
            x_np = x[:, :, 0].numpy().astype(np.float64)  # (N, T)
            N, T = x_np.shape

            if "n_regions" not in stats:
                stats["n_regions"] = N
                stats["T_ref"] = T
                stats["fc_sum"] = np.zeros((N, N), dtype=np.float64)
                stats["data_sum"] = np.zeros(N, dtype=np.float64)
                stats["data_sq_sum"] = np.zeros(N, dtype=np.float64)
                # Capture MNI coordinates if available
                pos_attr = getattr(g[nt], "pos", None)
                stats["pos"] = (
                    pos_attr.numpy().astype(np.float32)
                    if pos_attr is not None else None
                )
            elif stats["n_regions"] != N:
                logger.debug("Skipping %s: n_regions mismatch (%d vs %d)", p.name, N, stats["n_regions"])
                continue

            # Pearson FC (safe even if a row is constant)
            std = x_np.std(axis=1, keepdims=True) + 1e-8
            x_z = (x_np - x_np.mean(axis=1, keepdims=True)) / std
            fc = (x_z @ x_z.T) / T
            np.fill_diagonal(fc, 1.0)
            stats["fc_sum"] += fc

            stats["data_sum"] += x_np.mean(axis=1)
            stats["data_sq_sum"] += (x_np ** 2).mean(axis=1)
            stats["n_loaded"] += 1

        except Exception as exc:
            logger.debug("Could not load %s: %s", p.name, exc)
        checked += 1

    if stats.get("n_loaded", 0) == 0:
        raise ValueError(
            f"Could not load any valid {modality} data from the provided graph paths.\n"
            "Ensure the folder contains V5 HeteroData .pt files, not training checkpoints."
        )

    n = stats["n_loaded"]
    stats["fc_mean"] = stats.pop("fc_sum") / n
    stats["data_mean"] = stats.pop("data_sum") / n
    var = stats.pop("data_sq_sum") / n - stats["data_mean"] ** 2
    stats["data_std"] = np.sqrt(np.maximum(var, 1e-8))
    logger.info(
        "加载参考统计：%d 个文件  n_regions=%d  T_ref=%d",
        n, stats["n_regions"], stats["T_ref"],
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Generative model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_generative_basis(
    n_regions: int,
    manifold_dim: int,
    fc_matrix: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build spatial generative basis U and variance weights λ.

    The observation model is:
        x[:, t] = U @ (λ ⊙ h[:, t]) + noise

    Parameters
    ----------
    n_regions:    number of brain regions / channels.
    manifold_dim: number of latent dimensions (controls PR).
    fc_matrix:    (N, N) real FC matrix.  If provided, U = top eigenvectors
                  of FC (preserving real network structure).  Otherwise random.
    rng:          numpy Generator for reproducibility.

    Returns
    -------
    U:       (n_regions, manifold_dim) orthonormal basis columns.
    lambdas: (manifold_dim,) non-negative variance weights.
             λ_k ∝ magnitude of the k-th manifold direction.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    md = min(manifold_dim, n_regions)

    if fc_matrix is not None and fc_matrix.shape == (n_regions, n_regions):
        # Symmetric eigen-decomposition of FC matrix
        # (FC is symmetric by construction; eigh is numerically more stable than eig)
        try:
            eigvals, eigvecs = np.linalg.eigh(fc_matrix)
        except np.linalg.LinAlgError:
            logger.warning("FC matrix eigendecomposition failed; using random basis.")
            fc_matrix = None

    if fc_matrix is None:
        # Random orthonormal basis (fallback when no real data available)
        A = rng.standard_normal((n_regions, md))
        U, _ = np.linalg.qr(A)
        U = U[:, :md]
        # Power-law decaying variances: λ_k = 1 / (k+1)^α, α≈1 (neural data)
        lambdas = 1.0 / (np.arange(1, md + 1, dtype=np.float64) ** 1.0)
    else:
        # Sort descending by eigenvalue magnitude
        idx = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

        U = eigvecs_sorted[:, :md].astype(np.float64)
        # Use |eigenvalue| as variance weight (FC can have negative eigenvalues
        # from noise; cap at _MIN_EIG_FLOOR to avoid degenerate zero-variance directions)
        lambdas = np.maximum(np.abs(eigvals_sorted[:md]), _MIN_EIG_FLOOR)

    # Normalise λ to sum = 1 (relative variance fractions)
    lambdas = lambdas / lambdas.sum()
    return U, lambdas


def _generate_ar1_latent(
    manifold_dim: int,
    n_timepoints: int,
    autocorr_rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate AR(1) latent state matrix of shape (manifold_dim, n_timepoints).

    The stationary AR(1) model:
        h[k, t] = ρ · h[k, t-1] + √(1-ρ²) · ε[k, t],  ε ~ N(0, 1)

    Stationary variance Var(h[k, :]) = 1 for all ρ ∈ (-1, 1).

    Notes
    -----
    Each latent direction is independent.  Cross-directional correlations arise
    only through the spatial basis U in the observation model.  This ensures
    PR ≈ manifold_dim in the generated data.
    """
    rho = float(np.clip(autocorr_rho, -0.99, 0.99))
    noise_std = float(np.sqrt(max(1.0 - rho ** 2, 1e-6)))
    h = np.zeros((manifold_dim, n_timepoints), dtype=np.float64)
    # Initialise from stationary distribution
    h[:, 0] = rng.standard_normal(manifold_dim)
    for t in range(1, n_timepoints):
        h[:, t] = rho * h[:, t - 1] + noise_std * rng.standard_normal(manifold_dim)
    return h


def _apply_hrf_bandpass(
    x: np.ndarray,
    TR: float,
    low_hz: float = 0.01,
    high_hz: float = 0.10,
) -> np.ndarray:
    """Apply BOLD bandpass filter to (N, T) array.

    Uses a 4th-order zero-phase Butterworth filter (scipy) or a moving-average
    fallback.  The BOLD signal is dominated by 0.01–0.1 Hz oscillations;
    filtering mimics the neurovascular coupling temporal smoothing and removes
    scanner drift (< 0.01 Hz).

    References
    ----------
    Biswal et al. (1995) Magn Reson Med — low-frequency BOLD fluctuations.
    """
    N, T = x.shape
    nyq = 0.5 / TR  # Nyquist frequency in Hz
    lo_n = low_hz / nyq
    hi_n = min(high_hz / nyq, 0.99)

    if lo_n >= hi_n:
        logger.warning("Bandpass [%.3f, %.3f] Hz invalid for TR=%.1f s; skipping filter.",
                       low_hz, high_hz, TR)
        return x

    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [lo_n, hi_n], btype="band")
        return filtfilt(b, a, x, axis=1)
    except ImportError:
        # Fallback: box-car low-pass (removes very high frequencies).
        # Approximate half-power cutoff: kernel_width ≈ 1 / (2 × high_hz × TR) samples
        # (the box-car is equivalent to a sinc-like filter with first null at 1/kernel_width Hz)
        kernel_size = max(3, int(1.0 / (high_hz * TR * 2)))
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        out = np.empty_like(x)
        for i in range(N):
            out[i] = np.convolve(x[i], kernel, mode="same")
        return out


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Z-score each row (region/channel) independently.  Safe for constant rows."""
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1,  keepdims=True)
    return (x - mu) / np.where(std < 1e-8, 1.0, std)


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ²  (robust intrinsic dimensionality)."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 0]
    if len(ev) == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Main synthesis function
# ─────────────────────────────────────────────────────────────────────────────

def generate_physiologically_constrained_data(
    graph_folder: Path,
    output_dir: Path,
    n_subjects: int = 10,
    n_timepoints: int = 300,
    TR: float = 2.0,
    manifold_dim: int = 10,
    autocorr_rho: float = 0.7,
    network_snr: float = 2.0,
    apply_hrf_filter: bool = True,
    seed: int = 42,
    modality: str = "fmri",
    plot_comparison: bool = True,
) -> Dict:
    """Generate physiologically-constrained synthetic brain data.

    This function reads the real graph-cache files in ``graph_folder`` to learn
    key statistical properties of the data (number of regions, functional
    connectivity structure).  It then generates ``n_subjects`` synthetic
    records that satisfy five physiological constraints — see module docstring
    for details — and saves them as V5 HeteroData ``.pt`` files that can be
    directly loaded by ``load_graph_for_inference`` and used as GNN input.

    Parameters
    ----------
    graph_folder:
        Path to folder containing real V5 graph-cache ``.pt`` files.
        Used to extract n_regions, FC structure and MNI positions.
        If no files are loadable, a ``ValueError`` is raised with guidance.
    output_dir:
        Where to save synthetic ``.pt`` files and diagnostic plots.
    n_subjects:
        Number of synthetic records to generate.  Each is saved as an
        independent ``.pt`` file named ``synth_{i:05d}.pt``.
    n_timepoints:
        Number of time steps (TRs) per synthetic record.  Minimum 50.
    TR:
        fMRI repetition time in seconds.  Controls the bandpass filter
        and the ``sampling_rate`` attribute stored in the output.
    manifold_dim:
        Target intrinsic manifold dimensionality (sets Participation Ratio
        PR ≈ manifold_dim).  Should match the empirical PR of the real data,
        which is typically 8–15 for whole-brain fMRI.  Setting this too high
        (> 20) will make the manifold indistinguishable from noise.
    autocorr_rho:
        AR(1) temporal autocorrelation coefficient.  0.6–0.8 is typical for
        BOLD signals (Laumann et al. 2015).  Higher values → smoother signals.
    network_snr:
        Signal-to-noise ratio: ratio of within-manifold variance to isotropic
        noise variance.  Higher → cleaner manifold structure, lower PR bias.
        Typical range: 1–5.  Set to 0.5 for high-noise "lesion" conditions.
    apply_hrf_filter:
        Whether to apply a 0.01–0.1 Hz Butterworth bandpass filter after
        generation to simulate hemodynamic smoothing.  Recommended for fMRI;
        set to ``False`` for EEG or for ablation experiments.
    seed:
        Master random seed.  Each subject uses ``seed + subject_index`` so
        the generated set is fully reproducible.
    modality:
        Which node type to create: ``'fmri'`` (default) or ``'eeg'``.
    plot_comparison:
        Whether to generate a comparison figure (real data PCA vs synthetic
        data PCA) and save it as ``physiological_synth_comparison.png``.

    Returns
    -------
    dict with keys:
        synthetic_files:    list of Path objects for the generated .pt files.
        reference_stats:    summary of reference data statistics.
        synthesis_params:   echo of all generation parameters.
        manifold_metrics:   PR, FC_corr (real vs synth), n_timepoints_actual.
        output_dir:         resolved output directory path (str).

    Raises
    ------
    ValueError:
        If no usable graph-cache files are found in ``graph_folder``.
    ImportError:
        If PyTorch is not installed (needed for saving HeteroData).

    Notes
    -----
    **Using the synthetic data with the GNN model**::

        from loader.load_model import load_graph_for_inference, load_trained_model
        model = load_trained_model("best_model.pt")
        graph = load_graph_for_inference("outputs/synthetic/synth_00000.pt")
        # graph now has edges rebuilt and is ready for model.predict_future()

    **Ablation experiments** (vary one parameter at a time)::

        for rho in [0.3, 0.5, 0.7, 0.9]:
            generate_physiologically_constrained_data(
                ..., autocorr_rho=rho, output_dir=f"synth_rho{rho:.1f}/"
            )

    **Manifold dimension scan**::

        for dim in [3, 5, 8, 10, 15, 20]:
            generate_physiologically_constrained_data(
                ..., manifold_dim=dim, output_dir=f"synth_dim{dim}/"
            )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_folder = Path(graph_folder)
    graph_paths = _find_graph_pts(graph_folder)
    if not graph_paths:
        raise ValueError(
            f"在 {graph_folder} 中未找到 .pt 图缓存文件。\n"
            "请确认路径正确，且文件不是训练检查点（best_model.pt 等）。"
        )

    # ── Step 1: Extract reference statistics from real data ──────────────────
    logger.info("Step 1: 从真实数据提取参考统计量 (%d 个文件)…", len(graph_paths))
    ref = _load_reference_statistics(graph_paths, modality=modality, max_files=15)
    n_regions = ref["n_regions"]
    fc_mean   = ref["fc_mean"]
    mni_pos   = ref.get("pos")  # (N, 3) or None

    n_timepoints = max(n_timepoints, 50)  # ensure minimum length
    manifold_dim = min(manifold_dim, n_regions - 1)

    logger.info(
        "  n_regions=%d  manifold_dim=%d  n_timepoints=%d  "
        "autocorr_rho=%.2f  snr=%.1f  hrf_filter=%s",
        n_regions, manifold_dim, n_timepoints,
        autocorr_rho, network_snr, apply_hrf_filter,
    )

    # ── Step 2: Build low-rank generative model from real FC ─────────────────
    logger.info("Step 2: 构建低秩生成模型（FC 主成分）…")
    U, lambdas = _build_generative_basis(
        n_regions, manifold_dim, fc_matrix=fc_mean,
        rng=np.random.default_rng(seed),
    )
    # Signal variance per direction: λ_k scaled by network_snr
    # Noise variance: σ² = 1 / network_snr  (so SNR = signal_var / noise_var = network_snr)
    noise_std = 1.0 / np.sqrt(max(network_snr, 0.01))

    # ── Step 3: Generate synthetic subjects ──────────────────────────────────
    logger.info("Step 3: 生成 %d 个合成被试…", n_subjects)
    synthetic_files: List[Path] = []
    all_synth_trajs: List[np.ndarray] = []  # for comparison plot

    for i in range(n_subjects):
        subj_rng = np.random.default_rng(seed + i)

        # 3a: AR(1) latent trajectory (manifold_dim, T)
        h = _generate_ar1_latent(
            manifold_dim, n_timepoints, autocorr_rho, subj_rng,
        )

        # 3b: Project onto spatial basis — observation model
        #     x[region, t] = U @ (√λ ⊙ h[:, t]) * signal_scale + noise
        #     The √λ normalisation ensures each direction contributes
        #     proportionally to its FC eigenvalue magnitude.
        signal_weights = np.sqrt(lambdas)  # (manifold_dim,)
        x_signal = U @ (signal_weights[:, None] * h)  # (n_regions, T)
        x_noise  = noise_std * subj_rng.standard_normal((n_regions, n_timepoints))
        x_raw = x_signal + x_noise  # (n_regions, T)

        # 3c: Apply BOLD bandpass filter (mimics neurovascular coupling)
        if apply_hrf_filter:
            x_filt = _apply_hrf_bandpass(x_raw, TR=TR, low_hz=0.01, high_hz=0.10)
        else:
            x_filt = x_raw

        # 3d: Z-score per region (matching V5 format: z-scored BOLD)
        x_zscored = _zscore_rows(x_filt)  # (n_regions, T)

        all_synth_trajs.append(x_zscored)

        # 3e: Save as V5 HeteroData .pt
        out_path = output_dir / f"synth_{i:05d}.pt"
        _save_as_heterodata(
            x_zscored, modality=modality, TR=TR, mni_pos=mni_pos,
            out_path=out_path,
        )
        synthetic_files.append(out_path)
        logger.debug("  Saved %s  (shape %s)", out_path.name, x_zscored.shape)

    logger.info("  合成数据已保存至 %s", output_dir)

    # ── Step 4: Compute manifold quality metrics ──────────────────────────────
    logger.info("Step 4: 计算流形质量指标…")
    synth_fc = _compute_mean_fc(all_synth_trajs)

    # Pool first 5 subjects for covariance analysis.
    # 5 is a practical balance: enough for stable covariance estimation
    # (T×5 >> N for typical T≥200) while avoiding memory pressure on large N.
    pool_data = np.vstack([x.T for x in all_synth_trajs[:5]])   # (5T, N)
    synth_pr   = _compute_pca_pr(pool_data)
    n_sig_dim  = _signal_dimension(pool_data)

    # FC similarity: Pearson correlation between upper-triangle entries
    fc_corr = _fc_similarity(fc_mean, synth_fc)
    logger.info(
        "  合成数据 PR=%.1f  MP信号维度=%d (目标 %d)  FC 相似度 r=%.3f",
        synth_pr, n_sig_dim, manifold_dim, fc_corr,
    )

    manifold_metrics = {
        # Primary quality indicators
        "fc_similarity_r": float(fc_corr),
        # Manifold dimensionality
        "synthetic_pr_full": float(synth_pr),
        "n_signal_dimensions_mp": int(n_sig_dim),
        "target_manifold_dim": int(manifold_dim),
        "note_pr": (
            "PR is computed from all N covariance eigenvalues.  "
            "After per-region z-scoring, the noise floor inflates the full PR "
            "above manifold_dim.  Use n_signal_dimensions_mp (Marchenko-Pastur "
            "threshold) for the empirical signal dimension count — this should be "
            "closer to manifold_dim when network_snr is high (≥ 3)."
        ),
        # Generation parameters echo
        "noise_std": float(noise_std),
        "network_snr": float(network_snr),
        "autocorr_rho": float(autocorr_rho),
        "n_timepoints": int(n_timepoints),
        "n_regions": int(n_regions),
        "hrf_filter_applied": apply_hrf_filter,
    }

    # ── Step 5: Comparison plot (real vs synthetic PCA) ─────────────────────
    if plot_comparison:
        logger.info("Step 5: 生成真实 vs 合成数据对比图…")
        try:
            _plot_synth_comparison(
                graph_paths=graph_paths,
                synth_trajs=all_synth_trajs,
                manifold_metrics=manifold_metrics,
                output_dir=output_dir,
                modality=modality,
                n_real_files=min(5, len(graph_paths)),
                max_pts=400,
            )
        except Exception as exc:
            logger.warning("对比图生成失败: %s", exc)

    # ── Step 6: Save JSON ─────────────────────────────────────────────────────
    results = {
        "synthetic_files": [str(p) for p in synthetic_files],
        "reference_stats": {
            "n_regions": int(n_regions),
            "n_ref_files_loaded": int(ref["n_loaded"]),
            "T_ref": int(ref.get("T_ref", 0)),
        },
        "synthesis_params": {
            "n_subjects": n_subjects,
            "n_timepoints": n_timepoints,
            "TR": TR,
            "manifold_dim": int(manifold_dim),
            "autocorr_rho": autocorr_rho,
            "network_snr": network_snr,
            "apply_hrf_filter": apply_hrf_filter,
            "seed": seed,
            "modality": modality,
        },
        "manifold_metrics": manifold_metrics,
        "output_dir": str(output_dir),
    }

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    json_path = output_dir / "physiological_synth_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, ensure_ascii=False)

    logger.info(
        "\n══════════════════════════════════════════════\n"
        "合成数据生成完成\n"
        "  生成被试数: %d  n_regions=%d  T=%d\n"
        "  manifold_dim=%d  autocorr_rho=%.2f  SNR=%.1f\n"
        "  FC 相似度 r=%.3f  MP信号维度=%d (目标 %d)\n"
        "  输出目录: %s\n"
        "══════════════════════════════════════════════",
        n_subjects, n_regions, n_timepoints,
        manifold_dim, autocorr_rho, network_snr,
        fc_corr, n_sig_dim, manifold_dim,
        output_dir,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_as_heterodata(
    x_zscored: np.ndarray,
    modality: str,
    TR: float,
    mni_pos: Optional[np.ndarray],
    out_path: Path,
) -> None:
    """Save (n_regions, T) z-scored array as a V5 HeteroData .pt file.

    Output format:
        g[modality].x            shape [N, T, 1]  float32  (z-scored)
        g[modality].num_nodes    = N
        g[modality].sampling_rate = 1.0 / TR
        g[modality].pos          (N, 3) float32, if mni_pos is not None
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        raise ImportError(
            "PyTorch and torch_geometric are required to save HeteroData files.\n"
            "Install with: pip install torch torch_geometric"
        )

    N, T = x_zscored.shape
    x_tensor = torch.tensor(
        x_zscored[:, :, None].astype(np.float32),  # (N, T, 1)
        dtype=torch.float32,
    )

    g = HeteroData()
    g[modality].x = x_tensor
    g[modality].num_nodes = N
    g[modality].sampling_rate = float(1.0 / TR)
    if mni_pos is not None:
        g[modality].pos = torch.tensor(mni_pos, dtype=torch.float32)

    torch.save(g, str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mean_fc(trajs: List[np.ndarray]) -> np.ndarray:
    """Compute mean Pearson FC matrix across a list of (N, T) arrays."""
    fc_sum = None
    for x in trajs:
        std = x.std(axis=1, keepdims=True) + 1e-8
        x_z = (x - x.mean(axis=1, keepdims=True)) / std
        fc = (x_z @ x_z.T) / x.shape[1]
        np.fill_diagonal(fc, 1.0)
        if fc_sum is None:
            fc_sum = fc
        else:
            fc_sum = fc_sum + fc
    if fc_sum is None:
        raise ValueError("Empty trajectory list")
    return fc_sum / max(len(trajs), 1)


def _compute_pca_pr(data: np.ndarray) -> float:
    """PR from full covariance matrix eigenspectrum (all N eigenvalues, untruncated).

    Notes
    -----
    After per-region z-scoring, all diagonal entries of the covariance matrix
    equal 1, so the noise floor contributes ~N eigenvalues near 1.  The full
    PR will therefore be inflated above the *structural* ``manifold_dim``.
    Use ``_signal_dimension`` for a noise-floor-corrected estimate.
    """
    data_c = data - data.mean(axis=0)
    if data_c.shape[0] < 2 or data_c.shape[1] < 2:
        return float("nan")
    # Use the covariance matrix to get ALL N eigenvalues
    if data_c.shape[0] >= data_c.shape[1]:
        cov = (data_c.T @ data_c) / (data_c.shape[0] - 1)
    else:
        cov = np.cov(data_c.T)
    ev = np.linalg.eigvalsh(cov)
    return _participation_ratio(np.maximum(ev, 0))


def _signal_dimension(data: np.ndarray) -> int:
    """Count eigenvalues above the Marchenko-Pastur (MP) noise threshold.

    The MP upper bound for a random T×N matrix is σ²(1 + √(N/T))².
    For z-scored data (σ²=1 per region), eigenvalues above this bound
    are unlikely to be pure noise and reflect genuine shared structure.

    Returns
    -------
    n_signal : int
        Number of covariance eigenvalues above the MP threshold.
        This is the empirical signal dimension (robust to noise floor).
    """
    T, N = data.shape
    if T < 3 or N < 2:
        return 0
    data_c = data - data.mean(axis=0)
    if T >= N:
        cov = (data_c.T @ data_c) / (T - 1)
    else:
        cov = np.cov(data_c.T)
    ev = np.linalg.eigvalsh(cov)
    # MP threshold: λ_max = (1 + √(N/T))²  (σ²=1 for z-scored data)
    mp_upper = (1.0 + np.sqrt(float(N) / max(float(T), 1))) ** 2
    return int(np.sum(ev > mp_upper))


def _fc_similarity(fc_a: np.ndarray, fc_b: np.ndarray) -> float:
    """Pearson r between upper-triangle entries of two FC matrices."""
    N = fc_a.shape[0]
    triu_idx = np.triu_indices(N, k=1)
    a = fc_a[triu_idx]
    b = fc_b[triu_idx]
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_synth_comparison(
    graph_paths: List[Path],
    synth_trajs: List[np.ndarray],
    manifold_metrics: Dict,
    output_dir: Path,
    modality: str = "fmri",
    n_real_files: int = 5,
    max_pts: int = 400,
) -> None:
    """Generate a 3-row comparison figure.

    Row 1 (Real data):   PC1-PC2 scatter (time-coloured) | FC matrix heatmap
    Row 2 (Synthetic):   PC1-PC2 scatter (time-coloured, same PCA space) | FC heatmap
    Row 3 (Diagnostics): Scree plot with PR lines | Autocorrelation comparison
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except Exception:
            pass
    except ImportError:
        logger.warning("matplotlib not available; skipping comparison plot.")
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("sklearn not available; skipping comparison plot.")
        return

    # ── Load real data ────────────────────────────────────────────────────────
    real_trajs: List[np.ndarray] = []
    try:
        import torch
        for p in graph_paths[:n_real_files]:
            try:
                g = torch.load(str(p), map_location="cpu", weights_only=False)
                if modality not in g.node_types:
                    continue
                x = g[modality].x
                if x.ndim == 3:
                    x = x[:, :, 0]
                real_trajs.append(x.numpy().astype(np.float64))
            except Exception:
                pass
    except ImportError:
        pass

    if not real_trajs:
        logger.warning("No real trajectories loaded; skipping comparison plot.")
        return

    # ── Fit joint PCA on real data ────────────────────────────────────────────
    n_comp = 20
    real_pool   = np.vstack([x.T for x in real_trajs])      # (sum T, N)
    synth_pool  = np.vstack([x.T for x in synth_trajs[:n_real_files]])  # same shape
    all_pool    = np.vstack([real_pool, synth_pool])
    all_pool_c  = all_pool - all_pool.mean(axis=0)

    nc = min(n_comp, all_pool_c.shape[0] - 1, all_pool_c.shape[1] - 1)
    if nc < 4:
        logger.warning("Not enough samples for PCA comparison; skipping.")
        return

    pca = PCA(n_components=nc, random_state=42)
    pca.fit(all_pool_c)
    evr = pca.explained_variance_ratio_
    ev  = pca.explained_variance_

    # Project each group
    real_pca  = pca.transform(real_pool  - all_pool.mean(axis=0))   # (N_r, nc)
    synth_pca = pca.transform(synth_pool - all_pool.mean(axis=0))   # (N_s, nc)

    pc1_var = float(evr[0] * 100)
    pc2_var = float(evr[1] * 100)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 15))
    fig.suptitle(
        "Real vs Physiologically-Constrained Synthetic Data\n"
        "(Shared joint PCA space — same axes for direct comparison)",
        fontsize=11,
    )

    # Helper: scatter time-coloured
    def _time_scatter(ax, pts, title, max_pts=max_pts):
        n = len(pts)
        if n > max_pts:
            idx = np.random.default_rng(0).choice(n, max_pts, replace=False)
            pts = pts[idx]
            c = np.linspace(0, 1, max_pts)
        else:
            c = np.linspace(0, 1, n)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=c, cmap="viridis",
                        s=4, alpha=0.4, rasterized=True)
        ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        return sc

    # Helper: FC heatmap
    def _fc_heatmap(ax, fc, title, n_show=50):
        fc_show = fc[:n_show, :n_show]
        im = ax.imshow(fc_show, aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Pearson r")
        ax.set_title(title)
        ax.set_xlabel(f"ROI (first {n_show})")
        ax.set_ylabel(f"ROI (first {n_show})")

    # ── Row 0: Real data ──────────────────────────────────────────────────────
    real_pr = _participation_ratio(ev)  # joint PCA PR
    sc0 = _time_scatter(
        axes[0, 0], real_pca,
        title=f"Real data — PC1–PC2 (joint PCA)\n(PR={real_pr:.1f}  n_files={len(real_trajs)})",
    )
    plt.colorbar(sc0, ax=axes[0, 0], label="Time (0=start)")

    real_fc = _compute_mean_fc(real_trajs)
    _fc_heatmap(axes[0, 1], real_fc, title="Real data — mean FC matrix")

    # ── Row 1: Synthetic data ─────────────────────────────────────────────────
    n_sig_str = str(int(manifold_metrics.get("n_signal_dimensions_mp", 0)))
    fc_corr   = float(manifold_metrics.get("fc_similarity_r", 0.0))
    sc1 = _time_scatter(
        axes[1, 0], synth_pca,
        title=(
            f"Synthetic data — PC1–PC2 (same PCA space)\n"
            f"(MP signal dim={n_sig_str}  target={manifold_metrics['target_manifold_dim']}  "
            f"FC sim r={fc_corr:.3f})"
        ),
    )
    plt.colorbar(sc1, ax=axes[1, 0], label="Time (0=start)")

    synth_fc = _compute_mean_fc(synth_trajs[:n_real_files])
    _fc_heatmap(axes[1, 1], synth_fc,
                title=f"Synthetic data — mean FC matrix (r={fc_corr:.3f} vs real)")

    # ── Row 2: Diagnostics ────────────────────────────────────────────────────
    # Left: Scree plot with PR lines
    cumvar = np.cumsum(evr) * 100
    ax_scree = axes[2, 0]
    ax_scree.plot(np.arange(1, nc + 1), cumvar, "o-b", ms=4, label="Cumulative var")
    ax_scree.bar(np.arange(1, nc + 1), evr * 100, alpha=0.4, color="steelblue")
    ax_scree.axvline(real_pr,  color="crimson", lw=1.5, linestyle="--",
                     label=f"Joint PR = {real_pr:.1f}")
    n_sig = int(manifold_metrics.get("n_signal_dimensions_mp", 0))
    if n_sig > 0:
        ax_scree.axvline(n_sig, color="darkorange", lw=1.5, linestyle=":",
                         label=f"Synth MP signal dim = {n_sig}")
    ax_scree.axhline(90, color="grey", lw=0.8, linestyle="--", label="90% var")
    ax_scree.set_xlabel("Principal components")
    ax_scree.set_ylabel("Explained variance (%)")
    ax_scree.set_title(
        f"Scree plot (shared PCA space)\n"
        f"target manifold_dim={manifold_metrics['target_manifold_dim']}"
    )
    ax_scree.legend(fontsize=8)
    ax_scree.grid(True, alpha=0.2)

    # Right: Temporal autocorrelation comparison
    ax_ac = axes[2, 1]
    max_lag = min(30, min(x.shape[1] for x in real_trajs + synth_trajs[:3]) - 1)
    lags = np.arange(0, max_lag + 1)

    def _mean_autocorr(trajs_list, max_lag):
        """Mean autocorrelation at each lag, averaged over regions and subjects."""
        acf_sum = np.zeros(max_lag + 1)
        count = 0
        for x in trajs_list[:5]:
            for i in range(min(20, x.shape[0])):
                ts = x[i]
                ts_c = ts - ts.mean()
                norm = float(np.dot(ts_c, ts_c))
                if norm < 1e-10:
                    continue
                for lag in range(max_lag + 1):
                    acf_sum[lag] += float(np.dot(ts_c[:len(ts_c) - lag],
                                                  ts_c[lag:]) / norm)
                count += 1
        return acf_sum / max(count, 1)

    acf_real  = _mean_autocorr(real_trajs, max_lag)
    acf_synth = _mean_autocorr(synth_trajs, max_lag)

    ax_ac.plot(lags, acf_real,  "b-o", ms=3, label="Real",      linewidth=1.5)
    ax_ac.plot(lags, acf_synth, "r-s", ms=3, label="Synthetic", linewidth=1.5, linestyle="--")
    ax_ac.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_ac.set_xlabel("Lag (TRs)")
    ax_ac.set_ylabel("Autocorrelation")
    ax_ac.set_title(
        f"Temporal autocorrelation (mean over regions)\n"
        f"Synthetic: AR(1) rho={manifold_metrics['autocorr_rho']:.2f}"
    )
    ax_ac.legend(fontsize=9)
    ax_ac.grid(True, alpha=0.2)

    plt.tight_layout()
    fig_path = output_dir / "physiological_synth_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → Comparison figure saved: %s", fig_path)

    # ── Also save FC difference heatmap ──────────────────────────────────────
    try:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        n_show = min(50, real_fc.shape[0])
        im0 = axes2[0].imshow(real_fc[:n_show, :n_show], cmap="RdBu_r",
                               vmin=-1, vmax=1, aspect="auto")
        axes2[0].set_title("Real FC")
        plt.colorbar(im0, ax=axes2[0])

        im1 = axes2[1].imshow(synth_fc[:n_show, :n_show], cmap="RdBu_r",
                               vmin=-1, vmax=1, aspect="auto")
        axes2[1].set_title(f"Synthetic FC (r={fc_corr:.3f})")
        plt.colorbar(im1, ax=axes2[1])

        diff = synth_fc[:n_show, :n_show] - real_fc[:n_show, :n_show]
        im2 = axes2[2].imshow(diff, cmap="coolwarm",
                               vmin=-0.5, vmax=0.5, aspect="auto")
        axes2[2].set_title("FC Difference (Synthetic - Real)")
        plt.colorbar(im2, ax=axes2[2])

        plt.tight_layout()
        fc_fig_path = output_dir / "physiological_synth_fc_comparison.png"
        plt.savefig(str(fc_fig_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  → FC comparison figure saved: %s", fc_fig_path)
    except Exception as exc:
        logger.debug("FC comparison plot failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.physiological_synth",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--graphs", required=True, type=Path,
                   help="包含真实 V5 图缓存 .pt 文件的文件夹")
    p.add_argument("--output", required=True, type=Path,
                   help="输出目录（自动创建）")
    p.add_argument("--n-subjects", type=int, default=10,
                   help="生成的合成被试数（默认 10）")
    p.add_argument("--n-timepoints", type=int, default=300,
                   help="每个被试的时间步数（默认 300 TR）")
    p.add_argument("--TR", type=float, default=2.0,
                   help="fMRI 重复时间（秒，默认 2.0）")
    p.add_argument("--manifold-dim", type=int, default=10,
                   help=(
                       "目标流形维度 / 参与率 PR（默认 10）。"
                       "设置为与真实数据的 PR 一致可生成最逼真的合成数据。"
                   ))
    p.add_argument("--autocorr-rho", type=float, default=0.7,
                   help="AR(1) 时域自相关系数（默认 0.7；典型 BOLD: 0.6–0.8）")
    p.add_argument("--network-snr", type=float, default=2.0,
                   help="网络信号 / 噪声方差比（默认 2.0；高值 → 更清晰的流形）")
    p.add_argument("--no-hrf-filter", action="store_true",
                   help="关闭 BOLD 带通滤波器（用于 EEG 或消融实验）")
    p.add_argument("--modality", default="fmri",
                   choices=["fmri", "eeg"],
                   help="目标模态（默认 fmri）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（默认 42）")
    p.add_argument("--no-plot", action="store_true",
                   help="关闭对比图生成（加快速度）")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="显示 DEBUG 级别日志")
    return p


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    generate_physiologically_constrained_data(
        graph_folder=args.graphs,
        output_dir=args.output,
        n_subjects=args.n_subjects,
        n_timepoints=args.n_timepoints,
        TR=args.TR,
        manifold_dim=args.manifold_dim,
        autocorr_rho=args.autocorr_rho,
        network_snr=args.network_snr,
        apply_hrf_filter=not args.no_hrf_filter,
        seed=args.seed,
        modality=args.modality,
        plot_comparison=not args.no_plot,
    )


if __name__ == "__main__":
    main()
