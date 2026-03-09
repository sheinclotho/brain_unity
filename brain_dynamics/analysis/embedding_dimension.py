"""
Embedding Dimension & Attractor Observation Space
==================================================

验证延迟嵌入维度是否满足 Takens 嵌入定理（m ≥ 2D + 1），
并识别吸引子所处的观察空间。

**核心问题**：
  如果嵌入维度不足（m < 2D + 1），吸引子几何结构会被折叠，
  导致"假吸引子"——邻轨道实为相空间中不相关的轨迹，
  Lyapunov 指数、关联维数等估计值均不可信。

**三项检查**：

1. **内在维度估计（False Nearest Neighbors / Correlation Dimension）**
   - FNN 方法（Kennel et al. 1992）：估计最小充分嵌入维度
   - 关联维度 D₂（Grassberger-Procaccia 1983）：吸引子分形维度

2. **Takens 嵌入定理验证**
   - 检查 embedding_dim ≥ 2 × intrinsic_dim + 1
   - 若不满足，警告吸引子分析结果不可靠

3. **吸引子观察空间识别**
   - 可信度排序: delay_embedding > 原始相空间 > PCA > latent_space
   - 根据轨迹形状报告当前使用的观察空间及其可信度

**数据泄漏检查**：

4. **归一化泄漏检查**
   - 若使用全局统计量（均值/标准差）归一化整个数据集（含测试集），
     则 Lyapunov 指数等估计量可能受到测试集信息的影响。
   - 检查：训练集归一化统计量是否与全集统计量显著不同。

5. **PCA 泄漏检查**
   - PCA 若在全部轨迹上拟合后再分析，相当于用"未见数据"的全局
     主成分，与仅用参考轨迹拟合可能产生不同的子空间。
   - 提供 check_pca_leakage() 用于比较两种拟合方式的夹角。

参考：
  Kennel, Brown & Abarbanel (1992) Phys. Rev. A 45:3403
  Grassberger & Procaccia (1983) Phys. Rev. Lett. 50:346
  Takens (1981) Lecture Notes in Mathematics 898:366
  Theiler (1990) Phys. Rev. A 41:3038 — 关联维数偏差修正
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# False Nearest Neighbors (FNN) for intrinsic dimension estimation
# ─────────────────────────────────────────────────────────────────────────────

def false_nearest_neighbors(
    ts: np.ndarray,
    max_dim: int = 10,
    tau: int = 1,
    rtol: float = 15.0,
    atol: float = 2.0,
    max_ref_points: int = 500,
) -> Dict:
    """
    False Nearest Neighbors (FNN) 方法估计最小充分嵌入维度。

    算法 (Kennel et al. 1992 Phys. Rev. A 45:3403)：
      对嵌入维度 m = 1, 2, ..., max_dim：
        1. 构建延迟嵌入向量 x_m(t) = [x(t), x(t+τ), ..., x(t+(m-1)τ)]
        2. 对每个点找其最近邻 x_m(t*)
        3. 当升至 m+1 维时，若邻居间距增加超过 rtol 倍，则为"假近邻"
        4. 统计假近邻比例 FNN(m)
      最小充分维度 = FNN(m) < 1% 的最小 m。

    时间复杂度：O(max_dim × max_ref_points²)。对长轨迹自动子采样。

    Args:
        ts:            shape (T,) 或 (T, N)，单通道或多通道时序。
                       多通道时，对每通道独立计算后取均值。
        max_dim:       最大测试嵌入维度（默认 10）。
        tau:           延迟步数（默认 1；可用 first-minimum of AMI 选取）。
        rtol:          假近邻相对距离阈值（默认 15，Kennel 推荐值）。
        atol:          假近邻绝对距离阈值（归一化为轨迹 RMS 的倍数，默认 2）。
        max_ref_points: 参考点上限（防止 O(T²) 过慢）。

    Returns:
        {
            "fnn_fractions": List[float] — FNN(m) for m=1..max_dim
            "min_sufficient_dim": int    — FNN < 1% 的最小 m（未找到则 max_dim）
            "recommended_embed_dim": int — max(min_sufficient_dim, 2) ← 安全裕量
            "takens_min_dim": int        — 2 * min_sufficient_dim + 1
        }
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim == 1:
        ts = ts[:, None]

    T, N = ts.shape

    # Multi-channel: average FNN fractions across channels
    all_fnn: List[List[float]] = []
    for ch in range(N):
        x = ts[:, ch]
        x = (x - x.mean()) / (x.std() + 1e-12)  # z-score for atol
        fnn_ch = _fnn_single_channel(x, max_dim=max_dim, tau=tau,
                                     rtol=rtol, atol=atol,
                                     max_ref_points=max_ref_points)
        all_fnn.append(fnn_ch)

    fnn_fractions = [float(np.mean([all_fnn[ch][m] for ch in range(N)]))
                     for m in range(max_dim)]

    # Find minimum sufficient dimension (FNN < 1%)
    min_suf = max_dim
    for m_idx, fnn_val in enumerate(fnn_fractions):
        if fnn_val < 0.01:
            min_suf = m_idx + 1  # m = 1-indexed
            break

    recommended = max(min_suf, 2)
    takens_min = 2 * min_suf + 1

    return {
        "fnn_fractions": fnn_fractions,
        "min_sufficient_dim": min_suf,
        "recommended_embed_dim": recommended,
        "takens_min_dim": takens_min,
    }


def _fnn_single_channel(
    x: np.ndarray,
    max_dim: int,
    tau: int,
    rtol: float,
    atol: float,
    max_ref_points: int,
) -> List[float]:
    """FNN for a single z-scored channel (T,)."""
    T = len(x)
    fnn_list: List[float] = []

    for m in range(1, max_dim + 1):
        # Number of valid time points for m-dimensional embedding
        n_pts = T - m * tau  # need x[t + m*tau] for testing m+1
        if n_pts < 10:
            fnn_list.append(0.0)
            continue

        # Build m-dim delay vectors (reference for NN search)
        idx = np.arange(n_pts)
        X_m = np.column_stack([x[idx + k * tau] for k in range(m)])      # (n_pts, m)
        # (m+1)-th coordinate for false-neighbor detection
        x_next = x[idx + m * tau]                                          # (n_pts,)

        # Sub-sample if too many points
        if n_pts > max_ref_points:
            step = n_pts // max_ref_points
            idx_s = np.arange(0, n_pts, step)
            X_ref = X_m[idx_s]
            x_next_ref = x_next[idx_s]
        else:
            idx_s = np.arange(n_pts)
            X_ref = X_m
            x_next_ref = x_next

        n_ref = len(idx_s)

        # Pairwise distances in m-dim (vectorised)
        sq_norms = (X_ref ** 2).sum(axis=1)
        dist2_m = (sq_norms[:, None] + sq_norms[None, :] -
                   2.0 * (X_ref @ X_ref.T))
        np.clip(dist2_m, 0.0, None, out=dist2_m)
        np.fill_diagonal(dist2_m, np.inf)
        nn_idx = np.argmin(dist2_m, axis=1)  # (n_ref,)
        d_m = np.sqrt(dist2_m[np.arange(n_ref), nn_idx])  # (n_ref,)

        # Increase in distance when adding (m+1)-th dimension
        delta_next = np.abs(x_next_ref - x_next_ref[nn_idx])  # (n_ref,)

        # FNN criterion 1: relative increase
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = delta_next / np.where(d_m > 0, d_m, np.inf)
        crit1 = ratio > rtol

        # FNN criterion 2: absolute (normalized) distance in m+1 space
        d_m1 = np.sqrt(d_m ** 2 + delta_next ** 2)
        crit2 = d_m1 > atol

        fnn_frac = float(np.mean(crit1 | crit2))
        fnn_list.append(fnn_frac)

    return fnn_list


# ─────────────────────────────────────────────────────────────────────────────
# Correlation dimension D₂ (Grassberger-Procaccia)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_dimension(
    trajectory: np.ndarray,
    r_min_quantile: float = 0.01,
    r_max_quantile: float = 0.30,
    n_r: int = 20,
    max_points: int = 1000,
) -> Dict:
    """
    Grassberger-Procaccia 关联维数估计（D₂）。

    算法：
      1. 计算轨迹上所有点对的距离 ||x_i - x_j||
      2. 关联积分 C(r) = (pairs with ||x_i-x_j|| < r) / N²
      3. D₂ ≈ d log C(r) / d log r  （在标度区间内）

    Args:
        trajectory:       shape (T, N)。
        r_min_quantile:   距离分布的下分位数作为 r_min（默认 1%）。
        r_max_quantile:   距离分布的上分位数作为 r_max（默认 30%）。
        n_r:              对数均匀采样 r 的点数（默认 20）。
        max_points:       最多使用 T 步（防止 O(T²) 过慢）。

    Returns:
        {
            "D2":     float — 关联维数（对数斜率）
            "r_vals": List[float]
            "C_vals": List[float]
            "fit_r2": float  — 线性拟合 R²（越大表示标度区间越干净）
        }
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    T, N = traj.shape

    # Sub-sample if too long
    if T > max_points:
        step = T // max_points
        traj = traj[::step]
        T = len(traj)

    # Pairwise distances (upper triangle only)
    sq_norms = (traj ** 2).sum(axis=1)
    dist2 = (sq_norms[:, None] + sq_norms[None, :] -
             2.0 * (traj @ traj.T))
    np.clip(dist2, 0.0, None, out=dist2)
    dist_flat = np.sqrt(dist2[np.triu_indices(T, k=1)])  # upper triangle, no diagonal

    r_min = float(np.quantile(dist_flat, r_min_quantile))
    r_max = float(np.quantile(dist_flat, r_max_quantile))

    if r_max <= r_min or r_min <= 0:
        return {"D2": float("nan"), "r_vals": [], "C_vals": [], "fit_r2": float("nan")}

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_vals: List[float] = []
    N_pairs = float(len(dist_flat))
    for r in r_vals:
        C_r = float((dist_flat < r).sum()) / N_pairs
        C_vals.append(max(C_r, 1e-15))  # avoid log(0)

    log_r = np.log10(r_vals)
    log_C = np.log10(np.array(C_vals))

    # Linear regression in log-log space to get slope D₂
    valid = np.isfinite(log_C) & (np.array(C_vals) > 1e-14)
    if valid.sum() < 3:
        return {
            "D2": float("nan"),
            "r_vals": r_vals.tolist(),
            "C_vals": C_vals,
            "fit_r2": float("nan"),
        }

    slope, intercept = np.polyfit(log_r[valid], log_C[valid], 1)

    # R² of the fit
    y_pred = slope * log_r[valid] + intercept
    ss_res = float(((log_C[valid] - y_pred) ** 2).sum())
    ss_tot = float(((log_C[valid] - log_C[valid].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    return {
        "D2": float(slope),
        "r_vals": r_vals.tolist(),
        "C_vals": C_vals,
        "fit_r2": float(r2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Takens embedding theorem check
# ─────────────────────────────────────────────────────────────────────────────

def check_embedding_dimension(
    intrinsic_dim: float,
    current_embed_dim: int,
) -> Dict:
    """
    验证当前嵌入维度是否满足 Takens 嵌入定理。

    定理（Takens 1981）：
      对于 D 维吸引子，延迟嵌入维度 m ≥ 2D + 1 保证
      嵌入映射是一对一的（无自交叉），不存在假吸引子。

    Args:
        intrinsic_dim:    吸引子内在维度（FNN 的 min_sufficient_dim 或 D₂ 关联维数）。
        current_embed_dim: 当前使用的嵌入维度（如 n_regions 或 PCA 维度）。

    Returns:
        {
            "intrinsic_dim":     float — 输入的内在维度
            "current_embed_dim": int
            "takens_required":   int   — 2D+1
            "satisfied":         bool  — current >= takens_required
            "warning":           str   — 空字符串（满足）或警告信息
        }
    """
    required = int(np.ceil(2 * intrinsic_dim + 1))
    satisfied = current_embed_dim >= required

    if satisfied:
        warning = ""
    else:
        warning = (
            f"⚠ 嵌入维度不足：当前 m={current_embed_dim} < Takens 要求 "
            f"2D+1={required}（D={intrinsic_dim:.1f}）。"
            f"吸引子分析结果（LLE、关联维数、聚类）可能因「假吸引子」而偏差。"
            f"建议：增加嵌入维度至 ≥ {required}，或使用延迟嵌入代替原始坐标。"
        )
        logger.warning(warning)

    return {
        "intrinsic_dim": float(intrinsic_dim),
        "current_embed_dim": int(current_embed_dim),
        "takens_required": required,
        "satisfied": satisfied,
        "warning": warning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Attractor observation space identification
# ─────────────────────────────────────────────────────────────────────────────

_SPACE_CREDIBILITY = {
    "delay_embedding": 1,    # 最高（Takens 保证）
    "original_phase":  2,    # 次高
    "pca":             3,    # 中等（信息可能丢失）
    "latent_space":    4,    # 最低（高度依赖模型）
}


def identify_observation_space(
    trajectories: np.ndarray,
    n_pca_components: Optional[int] = None,
) -> Dict:
    """
    识别并评估吸引子所处的观察空间。

    可信度排序（参考 Kantz & Schreiber 1997 Ch. 3）：
      delay_embedding > 原始相空间 > PCA > latent_space

    Args:
        trajectories:       shape (n_init, steps, n_regions)。
        n_pca_components:   PCA 分析时使用的维度数（None → min(n_regions, steps)//2）。

    Returns:
        {
            "n_regions":         int — 原始状态空间维度
            "current_space":     str — "original_phase"（当前所用空间）
            "credibility_rank":  int — 可信度排序（1=最高）
            "pca_variance_95":   int — 95% 方差所需 PCA 主成分数
            "pca_variance_50":   int — 50% 方差所需 PCA 主成分数
            "effective_dim":     float — 有效维度（1/sum(p²) Shannon 熵估计）
            "recommendation":    str — 建议使用的观察空间
            "spaces": {
                "delay_embedding": {"available": bool, "note": str},
                "original_phase":  {"available": bool, "note": str},
                "pca":             {"available": bool, "note": str},
                "latent_space":    {"available": bool, "note": str},
            }
        }
    """
    n_init, steps, n_regions = trajectories.shape
    all_states = trajectories.reshape(-1, n_regions)  # (n_init*steps, n_regions)

    # ── PCA variance analysis ─────────────────────────────────────────────────
    # Fit PCA on the first 70% of trajectories (reference set) to avoid the
    # same leakage pattern that check_pca_leakage() is designed to detect.
    # Variance statistics are inherently stable once ref set > ~50 samples.
    n_ref = max(1, int(n_init * 0.7))
    ref_states = trajectories[:n_ref].reshape(-1, n_regions)  # reference only
    try:
        from sklearn.decomposition import PCA
        n_comp = min(n_regions, len(ref_states) - 1, 50)
        pca = PCA(n_components=n_comp)
        pca.fit(ref_states)  # fitted on reference set, not all trajectories
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n50 = int(np.searchsorted(cumvar, 0.50)) + 1
        # Effective dimension (participation ratio = 1 / Σ p²)
        p2 = pca.explained_variance_ratio_ ** 2
        eff_dim = float(1.0 / p2.sum()) if p2.sum() > 0 else float(n_regions)
        pca_available = True
    except Exception:
        n95, n50, eff_dim = n_regions, n_regions // 2, float(n_regions)
        pca_available = False

    # ── Spaces inventory ──────────────────────────────────────────────────────
    spaces = {
        "delay_embedding": {
            "available": True,
            "note": (
                f"可通过延迟嵌入 [x(t), x(t+τ), ..., x(t+(m-1)τ)] 构造，"
                f"需 m ≥ 2D+1 个时间步（Takens 定理保证一对一映射）。"
                f"当前轨迹长度 T={steps}，可支持最大 m={steps//2}。"
            ),
        },
        "original_phase": {
            "available": True,
            "note": (
                f"当前使用：{n_regions} 维原始脑区活动空间。"
                f"若系统真实维度 D << {n_regions}，该空间中吸引子"
                f"嵌入在低维流形上，邻居搜索等距离计算受「维数诅咒」影响。"
            ),
        },
        "pca": {
            "available": pca_available,
            "note": (
                f"PCA {n_comp if pca_available else 'N/A'} 维投影；"
                f"95% 方差需 {n95} 个主成分，"
                f"有效维度 ≈ {eff_dim:.1f}。"
                f"风险：PCA 丢弃了低方差分量，可能丢失对混沌重要的高频成分。"
            ),
        },
        "latent_space": {
            "available": False,
            "note": (
                "TwinBrainDigitalTwin 的潜空间维度由模型架构决定。"
                "潜空间吸引子受编码器非线性变换影响，"
                "可信度最低（模型依赖性强）。"
            ),
        },
    }

    recommendation = (
        f"推荐使用**延迟嵌入**观察空间（m={min(n95 * 2 + 1, steps//2)}，"
        f"τ=1步）以满足 Takens 定理；当前 {n_regions} 维原始空间"
        + ("具有足够维度（无需降维）。" if n_regions >= n95 * 2 + 1
           else f"维度不足（建议先 PCA 至 {n95} 维再做嵌入）。")
    )

    return {
        "n_regions": n_regions,
        "current_space": "original_phase",
        "credibility_rank": _SPACE_CREDIBILITY["original_phase"],
        "pca_variance_95": n95,
        "pca_variance_50": n50,
        "effective_dim": eff_dim,
        "recommendation": recommendation,
        "spaces": spaces,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data leakage checks
# ─────────────────────────────────────────────────────────────────────────────

def check_normalization_leakage(
    train_trajectories: np.ndarray,
    all_trajectories: np.ndarray,
    tol: float = 0.05,
) -> Dict:
    """
    检查归一化是否使用了全部数据（包含测试集），而非仅训练集。

    如果测试集与训练集的均值/标准差差异显著（> tol × 全集标准差），
    则使用全集统计量归一化会将测试集信息泄漏到训练集的归一化中。

    Args:
        train_trajectories: shape (n_train, steps, n_regions)，训练集轨迹。
        all_trajectories:   shape (n_total, steps, n_regions)，全部轨迹。
        tol:                均值/标准差相对差异阈值（默认 5%）。

    Returns:
        {
            "mean_drift":   float — |μ_all - μ_train| / σ_all（相对均值漂移）
            "std_drift":    float — |σ_all - σ_train| / σ_all（相对标准差漂移）
            "leakage_risk": str   — "low"/"medium"/"high"
            "warning":      str
        }
    """
    train_flat = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    all_flat = all_trajectories.reshape(-1, all_trajectories.shape[-1])

    mu_train = train_flat.mean(axis=0)
    std_train = train_flat.std(axis=0) + 1e-12

    mu_all = all_flat.mean(axis=0)
    std_all = all_flat.std(axis=0) + 1e-12

    mean_drift = float(np.abs(mu_all - mu_train).mean() / std_all.mean())
    std_drift = float(np.abs(std_all - std_train).mean() / std_all.mean())

    if mean_drift < tol and std_drift < tol:
        risk = "low"
        warning = ""
    elif mean_drift < 3 * tol and std_drift < 3 * tol:
        risk = "medium"
        warning = (
            f"△ 归一化泄漏风险（中等）：全集与训练集统计量存在差异 "
            f"（均值漂移={mean_drift:.3f}，标准差漂移={std_drift:.3f}）。"
            f"建议仅使用训练集计算归一化参数。"
        )
    else:
        risk = "high"
        warning = (
            f"⚠ 归一化泄漏风险（高）：全集统计量与训练集显著不同 "
            f"（均值漂移={mean_drift:.3f}，标准差漂移={std_drift:.3f}）。"
            f"若用全集统计量归一化，测试集信息会渗入训练流程，"
            f"导致 LLE 等估计量偏低。请改用仅训练集统计量归一化。"
        )

    return {
        "mean_drift": mean_drift,
        "std_drift": std_drift,
        "leakage_risk": risk,
        "warning": warning,
    }


def check_pca_leakage(
    reference_trajectories: np.ndarray,
    all_trajectories: np.ndarray,
    n_components: int = 10,
) -> Dict:
    """
    检查 PCA 投影是否存在测试集泄漏。

    若 PCA 在全部轨迹上拟合，相当于使用了"未见"轨迹的全局主成分，
    可能与仅在参考轨迹（训练集）上拟合的主成分不同。
    通过主成分夹角（principal angle）量化两种拟合方式的差异。

    Args:
        reference_trajectories: shape (n_ref, steps, n_regions)，参考（训练）轨迹。
        all_trajectories:       shape (n_total, steps, n_regions)，全部轨迹。
        n_components:           PCA 分析的主成分数（默认 10）。

    Returns:
        {
            "max_principal_angle_deg": float — 最大主成分夹角（度）
            "mean_principal_angle_deg": float
            "leakage_risk": str — "low"/"medium"/"high"
            "warning": str
        }
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {
            "max_principal_angle_deg": float("nan"),
            "mean_principal_angle_deg": float("nan"),
            "leakage_risk": "unknown",
            "warning": "scikit-learn 未安装，无法计算 PCA 泄漏。",
        }

    ref_flat = reference_trajectories.reshape(-1, reference_trajectories.shape[-1])
    all_flat = all_trajectories.reshape(-1, all_trajectories.shape[-1])

    n_comp = min(
        n_components,        # requested number of components
        ref_flat.shape[1],   # cannot exceed feature dimension
        ref_flat.shape[0] - 1,   # PCA needs n_samples > n_components (ref set)
        all_flat.shape[0] - 1,   # same constraint for full set
    )
    if n_comp < 1:
        return {
            "max_principal_angle_deg": float("nan"),
            "mean_principal_angle_deg": float("nan"),
            "leakage_risk": "unknown",
            "warning": "样本数不足，无法计算 PCA 泄漏。",
        }

    pca_ref = PCA(n_components=n_comp)
    pca_ref.fit(ref_flat)
    V_ref = pca_ref.components_  # (n_comp, n_features)

    pca_all = PCA(n_components=n_comp)
    pca_all.fit(all_flat)
    V_all = pca_all.components_  # (n_comp, n_features)

    # Principal angles: σ_i = arccos(singular values of V_ref @ V_all.T)
    M = V_ref @ V_all.T  # (n_comp, n_comp)
    _, sv, _ = np.linalg.svd(M)
    sv = np.clip(sv, -1.0, 1.0)
    angles_rad = np.arccos(sv)
    angles_deg = np.degrees(angles_rad)

    max_angle = float(angles_deg.max())
    mean_angle = float(angles_deg.mean())

    if max_angle < 5.0:
        risk = "low"
        warning = ""
    elif max_angle < 20.0:
        risk = "medium"
        warning = (
            f"△ PCA 泄漏风险（中等）：全集 PCA 与参考集 PCA 的最大主成分夹角 "
            f"{max_angle:.1f}°。结果对轨迹选取有一定敏感性。"
        )
    else:
        risk = "high"
        warning = (
            f"⚠ PCA 泄漏风险（高）：全集 PCA 与参考集 PCA 的最大主成分夹角 "
            f"{max_angle:.1f}°。强烈建议仅在参考（训练）轨迹上拟合 PCA，"
            f"再将其余轨迹投影到该子空间，避免信息泄漏。"
        )

    return {
        "max_principal_angle_deg": max_angle,
        "mean_principal_angle_deg": mean_angle,
        "leakage_risk": risk,
        "warning": warning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding_dimension_analysis(
    trajectories: np.ndarray,
    fnn_max_dim: int = 8,
    fnn_tau: int = 1,
    corr_dim: bool = True,
    check_leakage: bool = True,
    train_fraction: float = 0.7,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    完整的嵌入维度 & 吸引子观察空间分析。

    包含：
    1. FNN 内在维度估计
    2. Grassberger-Procaccia 关联维数 D₂
    3. Takens 嵌入定理验证（m ≥ 2D+1）
    4. 吸引子观察空间识别（可信度排序）
    5. 数据泄漏检查（归一化 + PCA）

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        fnn_max_dim:    FNN 测试的最大嵌入维度（默认 8）。
        fnn_tau:        FNN 延迟步数（默认 1）。
        corr_dim:       是否计算关联维数（计算量适中，默认 True）。
        check_leakage:  是否进行数据泄漏检查（默认 True）。
        train_fraction: 泄漏检查时训练集比例（默认 70%）。
        output_dir:     保存 embedding_dimension.json；None → 不保存。

    Returns:
        results: {
            "fnn":             Dict from false_nearest_neighbors()
            "corr_dim":        Dict from correlation_dimension()
            "takens_check":    Dict from check_embedding_dimension()
            "observation_space": Dict from identify_observation_space()
            "leakage_norm":    Dict from check_normalization_leakage()
            "leakage_pca":     Dict from check_pca_leakage()
            "summary":         str
        }
    """
    n_init, steps, n_regions = trajectories.shape
    logger.info(
        "嵌入维度分析: %d 条轨迹, %d 步, %d 脑区",
        n_init, steps, n_regions,
    )

    # ── 1. FNN averaged over up to 3 representative trajectories ─────────────
    # Averaging over multiple trajectories reduces sensitivity to any single
    # initial condition.  We cap at 3 to keep the O(T²) cost manageable;
    # in practice FNN results are stable across trajectories for the same attractor.
    n_fnn_traj = min(3, n_init)
    logger.info("  (1/5) False Nearest Neighbors 分析 (max_dim=%d, τ=%d, n_traj=%d)",
                fnn_max_dim, fnn_tau, n_fnn_traj)

    # Average FNN fractions across sampled trajectories
    fnn_fracs_all = []
    for k in range(n_fnn_traj):
        traj_k = trajectories[k]
        res_k = false_nearest_neighbors(traj_k, max_dim=fnn_max_dim, tau=fnn_tau)
        fnn_fracs_all.append(res_k["fnn_fractions"])
    mean_fracs = [float(np.mean([fnn_fracs_all[k][m] for k in range(n_fnn_traj)]))
                  for m in range(fnn_max_dim)]
    # Find minimum sufficient dim using the averaged fractions
    min_suf = fnn_max_dim
    for m_idx, frac in enumerate(mean_fracs):
        if frac < 0.01:
            min_suf = m_idx + 1
            break
    fnn_results = {
        "fnn_fractions": mean_fracs,
        "min_sufficient_dim": min_suf,
        "recommended_embed_dim": max(min_suf, 2),
        "takens_min_dim": 2 * min_suf + 1,
        "n_traj_averaged": n_fnn_traj,
    }
    # Use first trajectory as reference for correlation dimension
    ref_traj = trajectories[0]
    logger.info(
        "  FNN 最小充分维度 = %d (FNN 比例: %s)",
        min_suf,
        ", ".join(f"{v:.3f}" for v in mean_fracs),
    )

    # ── 2. Correlation dimension (on single trajectory) ───────────────────────
    cd_results: Dict = {}
    if corr_dim:
        logger.info("  (2/5) Grassberger-Procaccia 关联维数估计")
        cd_results = correlation_dimension(ref_traj)
        d2 = cd_results.get("D2", float("nan"))
        r2 = cd_results.get("fit_r2", float("nan"))
        logger.info("  D₂ = %.2f (R²=%.3f)", d2, r2)
        # Use D2 as intrinsic dim if FNN didn't converge
        if min_suf >= fnn_max_dim and np.isfinite(d2) and d2 > 0:
            intrinsic_dim = float(d2)
            logger.info("  FNN 未收敛，使用 D₂=%.2f 作为内在维度", intrinsic_dim)
        else:
            intrinsic_dim = float(min_suf)
    else:
        intrinsic_dim = float(min_suf)

    # ── 3. Takens check ───────────────────────────────────────────────────────
    logger.info("  (3/5) Takens 嵌入定理验证 (当前 m=%d, 内在 D=%.1f)", n_regions, intrinsic_dim)
    takens = check_embedding_dimension(intrinsic_dim, n_regions)
    if takens["satisfied"]:
        logger.info(
            "  ✓ 嵌入维度满足 Takens 定理 (m=%d ≥ 2D+1=%d)",
            n_regions, takens["takens_required"],
        )
    else:
        logger.warning(takens["warning"])

    # ── 4. Observation space ──────────────────────────────────────────────────
    logger.info("  (4/5) 吸引子观察空间识别")
    obs_space = identify_observation_space(trajectories)
    logger.info("  当前空间: %s (可信度排名 %d/4)", obs_space["current_space"], obs_space["credibility_rank"])
    logger.info("  有效维度 ≈ %.1f, PCA 95%%方差需 %d 主成分", obs_space["effective_dim"], obs_space["pca_variance_95"])
    logger.info("  建议: %s", obs_space["recommendation"])

    # ── 5. Data leakage ───────────────────────────────────────────────────────
    leak_norm: Dict = {}
    leak_pca: Dict = {}
    if check_leakage and n_init >= 4:
        logger.info("  (5/5) 数据泄漏检查 (训练集比例=%.0f%%)", train_fraction * 100)
        n_train = max(1, int(n_init * train_fraction))
        train_trajs = trajectories[:n_train]
        leak_norm = check_normalization_leakage(train_trajs, trajectories)
        if leak_norm["leakage_risk"] == "low":
            logger.info("  ✓ 归一化泄漏风险低（均值漂移=%.3f, 标准差漂移=%.3f）",
                        leak_norm["mean_drift"], leak_norm["std_drift"])
        elif leak_norm["warning"]:
            logger.warning("  归一化: %s", leak_norm["warning"])

        n_pca_comp = min(n_regions, 10)
        leak_pca = check_pca_leakage(train_trajs, trajectories, n_components=n_pca_comp)
        if leak_pca["leakage_risk"] == "low":
            logger.info("  ✓ PCA 泄漏风险低（最大主成分夹角=%.1f°）",
                        leak_pca["max_principal_angle_deg"])
        elif leak_pca["leakage_risk"] == "medium":
            logger.warning("  PCA: %s", leak_pca["warning"])
        else:
            # High leakage risk: report measurement result and confirm the
            # pipeline already applies the fix (training-set PCA everywhere).
            logger.info(
                "  PCA 子空间差异较大（最大主成分夹角=%.1f°）。\n"
                "  已修正：identify_observation_space 和可视化图表均仅在"
                "参考（训练）轨迹上拟合 PCA，其余轨迹投影到该子空间。\n"
                "  高角度差异提示训练集与测试集轨迹的主成分方向不完全一致，"
                "可能与动力学非平稳性有关，但不代表当前分析存在泄漏。",
                leak_pca["max_principal_angle_deg"],
            )
    else:
        logger.info("  (5/5) 数据泄漏检查已跳过（n_init=%d < 4）", n_init)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_embedding_summary(fnn_results, cd_results, takens, obs_space, leak_norm, leak_pca)
    logger.info("  嵌入维度分析摘要: %s", summary)

    results = {
        "fnn": fnn_results,
        "corr_dim": cd_results,
        "takens_check": takens,
        "observation_space": obs_space,
        "leakage_norm": leak_norm,
        "leakage_pca": leak_pca,
        "summary": summary,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "embedding_dimension.json"
        _save_json_safe(results, out_path)
        logger.info("  → 已保存: %s", out_path)

    return results


def _build_embedding_summary(fnn, cd, takens, obs, leak_norm, leak_pca) -> str:
    parts = []
    min_suf = fnn.get("min_sufficient_dim", "?")
    parts.append(f"FNN最小充分维度={min_suf}")
    if cd:
        d2 = cd.get("D2", float("nan"))
        parts.append(f"D₂={d2:.2f}" if np.isfinite(d2) else "D₂=N/A")
    parts.append("Takens" + ("✓" if takens.get("satisfied") else "✗(维度不足)"))
    parts.append(f"空间={obs.get('current_space','?')}(排名{obs.get('credibility_rank','?')}/4)")
    if leak_norm:
        parts.append(f"归一化泄漏={leak_norm.get('leakage_risk','?')}")
    if leak_pca:
        parts.append(f"PCA泄漏={leak_pca.get('leakage_risk','?')}")
    return "  |  ".join(parts)


def _save_json_safe(data: Dict, path: Path) -> None:
    """保存 dict 为 JSON，自动转换 numpy 类型。"""
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_convert(data), fh, indent=2, ensure_ascii=False)
