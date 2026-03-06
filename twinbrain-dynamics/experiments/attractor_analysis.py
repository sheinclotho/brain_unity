"""
Attractor Analysis
==================

检测模型是否存在 **有限数量吸引子状态**。

流程：
1. 从自由动力学轨迹中提取末端状态（tail_steps 步的均值）
2. 用 KMeans / DBSCAN 聚类，识别吸引子数量
3. 报告每个吸引子的吸引域分布（basin distribution）

输出：
  - outputs/attractor_states.npy    — 聚类中心
  - stdout / logger                 — basin distribution 报告
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_final_states(
    trajectories: np.ndarray,
    tail_steps: int = 100,
) -> np.ndarray:
    """
    从轨迹中提取末端状态作为吸引子候选。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        tail_steps:   取轨迹末尾多少步的均值（默认 100）。

    Returns:
        final_states: shape (n_init, n_regions)。
    """
    tail = trajectories[:, -tail_steps:, :]   # (n_init, tail_steps, n_regions)
    return tail.mean(axis=1)                   # (n_init, n_regions)


def run_attractor_analysis(
    trajectories: np.ndarray,
    tail_steps: int = 100,
    k_candidates: List[int] = (2, 3, 4, 5, 6),
    k_best: int = 3,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行吸引子分析。

    Args:
        trajectories:       shape (n_init, steps, n_regions)。
        tail_steps:         末端状态平均窗口。
        k_candidates:       KMeans 测试的 K 值列表（用 silhouette score 选最优）。
        k_best:             若无法自动选择，使用此默认 K。
        dbscan_eps:         DBSCAN epsilon 参数。
        dbscan_min_samples: DBSCAN min_samples 参数。
        output_dir:         保存 attractor_states.npy；None → 不保存。

    Returns:
        results: {
            "final_states":       np.ndarray (n_init, n_regions),
            "kmeans_labels":      np.ndarray (n_init,),
            "kmeans_centers":     np.ndarray (K, n_regions),
            "kmeans_k":           int,
            "basin_distribution": Dict[int, float],  # {cluster_id: fraction}
            "silhouette_score":   float | None,
            "dbscan_labels":      np.ndarray (n_init,),
            "dbscan_n_clusters":  int,
        }
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required. Install it with: pip install scikit-learn"
        )

    final_states = extract_final_states(trajectories, tail_steps=tail_steps)
    n_samples = final_states.shape[0]

    logger.info(
        "吸引子分析: n_samples=%d, n_regions=%d",
        n_samples,
        final_states.shape[1],
    )

    # ── KMeans with automatic K selection via silhouette score ────────────────
    best_k = k_best
    best_sil = -1.0
    best_km = None

    valid_k = [k for k in k_candidates if 2 <= k < n_samples]
    for k in valid_k:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(final_states)
        if len(np.unique(labels)) < 2:
            continue
        try:
            sil = silhouette_score(final_states, labels)
        except Exception:
            sil = -1.0
        logger.debug("  K=%d: silhouette=%.4f", k, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_km = km

    if best_km is None:
        # Fallback: use k_best without silhouette selection
        best_km = KMeans(n_clusters=min(k_best, n_samples), n_init=10, random_state=42)
        best_km.fit(final_states)
        best_sil = None

    kmeans_labels = best_km.labels_
    kmeans_centers = best_km.cluster_centers_

    # Basin distribution
    basin: Dict[int, float] = {}
    for cid in range(best_k):
        fraction = float((kmeans_labels == cid).sum()) / n_samples
        basin[cid] = round(fraction, 4)

    logger.info("  KMeans 最优 K=%d, silhouette=%.4f", best_k, best_sil or -1.0)
    _report_basin(basin)

    # ── DBSCAN (alternative / validation) ─────────────────────────────────────
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    db_labels = db.fit_predict(final_states)
    db_n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    logger.info("  DBSCAN 检测到 %d 个吸引子（噪声点排除后）", db_n_clusters)

    results = {
        "final_states": final_states,
        "kmeans_labels": kmeans_labels,
        "kmeans_centers": kmeans_centers,
        "kmeans_k": best_k,
        "basin_distribution": basin,
        "silhouette_score": best_sil,
        "dbscan_labels": db_labels,
        "dbscan_n_clusters": db_n_clusters,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "attractor_states.npy", kmeans_centers)
        logger.info("  → 已保存: %s/attractor_states.npy", output_dir)

    return results


# ── Private helpers ───────────────────────────────────────────────────────────

def _report_basin(basin: Dict[int, float]) -> None:
    """Pretty-print basin distribution to logger."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for cid, frac in sorted(basin.items()):
        label = labels[cid] if cid < len(labels) else str(cid)
        logger.info("  Attractor %s : %.1f%%", label, frac * 100)
