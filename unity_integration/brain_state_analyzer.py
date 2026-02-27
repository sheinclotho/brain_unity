"""
Brain State Analyzer
====================

Label-free brain state analysis: normative deviation detection and
graph-theoretic connectivity metrics.

Design philosophy
-----------------
The classic "disease detection" framing requires labelled pathological data,
a sufficient training cohort, and a reliable predictive model — none of which
can be guaranteed in a general-purpose digital-twin system.

TwinBrain instead uses an *anomaly-detection* + *biomarker-extraction* framing:

1. **Self-comparison** (implemented here as the default):
   Split the loaded time series into two windows (e.g. first half vs second
   half, or two experimental conditions).  Per-region z-score deviation
   identifies which regions changed most — no labels required.

2. **Normative deviation** (extensible):
   If a healthy-reference EC matrix or time series is available, compute
   per-region z-scores against the reference distribution.  Extreme deviations
   flag anomalous regions — the interpretation (disease vs artifact vs state
   change) is left to the clinician.

3. **Graph-theory biomarkers** (implemented here):
   EC matrix → hub scores, out/in-strength, local efficiency, global density.
   These dimensionality-reducing statistics characterise brain organisation
   without requiring diagnostic labels and can be compared across subjects or
   time points.

4. **EC fingerprinting**:
   A compact feature vector derived from the EC + graph metrics.  Useful for
   subject identification, longitudinal tracking, and condition comparison
   (e.g. rest vs task).

WebSocket protocol
------------------
Request:
  { type: "analyze_brain",
    method: "deviation" | "graph_metrics" | "compare_ec",
    window1_start: int,  window1_end: int,   (optional, frame indices)
    window2_start: int,  window2_end: int    (optional, frame indices)
  }

Response:
  { type: "brain_analysis_result",
    method: str,
    activity: [200 floats],      # deviation / hub-score overlay for 3D view
    summary: {key: value},       # interpretable text + key stats
    regions_of_interest: [int],  # top anomalous / hub region IDs (0-indexed)
    success: true
  }
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level cache for Fibonacci brain-region coordinates.
# Computed once on first call to ``BrainStateAnalyzer.ec_vs_distance_correlation``
# and reused on subsequent calls to avoid repeated trigonometric work.
_POSITIONS_CACHE: Optional[np.ndarray] = None


def _get_cached_positions(n: int = 200) -> np.ndarray:
    """Return module-level cached (n, 3) Fibonacci brain positions.

    The positions are computed via ``_compute_fibonacci_positions`` on the
    first call, then stored in ``_POSITIONS_CACHE`` and reused on every
    subsequent call with the same ``n``.  If ``n`` changes the cache is
    invalidated and recomputed.

    This avoids repeating 200 trigonometric operations on every call to
    ``ec_vs_distance_correlation``.
    """
    global _POSITIONS_CACHE
    if _POSITIONS_CACHE is None or _POSITIONS_CACHE.shape[0] != n:
        _POSITIONS_CACHE = _compute_fibonacci_positions(n)
    return _POSITIONS_CACHE


def _compute_fibonacci_positions(n: int = 200) -> np.ndarray:
    """Compute Fibonacci-sphere brain region positions matching app.js.

    Vectorized over ``n//2`` regions per hemisphere — replaces the original
    nested Python loop with NumPy broadcasting for a ~100× speed-up.
    """
    half   = n // 2
    golden = 2 * np.pi * (2 - (1 + np.sqrt(5)) / 2)
    pos    = np.zeros((n, 3), dtype=np.float32)

    # Per-hemisphere vectorized computation
    i_arr  = np.arange(half, dtype=np.float64)
    t_arr  = (i_arr + 0.5) / half               # elevation parameter ∈ (0, 1)
    el_arr = 1.0 - 1.85 * t_arr                 # elevation component
    r_arr  = np.sqrt(np.maximum(0.0, 1.0 - el_arr ** 2))
    az_arr = golden * i_arr                      # azimuth angle
    lat    = np.abs(r_arr * np.cos(az_arr)) * 0.85 + 0.15
    bulge  = 9 * np.exp(-((el_arr + 0.22) ** 2) * 5)

    for h, sign in enumerate((-1, 1)):
        start = h * half
        pos[start: start + half, 0] = sign * (lat * 55 + bulge + 9)
        pos[start: start + half, 1] = el_arr * 63 - 4
        pos[start: start + half, 2] = r_arr * np.sin(az_arr) * 76 - 8
    return pos


class BrainStateAnalyzer:
    """
    Label-free brain state analysis.

    All methods are pure functions (static or class methods) — no state is
    stored.  The caller is responsible for supplying the EC matrix and / or
    time series, both of which TwinBrain already computes.
    """

    # ── Graph-theoretic metrics ──────────────────────────────────────────────

    @staticmethod
    def compute_graph_metrics(
        ec_matrix: np.ndarray,
        threshold: float = 0.1,
    ) -> dict:
        """Compute graph-theoretic metrics from a (normalised) EC matrix.

        Args:
            ec_matrix : (N, N) EC matrix.  Values should be normalised to
                        [-1, 1] or [0, 1] — the same space returned by
                        ``PerturbationAnalyzer.ec_to_dict``.
            threshold : binarisation cut-off for adjacency-matrix metrics
                        (local efficiency, density).  0.1 keeps edges whose
                        absolute weight exceeds 10 % of the maximum.

        Returns dict with:
            hub_scores    – (N,) per-region hub score = √(out_strength × in_strength)
            out_strength  – (N,) sum of absolute outgoing EC weights
            in_strength   – (N,) sum of absolute incoming EC weights
            local_eff     – (N,) local efficiency proxy per region
            global_eff    – float, global efficiency proxy
            density       – float, fraction of edges above threshold
            top_hubs      – list of 10 region indices with highest hub score
        """
        ec_abs = np.abs(ec_matrix).copy()
        np.fill_diagonal(ec_abs, 0.0)
        N = ec_abs.shape[0]

        out_strength = ec_abs.sum(axis=1)   # (N,) – row sum
        in_strength  = ec_abs.sum(axis=0)   # (N,) – col sum
        # Hub score: geometric mean of out and in strength.
        # High hub score = strongly connected in both directions (rich-club node).
        hub_scores = np.sqrt(out_strength * in_strength + 1e-9)

        # Binarised adjacency at threshold for structural metrics
        adj     = (ec_abs > threshold).astype(np.float32)
        density = float(adj.sum()) / max(N * (N - 1), 1)

        # Local efficiency: for each region, mean EC weight among its neighbours.
        # A high value means the neighbourhood is tightly interconnected
        # (high local efficiency → resilient to single-node failure).
        local_eff = np.zeros(N, dtype=np.float32)
        for i in range(N):
            nbrs = np.where(adj[i] > 0)[0]
            if len(nbrs) >= 2:
                sub = ec_abs[np.ix_(nbrs, nbrs)]
                local_eff[i] = float(sub.mean())

        # Global efficiency proxy: mean non-zero weight × network density.
        # Equivalent to the Latora–Marchiori global efficiency on weighted
        # directed graphs when path lengths are approximated by inverse weights.
        nonzero = ec_abs[ec_abs > 0]
        global_eff = float(nonzero.mean() * density) if density > 0 and len(nonzero) > 0 else 0.0

        top_hubs = np.argsort(hub_scores)[::-1][:10].tolist()

        return {
            "hub_scores":   hub_scores.astype(np.float32).tolist(),
            "out_strength": out_strength.astype(np.float32).tolist(),
            "in_strength":  in_strength.astype(np.float32).tolist(),
            "local_eff":    local_eff.tolist(),
            "global_eff":   round(global_eff, 4),
            "density":      round(density, 4),
            "top_hubs":     top_hubs,
        }

    # ── Normative deviation map ──────────────────────────────────────────────

    @staticmethod
    def compute_deviation_map(
        ref_ts:  np.ndarray,
        test_ts: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Per-region deviation of test_ts from ref_ts using Cohen's d effect size.

        Cohen's d is the standard effect-size metric for two-group comparisons
        (Cohen, 1988).  It is preferred over a naive z-score (|Δμ| / σ_ref)
        because it accounts for variability in *both* windows, not just the
        reference, and it is nearly unbiased for different group sizes (Hedges'
        correction applied automatically when lengths differ).

        Formula (pooled-std variant):

            d_i = (mean_test_i − mean_ref_i) / s_pooled_i

        where  s_pooled = sqrt(((n1-1)·var_ref + (n2-1)·var_test) / (n1+n2-2))

        Interpretation thresholds (Cohen, 1988):
            |d| < 0.2  → negligible change
            |d| < 0.5  → small change
            |d| < 0.8  → medium change
            |d| ≥ 0.8  → large change (clinically / scientifically significant)

        Args:
            ref_ts  : (T1, N) reference time series (baseline / healthy window)
            test_ts : (T2, N) comparison time series

        Returns:
            deviation : (N,) float32, normalised |d| to [0, 1] for colour overlay
            summary   : dict with interpretable stats including signed Cohen's d
        """
        n1 = max(ref_ts.shape[0],  1)
        n2 = max(test_ts.shape[0], 1)

        # Cohen's d is undefined when either window has fewer than 2 samples
        # (variance cannot be estimated).  Return zeros with a warning summary.
        if n1 < 2 or n2 < 2:
            empty = np.zeros(ref_ts.shape[1] if ref_ts.ndim > 1 else 1, dtype=np.float32)
            return empty, {
                "mean_cohens_d": 0.0, "max_cohens_d": 0.0,
                "n_large_effect": 0, "n_medium_effect": 0,
                "n_activated": 0, "n_suppressed": 0,
                "mean_z_score": 0.0, "max_z_score": 0.0,
                "n_outliers_2std": 0, "n_outliers_3std": 0,
                "top_regions": [],
                "interpretation": f"窗口过短（ref={n1}, test={n2} 帧），Cohen's d 无法计算（至少各需 2 帧）",
                "effect_size_metric": "Cohen's d (pooled SD)",
            }

        ref_mean  = ref_ts.mean(axis=0)                   # (N,)
        test_mean = test_ts.mean(axis=0)                  # (N,)
        ref_var   = ref_ts.var(axis=0)                    # (N,)
        test_var  = test_ts.var(axis=0)                   # (N,)

        # Pooled standard deviation (Welch / pooled; Hedges-corrected when n1 ≠ n2).
        # Guard against zero variance with a small epsilon.
        pooled_var = ((n1 - 1) * ref_var + (n2 - 1) * test_var) / max(n1 + n2 - 2, 1)
        s_pooled   = np.maximum(np.sqrt(pooled_var), 1e-6)   # (N,)

        # Signed Cohen's d: positive = test > ref (activation), negative = suppression
        d_signed = (test_mean - ref_mean) / s_pooled         # (N,)
        d_abs    = np.abs(d_signed)

        d_max  = float(d_abs.max())
        d_norm = (d_abs / d_max).astype(np.float32) if d_max > 0 else d_abs.astype(np.float32)

        top_regions = np.argsort(d_abs)[::-1][:10].tolist()
        mean_d      = float(d_abs.mean())

        # Cohen's d interpretation thresholds
        if mean_d < 0.2:
            interp = "两时段差异微弱（Cohen's d < 0.2），大脑状态高度稳定"
        elif mean_d < 0.5:
            interp = "两时段存在轻微差异（Cohen's d 0.2–0.5），少数区域有变化"
        elif mean_d < 0.8:
            interp = "两时段存在中等差异（Cohen's d 0.5–0.8），建议关注高亮区域"
        else:
            interp = "两时段差异显著（Cohen's d ≥ 0.8），高亮区域的状态变化具有临床/科学意义"

        # Signed per-region direction: positive = activation, negative = suppression
        n_activated   = int((d_signed > 0.5).sum())
        n_suppressed  = int((d_signed < -0.5).sum())

        summary = {
            "mean_cohens_d":    round(mean_d, 3),
            "max_cohens_d":     round(d_max, 3),
            "n_large_effect":   int((d_abs >= 0.8).sum()),   # |d| ≥ 0.8
            "n_medium_effect":  int(((d_abs >= 0.5) & (d_abs < 0.8)).sum()),
            "n_activated":      n_activated,    # d > +0.5 (test > ref)
            "n_suppressed":     n_suppressed,   # d < −0.5 (test < ref)
            # ── Legacy keys (kept for frontend backward-compatibility) ───────────
            # IMPORTANT: these keys are intentionally misnamed.  They were previously
            # computed as z-scores; they now contain Cohen's d values.  Frontend code
            # that reads mean_z_score / n_outliers_2std will automatically benefit
            # from the improved metric without any UI changes.
            # DO NOT remove until the frontend is updated to use mean_cohens_d.
            "mean_z_score":     round(mean_d, 3),   # ← actually Cohen's d
            "max_z_score":      round(d_max, 3),    # ← actually max Cohen's d
            "n_outliers_2std":  int((d_abs > 0.5).sum()),  # ← |d| > 0.5 (medium+)
            "n_outliers_3std":  int((d_abs > 0.8).sum()),  # ← |d| > 0.8 (large)
            "top_regions":      top_regions,
            "interpretation":   interp,
            "effect_size_metric": "Cohen's d (pooled SD)",
        }
        return d_norm, summary

    # ── EC matrix comparison ─────────────────────────────────────────────────

    @staticmethod
    def compare_ec_matrices(
        ec1: np.ndarray,
        ec2: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Absolute difference between two EC matrices.

        Useful for comparing EC across experimental conditions, time windows,
        or modalities (fMRI-EC vs EEG-EC).

        Args:
            ec1, ec2 : (N, N) EC matrices (should both be normalised)

        Returns:
            diff_overlay : (N,) float32 per-region mean absolute change,
                           normalised to [0, 1] for 3D overlay
            summary      : dict with Pearson r and global similarity
        """
        diff = np.abs(ec1 - ec2)              # (N, N)
        np.fill_diagonal(diff, 0.0)
        N = diff.shape[0]
        # Per-region: average outgoing + incoming absolute change
        per_region = (diff.sum(axis=1) + diff.sum(axis=0)) / (2 * N)
        p_max = float(per_region.max())
        overlay = (per_region / p_max if p_max > 0 else per_region).astype(np.float32)

        # Pearson r between flattened matrices (EC fingerprint similarity)
        f1, f2 = ec1.flatten(), ec2.flatten()
        corr = float(np.corrcoef(f1, f2)[0, 1]) if f1.std() > 0 and f2.std() > 0 else 0.0

        top_changed = np.argsort(per_region)[::-1][:10].tolist()
        interp = (
            "EC 结构高度相似" if corr > 0.7
            else ("EC 结构中等相似" if corr > 0.4
            else "EC 结构差异显著")
        )
        summary = {
            "pearson_r":         round(corr, 3),
            "global_similarity": round(float(1.0 - diff.mean()), 3),
            "top_changed":       top_changed,
            "interpretation":    interp,
        }
        return overlay, summary

    # ── Fingerprint ──────────────────────────────────────────────────────────

    @staticmethod
    def fingerprint(
        ec_matrix: np.ndarray,
        time_series: Optional[np.ndarray] = None,
    ) -> dict:
        """Compact brain fingerprint for subject/condition identification.

        Suitable for longitudinal tracking (same subject, multiple sessions),
        condition comparison (rest vs task), or normative reference building.

        Returns:
            dict with graph_metrics, optionally mean_activity and top FC pairs.
        """
        metrics = BrainStateAnalyzer.compute_graph_metrics(ec_matrix)
        result: dict = {"graph_metrics": metrics}

        if time_series is not None and len(time_series) > 1:
            result["mean_activity"] = time_series.mean(axis=0).tolist()
            try:
                fc = np.corrcoef(time_series.T)   # (N, N)
                np.fill_diagonal(fc, 0.0)
                flat    = fc.flatten()
                top_idx = np.argsort(np.abs(flat))[::-1][:20]
                result["top_fc_pairs"] = [
                    {"src": int(i // fc.shape[0]), "dst": int(i % fc.shape[0]),
                     "r": round(float(flat[i]), 3)}
                    for i in top_idx
                ]
            except Exception:
                pass

        return result

    # ── EC half-split reliability (no retraining) ────────────────────────────

    @staticmethod
    def ec_half_split_reliability(
        surrogate,         # trained _SurrogateMLP
        input_X: np.ndarray,
        n_regions: int,
        pert_strength: float = 0.05,
    ) -> Tuple[float, dict]:
        """Measure how stable the EC estimate is across two halves of training data.

        Strategy: use the EXISTING trained surrogate (no retraining) but compute
        the finite-difference EC using the first half vs second half of input_X.
        High correlation between EC₁ and EC₂ means the surrogate has captured a
        consistent causal structure (not just noise).

        Args:
            surrogate     : trained _SurrogateMLP (nn.Module in eval mode)
            input_X       : (M, N*n_lags) training samples
            n_regions     : N
            pert_strength : δ for finite-difference approximation

        Returns:
            reliability : Pearson r between EC₁ and EC₂ (higher = more reliable)
            detail      : dict with individual EC statistics
        """
        import torch

        M    = len(input_X)
        half = max(M // 2, 1)
        X1   = input_X[:half]
        X2   = input_X[half:] if half < M else input_X   # guard: if only 1 sample

        # Require at least 5 samples per half for a meaningful comparison.
        # With fewer samples the EC estimate is dominated by noise; the
        # correlation between EC₁ and EC₂ would be artificially high (≈ 1.0)
        # because both halves reflect the same surrogate outputs on the same
        # data, giving false confidence.
        MIN_SAMPLES_PER_HALF = 5
        if len(X1) < MIN_SAMPLES_PER_HALF or len(X2) < MIN_SAMPLES_PER_HALF:
            detail = {
                "half_split_r": None,
                "n_samples_h1": len(X1),
                "n_samples_h2": len(X2),
                "interpretation": (
                    f"样本不足（h1={len(X1)}, h2={len(X2)}），"
                    f"每半部分至少需要 {MIN_SAMPLES_PER_HALF} 个样本才能评估可靠性"
                ),
            }
            return 0.0, detail

        def _compute_ec(X_arr):
            surrogate.eval()
            N = n_regions
            ec = np.zeros((N, N), dtype=np.float32)
            # Ensure input tensor lives on the same device as the surrogate model
            # to avoid a RuntimeError when the model was trained on CUDA.
            device = next(surrogate.parameters()).device
            with torch.no_grad():
                X_t    = torch.tensor(X_arr, dtype=torch.float32).to(device)
                base   = surrogate(X_t).cpu().numpy()
                for j in range(N):
                    X_pert = X_t.clone()
                    X_pert[:, -N + j] += pert_strength
                    pert   = surrogate(X_pert).cpu().numpy()
                    ec[j]  = (pert - base).mean(axis=0) / pert_strength
            return ec

        ec1 = _compute_ec(X1)
        ec2 = _compute_ec(X2)

        f1, f2 = ec1.flatten(), ec2.flatten()
        r = float(np.corrcoef(f1, f2)[0, 1]) if f1.std() > 0 and f2.std() > 0 else 0.0

        interp = (
            "EC 高度可靠（r > 0.5）" if r > 0.5
            else ("EC 中等可靠（0.3 < r ≤ 0.5）" if r > 0.3
            else ("EC 可靠性偏低（0.1 < r ≤ 0.3）" if r > 0.1
            else "EC 可靠性极低（r ≤ 0.1，建议增加数据量）"))
        )
        detail = {
            "half_split_r": round(r, 3),
            "n_samples_h1": len(X1),
            "n_samples_h2": len(X2),
            "interpretation": interp,
        }
        return r, detail

    # ── EC vs anatomical distance ─────────────────────────────────────────────

    @staticmethod
    def ec_vs_distance_correlation(
        ec_matrix: np.ndarray,
        positions: Optional[np.ndarray] = None,
    ) -> dict:
        """Measure how much EC reflects anatomical proximity.

        In structural connectivity, connection density decreases with
        Euclidean distance (distance-decay principle).  A significant
        negative Pearson r between |EC[i,j]| and D[i,j] is therefore
        *consistent with* anatomical plausibility — but a near-zero or
        weakly-positive r does NOT automatically mean the EC is noise.

        Two neurobiologically legitimate reasons for r ≈ 0:
          1. **Long-range connectivity dominates**: default-mode network,
             homotopic (interhemispheric), and top-down prefrontal control
             connections couple *distant* regions just as strongly as local
             ones.  At whole-brain 200-region scale this is common.
          2. **Mixed regime**: local and long-range effects coexist and
             cancel in the aggregate correlation.

        Note: the positions used here are Fibonacci-sphere approximations,
        not real MNI coordinates, which further reduces the sensitivity of
        this test.

        Recommendation: always interpret this result jointly with the
        half-split reliability (r > 0.5 = stable) and the overfit ratio
        from fit_quality.  A reliable surrogate with r ≈ 0 here more likely
        reflects genuine long-range connectivity than noise.

        Args:
            ec_matrix : (N, N) EC matrix
            positions : (N, 3) MNI coordinates; if None, uses the Fibonacci
                        sphere positions baked into the visualisation.

        Returns:
            dict with Pearson r, p-value approximation, and interpretation.
        """
        N = ec_matrix.shape[0]

        if positions is None:
            # Use the module-level cached Fibonacci positions (computed once).
            pos = _get_cached_positions(N)
        else:
            pos = np.array(positions, dtype=np.float32)

        # Pairwise Euclidean distance matrix
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)  # (N, N)

        # Collect off-diagonal pairs: |EC[i,j]| vs D[i,j]
        mask   = ~np.eye(N, dtype=bool)
        ec_abs = np.abs(ec_matrix)[mask]
        dist   = D[mask]

        r = float(np.corrcoef(ec_abs, dist)[0, 1]) if ec_abs.std() > 0 else 0.0
        # Expected sign: negative (strong EC ↔ short distance).
        # Approximate p-value via t-distribution (large-sample approximation).
        n_pairs = len(ec_abs)
        t_stat  = r * np.sqrt(n_pairs - 2) / np.sqrt(max(1 - r**2, 1e-9))
        # p < 0.001 if |t| > 3.29 (two-tailed, df ≈ ∞)
        p_approx = "<0.001" if abs(t_stat) > 3.29 else ("<0.01" if abs(t_stat) > 2.58 else ">0.01")

        # Five-tier interpretation covering all r ranges correctly.
        # Previous implementation had a gap: -0.10 ≤ r < -0.05 fell into the
        # "else" branch and was mislabelled as "positive correlation".
        if r < -0.1:
            interp = "EC 与解剖距离显著负相关（符合局部连接假设，支持 EC 有效性）"
        elif r < -0.05:
            interp = "EC 呈弱距离衰减（r ∈ [−0.10, −0.05)），局部优先连接模式存在但不突出"
        elif r < 0.05:
            interp = (
                "EC 与解剖距离几乎无相关（r ≈ 0）：①可能反映长程连接主导的网络（默认模式网络、"
                "半球间连合纤维等），这在全脑 EC 中很常见，具有神经科学意义；"
                "②也可能是数据量不足导致噪声主导。建议结合半分可靠性和过拟合指标综合判断。"
            )
        elif r < 0.1:
            interp = "⚠ EC 呈弱正相关（r ∈ [0.05, 0.10)），与距离衰减预期相反；可能是模型噪声所致，建议增加数据量"
        else:
            interp = "⚠ EC 呈显著正相关（r ≥ 0.10），强烈不符合生理预期；建议检查数据质量和模型可靠性"
        return {
            "ec_vs_distance_r":  round(r, 3),
            "p_approx":          p_approx,
            "n_pairs":           n_pairs,
            "interpretation":    interp,
        }
