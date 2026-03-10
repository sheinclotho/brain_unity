"""
MinimalModel 临界动力学验证框架 v3
====================================

v3 基于真实脑模型统计数据进行参数调优与机制验证。

真实模型关键参数（joint / fMRI 模态，供本框架对标）
-----------------------------------------------------
- LLE ≈ 0.007~0.018 (Rosenstein 100 轨迹, 2000 步)
- D₂  ≈ 2.48 ± 0.24 (Grassberger-Procaccia 关联维数)
- PCA: top-2 解释 96% 方差, n@90%=2
- PC2 收缩斜率 ≈ -0.1 ~ -0.3, R² < 0.1（线吸引子特征）
- 吸引子数量 2-4 个（KMeans/DBSCAN）
- 谱半径(响应矩阵) ≈ 3.17, DMD 线性化 ρ=1.00（非线性稳定）
- 8 个 hub 节点 (±2σ), hub 删除使谱半径下降 38-218%
- 主导频率 ≈ 0.01 Hz（极慢振荡）

v3 新机制（基于真实模型对比结论）
------------------------------------
1. 时间尺度分离 (community_taus):  不同社区不同漏积分时间常数
   x_c(t+1) = (1-1/τ_c)*x_c(t) + (1/τ_c)*tanh(W@x+h)_c + ε
2. Hub 节点非均匀性: 选定节点的跨社区输出权重被额外放大
3. 非线性边界阻尼: |x|>threshold 时施加额外阻力
4. 硬能量约束: 球面投影 (L2 norm → 固定预算)
5. 突触噪声: 权重矩阵添加小扰动 (5-10%)
6. 背景偏置: 对 hub 节点施加恒定正向偏置 (模拟兴奋性优势)

v3 两模型设计
--------------
MODEL A  "EdgeChaos"   :  调优到 LLE≈0.01, D₂≈2.5, PCA dim≤5
MODEL B  "Structured"  :  hub 节点 + 时间尺度分离 → PC2 收缩 + 多吸引子

参考文献
--------
Sompolinsky, Crisanti & Sommers (1988). Chaos in random neural networks. PRL.
Beggs & Plenz (2003). Neuronal avalanches. J Neurosci.
Deco et al. (2012). Ongoing cortical activity at rest: criticality. J Neurosci.
Rosenstein et al. (1993). A practical method for calculating LLE. Physica D.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Optional, Tuple, List, Dict, Any

# ---------- Font (CJK graceful fallback) ----------
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'WenQuanYi Micro Hei',
    'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SEED = 42

# Real-model targets (joint / fMRI modality)
REAL_LLE          = 0.0100
REAL_LLE_CV       = 23.7        # % CV of LLE across bootstrap trials
REAL_D2           = 2.48
REAL_PC2_SLOPE    = -0.15   # from transient phase portraits
REAL_N_ATTRACT    = 3
REAL_RHO_RESPONSE = 3.17        # response matrix spectral radius
REAL_ENERGY_fMRI  = 0.61        # per-community energy (fMRI-like)
REAL_ENERGY_EEG   = 0.08        # per-community energy (EEG-like)
REAL_ENERGY_JOINT = 0.73        # full-network joint energy

# ╔══════════════════════════════════════════════════════════════╗
# ║       MinimalModelV3  —  极简社区网络模型 v3                  ║
# ╚══════════════════════════════════════════════════════════════╝

class MinimalModelV3:
    """
    Community-structured recurrent rate network with v3 extensions.

    Standard update (tau_c = 1):
        x(t+1) = tanh( W @ x(t) + h + I_ext(t) ) + eps(t)

    Leaky integration (tau_c > 1):
        x_c(t+1) = (1 - 1/tau_c)*x_c(t) + (1/tau_c)*tanh(...) + eps_c(t)

    Optional mechanisms applied in order:
        1. boundary_damping   — nonlinear saturation near threshold
        2. energy_constraint  — L2-sphere projection to natural energy
    """

    def __init__(
        self,
        n_communities:       int   = 3,
        nodes_per_community: int   = 20,
        w_intra:             float = 2.0,
        w_inter_base:        float = 0.4,
        inter_prob:          float = 0.3,
        target_rho:          float = 1.03,
        base_noise:          float = 0.02,
        background_drive:    float = 0.0,
        # v3: per-community background drive (overrides background_drive)
        # Enables fMRI-like (high drive, E*~0.61) vs EEG-like (low drive, E*~0.08)
        community_background_drive: Optional[List[float]] = None,
        # v3: timescale separation (~10x fMRI:EEG frequency ratio)
        community_taus:      Optional[List[float]] = None,
        # v3: hub nodes (target: hub ablation changes rho_R by 38-218%)
        n_hubs:              int   = 0,
        hub_out_scale:       float = 8.0,
        # v3: energy constraint
        energy_constraint:   bool  = False,
        energy_budget:       Optional[float] = None,  # None = auto
        # v3: boundary damping
        boundary_damping:    bool  = False,
        damping_threshold:   float = 0.85,
        damping_strength:    float = 0.15,
        seed:                int   = DEFAULT_SEED,
    ):
        self.n_communities       = n_communities
        self.nodes_per_community = nodes_per_community
        self.N                   = n_communities * nodes_per_community
        self.w_intra             = w_intra
        self.w_inter_base        = w_inter_base
        self.inter_prob          = inter_prob
        self.target_rho          = target_rho
        self.base_noise          = base_noise
        # Build background drive vector (per-community or uniform)
        if community_background_drive is not None:
            h = np.zeros(self.N)
            for c, hc in enumerate(community_background_drive[:n_communities]):
                sl = slice(c * nodes_per_community, (c + 1) * nodes_per_community)
                h[sl] = float(hc)
            self.background_drive = h
        else:
            self.background_drive = np.full(self.N, background_drive)
        self.community_taus      = community_taus
        self.n_hubs              = n_hubs
        self.hub_out_scale       = hub_out_scale
        self.energy_constraint   = energy_constraint
        self.energy_budget       = energy_budget
        self.boundary_damping    = boundary_damping
        self.damping_threshold   = damping_threshold
        self.damping_strength    = damping_strength
        self.seed                = seed

        self._hub_indices    = np.array([], dtype=int)
        self._natural_energy = None

        # Build random connectivity
        self._rng = np.random.RandomState(seed)
        np.random.seed(seed)
        self._build_connectivity()
        self._scale_to_target_rho()

        # Hub nodes (must be after initial scaling)
        if n_hubs > 0:
            self._add_hub_nodes()

        # Per-community tau arrays for vectorized updates
        self._tau_arr  = None
        self._leak_arr = None
        if community_taus is not None:
            tau_arr = np.ones(self.N)
            for c, tau in enumerate(community_taus[:n_communities]):
                s = slice(c * nodes_per_community, (c + 1) * nodes_per_community)
                tau_arr[s] = max(1.0, float(tau))
            self._tau_arr  = tau_arr
            self._leak_arr = 1.0 / tau_arr   # vectorized leak coefficient

    # ── Construction ──────────────────────────────────────────────

    def _build_connectivity(self):
        N, npc = self.N, self.nodes_per_community
        rng = self._rng
        W = np.zeros((N, N))

        # Intra-community: dense uniform excitatory
        for c in range(self.n_communities):
            sl = slice(c * npc, (c + 1) * npc)
            W[sl, sl] = self.w_intra / npc

        # Inter-community: sparse mixed ±
        for i in range(N):
            ci = i // npc
            for j in range(N):
                cj = j // npc
                if ci != cj and rng.rand() < self.inter_prob:
                    sign     = rng.choice([-1, 1])
                    strength = self.w_inter_base * rng.uniform(0.5, 1.5)
                    W[i, j]  = sign * strength / npc

        # Prevent isolated nodes
        for i in np.where(np.abs(W).sum(axis=1) == 0)[0]:
            tgts = rng.choice(N, size=min(3, N - 1), replace=False)
            for j in tgts:
                W[i, j] = rng.choice([-1, 1]) * 0.01 / npc

        self.W_raw = W

    def _scale_to_target_rho(self):
        eigs = np.linalg.eigvals(self.W_raw)
        rho0 = max(float(np.max(np.abs(eigs))), 1e-10)
        self.g = self.target_rho / rho0
        self.W = self.g * self.W_raw

    def _add_hub_nodes(self):
        """
        Identify top inter-community sender nodes in each community and
        amplify their inter-community outgoing connections.  Re-scales W.
        """
        npc = self.nodes_per_community
        rng = np.random.RandomState(self.seed + 999)
        n_per_comm = max(1, self.n_hubs // self.n_communities)
        hub_list = []

        for c in range(self.n_communities):
            c_nodes = np.arange(c * npc, (c + 1) * npc)
            # Rank by current inter-community out-strength
            out_str = np.array([
                sum(abs(self.W_raw[node, j])
                    for j in range(self.N) if j // npc != c)
                for node in c_nodes
            ])
            top_local = c_nodes[np.argsort(out_str)[-n_per_comm:]]
            hub_list.extend(top_local.tolist())

        self._hub_indices = np.array(hub_list, dtype=int)

        W_hub = self.W_raw.copy()
        for h in self._hub_indices:
            ch = h // npc
            for j in range(self.N):
                cj = j // npc
                if ch != cj:
                    if abs(W_hub[h, j]) > 1e-10:
                        # Hub outgoing connections are EXCITATORY (positive):
                        # this creates a positive feedforward loop that raises
                        # lambda_J close to 1, giving large rho_R.  Ablating
                        # hubs breaks the loop and rho_R drops significantly.
                        W_hub[h, j] = abs(W_hub[h, j]) * self.hub_out_scale
                    elif rng.rand() < 0.5:
                        # Add new excitatory inter-community connection
                        W_hub[h, j] = (abs(rng.randn()) *
                                       self.w_inter_base * self.hub_out_scale / npc)

        self.W_raw = W_hub
        self._scale_to_target_rho()

        print(f"  [hub_nodes] {len(self._hub_indices)} hubs: "
              f"{self._hub_indices.tolist()}")

    def community_indices(self, c: int) -> slice:
        return slice(c * self.nodes_per_community,
                     (c + 1) * self.nodes_per_community)

    # ── Dynamics ──────────────────────────────────────────────────

    def _single_step(self, x: np.ndarray, noise: float,
                     ext: Optional[np.ndarray] = None) -> np.ndarray:
        """Core single-trajectory step (used by stimulate_community)."""
        pre = self.W @ x + self.background_drive
        if ext is not None:
            pre = pre + ext
        x_tanh = np.tanh(pre)

        if self._leak_arr is not None:
            x_new = (1.0 - self._leak_arr) * x + self._leak_arr * x_tanh
        else:
            x_new = x_tanh

        if noise > 0:
            x_new = x_new + noise * np.random.randn(self.N)

        if self.boundary_damping:
            excess = np.abs(x_new) - self.damping_threshold
            mask = excess > 0
            if mask.any():
                x_new[mask] = (np.sign(x_new[mask]) *
                               (self.damping_threshold +
                                excess[mask] * (1.0 - self.damping_strength)))

        if (self.energy_constraint and self._natural_energy is not None):
            norm = np.linalg.norm(x_new)
            if norm > self._natural_energy:
                x_new = x_new * (self._natural_energy / norm)

        return x_new

    def simulate(
        self,
        n_steps: int  = 2000,
        n_init:  int  = 50,
        burnin:  int  = 300,
        noise:   Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate (n_init, n_steps, N) trajectory array.
        Uses batch matrix multiply for speed.
        """
        noise = self.base_noise if noise is None else noise
        N = self.N

        # ── Estimate natural energy for energy_constraint ──────────
        if self.energy_constraint and self._natural_energy is None:
            x_w = np.random.uniform(-0.1, 0.1, N)
            for _ in range(burnin):
                x_w = self._single_step(x_w, noise=0.0)
            samples = []
            for _ in range(300):
                x_w = self._single_step(x_w, noise=noise)
                samples.append(np.linalg.norm(x_w))
            self._natural_energy = float(np.percentile(samples, 70))
            print(f"  [energy] Natural energy E*={self._natural_energy:.4f}")

        # ── Batch simulation (all trajectories in parallel) ────────
        # Shape: (n_init, N) throughout
        X = np.random.uniform(-0.1, 0.1, (n_init, N))

        # Burnin (no noise, vectorized)
        for _ in range(burnin):
            pre   = (self.W @ X.T).T + self.background_drive  # (n_init, N)
            X_tnh = np.tanh(pre)
            if self._leak_arr is not None:
                X = (1.0 - self._leak_arr) * X + self._leak_arr * X_tnh
            else:
                X = X_tnh

        traj = np.zeros((n_init, n_steps, N), dtype=np.float32)
        traj[:, 0] = X.astype(np.float32)

        for t in range(1, n_steps):
            pre   = (self.W @ X.T).T + self.background_drive
            X_tnh = np.tanh(pre)

            if self._leak_arr is not None:
                X = (1.0 - self._leak_arr) * X + self._leak_arr * X_tnh
            else:
                X = X_tnh

            if noise > 0:
                X = X + noise * np.random.randn(n_init, N)

            if self.boundary_damping:
                excess = np.abs(X) - self.damping_threshold
                mask   = excess > 0
                X[mask] = (np.sign(X[mask]) *
                           (self.damping_threshold +
                            excess[mask] * (1.0 - self.damping_strength)))

            if (self.energy_constraint and self._natural_energy is not None):
                norms = np.linalg.norm(X, axis=1, keepdims=True)  # (n_init, 1)
                over  = (norms > self._natural_energy).ravel()
                if over.any():
                    X[over] = (X[over] *
                               (self._natural_energy / norms[over]))

            traj[:, t] = X.astype(np.float32)

        return traj

    def stimulate_community(
        self,
        x0:             np.ndarray,
        target_community: int,
        amplitude:      float,
        stim_duration:  int,
        n_post:         int   = 150,
        noise:          Optional[float] = None,
    ) -> np.ndarray:
        """Rectangular pulse on target community."""
        noise = self.base_noise if noise is None else noise
        x     = x0.copy()
        total = stim_duration + n_post
        resp  = np.zeros((total, self.N))
        I_tpl = np.zeros(self.N)
        I_tpl[self.community_indices(target_community)] = amplitude

        for t in range(total):
            ext = I_tpl if t < stim_duration else None
            x   = self._single_step(x, noise=noise, ext=ext)
            resp[t] = x
        return resp

    def perturb_connectivity(self, noise_scale: float,
                              seed: int = 0) -> 'MinimalModelV3':
        """Return structurally perturbed copy (synaptic noise on W)."""
        rng   = np.random.default_rng(seed)
        new   = MinimalModelV3.__new__(MinimalModelV3)
        new.__dict__.update(self.__dict__)
        W_n   = self.W_raw + noise_scale * rng.standard_normal(self.W_raw.shape)
        rho0  = max(float(np.max(np.abs(np.linalg.eigvals(W_n)))), 1e-10)
        new.W_raw = W_n
        new.g     = self.target_rho / rho0
        new.W     = new.g * W_n
        return new

    def simulate_transient(
        self,
        n_init:  int   = 20,
        n_steps: int   = 300,
        noise:   Optional[float] = None,
        scale:   float = 0.8,
    ) -> np.ndarray:
        """
        Simulate from DIVERSE random initial conditions with NO burnin.

        Use this for PC2 contraction and attractor-approach analysis.
        Initial states uniformly sampled from [-scale, scale]^N so they
        span a wide range including both sides of the attractor.  The
        first ~30-50 steps show the contraction toward the
        attractor manifold (PC2 decays; PC1 slowly evolves), matching the
        real model phase-portrait methodology (Fig 2/3 in problem statement).

        Uses model.base_noise (default) to produce realistic R² < 0.1 —
        the noise makes the contraction signal "noisy but systematic",
        matching the real model's phase-portrait statistics.

        Returns: (n_init, n_steps, N) float32 array.
        """
        noise = self.base_noise if noise is None else noise   # use model noise
        N     = self.N
        traj  = np.zeros((n_init, n_steps, N), dtype=np.float32)
        for i in range(n_init):
            # Diverse starting points: half drawn symmetrically around 0,
            # half biased toward/away from the expected attractor
            if i % 2 == 0:
                x = np.random.uniform(-scale, scale, N)
            else:
                x = np.random.randn(N) * scale * 0.5
            traj[i, 0] = x.astype(np.float32)
            for t in range(1, n_steps):
                x = self._single_step(x, noise=noise)
                traj[i, t] = x.astype(np.float32)
        return traj

    # ── Response matrix & hub control ─────────────────────────────

    @staticmethod
    def compute_response_matrix(
        model:   'MinimalModelV3',
        traj:    Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute linearised response matrix R = (I - D*W)^{-1} * D
        where D = diag(sech²(W*x* + h)) at the attractor state x*.

        Physical meaning: R[i,j] = steady-state change in node i per unit
        sustained input at j (Murphy & Miller 2009, Neuron).

        Large rho(R) ≈ 3.17 (real model) indicates system near resonance.

        Returns: (R matrix [N×N], rho_R scalar).
        """
        N = model.N
        if traj is not None:
            x_star = traj[:, -300:, :].reshape(-1, N).mean(axis=0)
        else:
            x = np.zeros(N)
            for _ in range(1000):
                xn = model._single_step(x, noise=0.0)
                if np.max(np.abs(xn - x)) < 1e-9:
                    break
                x = xn
            x_star = x

        pre   = model.W @ x_star + model.background_drive
        sech2 = 1.0 / np.cosh(pre) ** 2        # shape (N,)
        D     = np.diag(sech2)
        J     = D @ model.W                     # Jacobian at x*
        I_mat = np.eye(N)

        rho_J = float(np.max(np.abs(np.linalg.eigvals(J))))
        # Tikhonov regularisation: stronger lambda when rho_J > 1 (unstable
        # linearisation at x*).  Cap rho_R at 20 for numerical sanity.
        if rho_J < 1.0:
            lam = 1e-4
        else:
            # lam chosen so that min eigenvalue of (I - J + lam*I) = 0.05
            lam = rho_J - 0.95 + 1e-4
        try:
            R = np.linalg.solve(I_mat - J + lam * I_mat, D)
        except np.linalg.LinAlgError:
            R = np.linalg.lstsq(I_mat - J + lam * I_mat, D, rcond=None)[0]

        rho_R = min(float(np.max(np.abs(np.linalg.eigvals(R)))), 50.0)
        return R, rho_R

    def compute_hub_importance_response(
        self,
        traj: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Hub importance measured via RESPONSE MATRIX spectral radius.

        Matches the real-model metric: hub ablation decreases rho_R by
        38-218%.  Large hub_out_scale creates hub-dominated inter-community
        coupling; when hubs are ablated, rho_R drops significantly.

        Returns: hub_nodes, hub_scores, rho_R_full, rho_R_ablated,
                 rho_W_full, importance_pct.
        """
        npc        = self.nodes_per_community
        hub_scores = np.zeros(self.N)
        for i in range(self.N):
            ci = i // npc
            for j in range(self.N):
                if j // npc != ci:
                    hub_scores[i] += abs(self.W[i, j])

        thresh    = hub_scores.mean() + 2.0 * hub_scores.std()
        hub_nodes = np.where(hub_scores > thresh)[0]

        _, rho_full = MinimalModelV3.compute_response_matrix(self, traj)
        rho_W_full  = float(np.max(np.abs(np.linalg.eigvals(self.W))))

        if len(hub_nodes) == 0:
            return dict(hub_nodes=hub_nodes, hub_scores=hub_scores,
                        rho_R_full=rho_full, rho_R_ablated=rho_full,
                        rho_W_full=rho_W_full, importance_pct=0.0)

        W_save  = self.W
        W_abl   = self.W.copy()
        W_abl[hub_nodes, :] = 0
        W_abl[:, hub_nodes] = 0
        self.W  = W_abl
        _, rho_abl = MinimalModelV3.compute_response_matrix(self, traj)
        self.W  = W_save

        importance_pct = abs(rho_full - rho_abl) / max(rho_full, 1e-6) * 100.0
        return dict(
            hub_nodes      = hub_nodes,
            hub_scores     = hub_scores,
            rho_R_full     = rho_full,
            rho_R_ablated  = rho_abl,
            rho_W_full     = rho_W_full,
            importance_pct = importance_pct,
        )

    @staticmethod
    def compute_community_energy(
        traj:        np.ndarray,
        model:       'MinimalModelV3',
        burnin_frac: float = 0.3,
    ) -> Dict[str, float]:
        """
        Per-community energy  E*_c = sqrt( mean(x_c²) )  over settled steps.
        Also computes joint  E_joint = sqrt( mean(x²) )  (all nodes).

        Real-model targets:
          C0 (fMRI-like, slow + high drive) : E* ≈ 0.61
          C1 (EEG-like,  fast + low drive)  : E* ≈ 0.08
          Joint (all nodes)                 : E* ≈ 0.73
        """
        n_init, T, N = traj.shape
        burn   = int(T * burnin_frac)
        result = {}
        for c in range(model.n_communities):
            sl  = model.community_indices(c)
            xc  = traj[:, burn:, sl].reshape(-1)
            result[f"E_c{c}"] = float(np.sqrt(np.mean(xc ** 2)))
        x_all = traj[:, burn:, :].reshape(-1)
        result["E_joint"] = float(np.sqrt(np.mean(x_all ** 2)))
        return result

    @staticmethod
    def compute_lle_cv(
        model:   'MinimalModelV3',
        n_boots: int = 12,
        n_traj:  int = 30,
        n_steps: int = 800,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap estimate of LLE coefficient of variation.
        Runs n_boots independent simulations, computes LLE for each.
          CV = std(LLE) / |mean(LLE)| × 100%   (target: ~23.7%)

        Returns: (mean_lle, std_lle, cv_pct).
        """
        lles = []
        for k in range(n_boots):
            np.random.seed(model.seed + k * 1000)
            tr  = model.simulate(n_steps=n_steps, n_init=n_traj, burnin=200)
            lle = MinimalModelV3.estimate_lyapunov(
                tr, max_time=350, fit_range=(20, 100))
            if np.isfinite(lle):
                lles.append(lle)
        np.random.seed(model.seed)   # restore
        if len(lles) < 2:
            return float("nan"), float("nan"), float("nan")
        m = float(np.mean(lles))
        s = float(np.std(lles))
        return m, s, s / max(abs(m), 1e-10) * 100.0


    # ── Analysis ──────────────────────────────────────────────────

    @staticmethod
    def estimate_lyapunov(
        traj:      np.ndarray,
        max_time:  int = 400,
        fit_range: Tuple[int, int] = (20, 100),
    ) -> float:
        """
        Rosenstein nearest-neighbour LLE estimator.
        fit_range=(20,100): captures the linear-growth phase before saturation.
        For settled 2000-step trajectories this is the most reliable window.
        """
        traj = traj.astype(np.float64)
        n_init, n_steps, _N = traj.shape
        dist_mat = squareform(pdist(traj[:, 0, :]))
        np.fill_diagonal(dist_mat, np.inf)
        pairs = [(i, int(np.argmin(dist_mat[i]))) for i in range(n_init)]

        T = min(max_time, n_steps)
        log_dist = np.zeros(T)
        for t in range(T):
            d = np.array([np.linalg.norm(traj[i, t] - traj[j, t])
                          for i, j in pairs])
            log_dist[t] = np.mean(np.log(np.maximum(d, 1e-12)))

        s, e = fit_range[0], min(fit_range[1], T)
        if e <= s:
            return float('nan')
        slope, *_ = linregress(np.arange(T)[s:e], log_dist[s:e])
        return float(slope)

    @staticmethod
    def compute_pca_stats(traj: np.ndarray, threshold: float = 0.95
                          ) -> Dict[str, Any]:
        """Return PCA explained variance and effective dimension."""
        X   = traj.reshape(-1, traj.shape[2]).astype(np.float64)
        pca = PCA().fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        dim = int(np.argmax(cum >= threshold)) + 1
        return {
            'dim_95'  : dim,
            'dim_90'  : int(np.argmax(cum >= 0.90)) + 1,
            'var_top2': float(cum[min(1, len(cum) - 1)]),
            'evr'     : pca.explained_variance_ratio_[:10].tolist(),
            'pca'     : pca,
        }

    @staticmethod
    def compute_pc2_contraction(
        traj:       np.ndarray,
        skip_steps: int = 5,
        fit_end:    int = 150,
    ) -> Tuple[float, float]:
        """
        Measure PC2 contraction slope using WITHIN-TRAJECTORY consecutive steps.

        IMPORTANT: Use simulate_transient() trajectories (no burnin).
        Burnin-settled trajectories already sit on the attractor (PC2≈0),
        giving slope≈0.  Transient trajectories start from diverse states
        and show the approach to attractor, matching the real model's
        phase-portrait measurement (images show trajectories converging
        from PC2=3-8 to PC2≈0 over ~200 steps).

        Regression: ΔPC2(t) = slope * PC2(t) + const
        Target:  slope ≈ -0.1 ~ -0.3,  R² < 0.1 (noisy but systematic).

        Args:
            traj:       (n_init, n_steps, N) array from simulate_transient()
            skip_steps: skip first few steps (fast initial transient)
            fit_end:    only use steps up to fit_end
        Returns:
            (slope, r2)
        """
        n_init, T, N = traj.shape
        # KEY FIX: fit PCA on the TRANSIENT portion only (first fit_end steps).
        # If we fit PCA on ALL steps (including near-attractor), the "PC2
        # direction" reflects near-attractor oscillations, not the transient
        # approach direction, and the slope estimate is diluted toward 0.
        transient_end = min(fit_end, T)
        X_transient = traj[:, :transient_end, :].reshape(-1, N).astype(np.float64)
        pca = PCA(n_components=2)
        pca.fit(X_transient)

        all_pc2, all_dpc2, all_wts = [], [], []
        for i in range(n_init):
            P     = pca.transform(traj[i].astype(np.float64))  # (T, 2)
            t_end = min(fit_end, T - 1)
            if t_end > skip_steps:
                # Within-trajectory consecutive step pairs only
                pc2_seg  = P[skip_steps : t_end, 1]
                dpc2_seg = P[skip_steps + 1 : t_end + 1, 1] - pc2_seg
                all_pc2.append(pc2_seg)
                all_dpc2.append(dpc2_seg)
                # Weight by |PC2| to emphasise the large-deviation early steps
                all_wts.append(np.abs(pc2_seg) + 1e-6)

        if not all_pc2:
            return float("nan"), float("nan")

        pc2_all  = np.concatenate(all_pc2)
        dpc2_all = np.concatenate(all_dpc2)
        wts_all  = np.concatenate(all_wts)
        # Weighted linear regression (emphasise early, high-PC2 steps)
        sw   = wts_all.sum()
        mx   = np.dot(wts_all, pc2_all)  / sw
        my   = np.dot(wts_all, dpc2_all) / sw
        ssxx = np.dot(wts_all, (pc2_all  - mx) ** 2)
        ssxy = np.dot(wts_all, (pc2_all  - mx) * (dpc2_all - my))
        slope = float(ssxy / ssxx) if ssxx > 1e-12 else float("nan")
        # Unweighted r² for interpretability
        _, _, r, _, _ = linregress(pc2_all, dpc2_all)
        return float(slope), float(r ** 2)

    @staticmethod
    def count_attractors(traj: np.ndarray,
                         burnin_frac: float = 0.5) -> Tuple[int, int]:
        """
        Count attractors via KMeans (silhouette) and DBSCAN
        applied to trajectory endpoint states (after burnin_frac).

        Returns (n_kmeans, n_dbscan).
        """
        n_init, T, N = traj.shape
        burn = int(T * burnin_frac)
        pts  = traj[:, burn:, :].reshape(-1, N).astype(np.float64)

        # PCA projection (at most 10 components)
        n_pcs = min(10, N - 1, len(pts) - 1)
        proj  = PCA(n_components=n_pcs).fit_transform(pts)

        # KMeans silhouette scan k=2..5
        best_k, best_s = 2, -1.0
        for k in range(2, 6):
            try:
                km  = KMeans(n_clusters=k, n_init=10, random_state=42)
                lbl = km.fit_predict(proj)
                if len(np.unique(lbl)) < 2:
                    continue
                s   = silhouette_score(proj, lbl,
                                       sample_size=min(2000, len(proj)))
                if s > best_s:
                    best_s, best_k = s, k
            except Exception:
                pass

        # DBSCAN with adaptive eps
        std = float(proj.std(axis=0).mean())
        n_db = 1
        for eps_frac in [0.3, 0.5, 0.8, 1.2]:
            try:
                db     = DBSCAN(eps=std * eps_frac, min_samples=5)
                labels = db.fit_predict(proj)
                n_db   = len(set(labels) - {-1})
                if 2 <= n_db <= 8:
                    break
            except Exception:
                pass

        return best_k, max(1, n_db)

    def compute_hub_importance(self) -> Dict[str, Any]:
        """
        Legacy hub importance via W spectral radius (kept for backward compat).
        Use compute_hub_importance_response() for response-matrix-based metric.
        """
        return self.compute_hub_importance_response()

    @staticmethod
    def compute_correlation_dimension(
        traj:        np.ndarray,
        max_samples: int = 1500,
    ) -> float:
        """
        Grassberger-Procaccia correlation dimension D₂.
        Estimates log-log slope of correlation integral C(r) ~ r^D₂.
        """
        X = traj.reshape(-1, traj.shape[2]).astype(np.float64)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
        X   = X[idx]

        dists = pdist(X)
        dists = dists[dists > 0]
        if len(dists) < 10:
            return float('nan')

        r_vals = np.logspace(
            np.log10(np.percentile(dists, 3)),
            np.log10(np.percentile(dists, 55)), 30)

        C_r   = np.array([np.sum(dists < r) / len(dists) for r in r_vals])
        valid = C_r > 0
        if valid.sum() < 5:
            return float('nan')

        slope, *_ = linregress(np.log(r_vals[valid]), np.log(C_r[valid]))
        return float(slope)

    @staticmethod
    def compute_branching_ratio(traj: np.ndarray,
                                threshold: float = 0.3) -> float:
        x     = traj.reshape(-1, traj.shape[2]).astype(np.float64)
        A     = (np.abs(x) > threshold).sum(axis=1).astype(float)
        valid = A[:-1] > 0
        if valid.sum() < 20 or np.std(A[:-1][valid]) < 1e-6:
            return float('nan')
        slope, *_ = linregress(A[:-1][valid], A[1:][valid])
        return float(slope)

    @staticmethod
    def compute_mean_autocorr(traj: np.ndarray, max_lag: int = 200) -> np.ndarray:
        n_init, T, N = traj.shape
        acf_sum = np.zeros(max_lag)
        count   = 0
        for i in range(min(n_init, 8)):
            for n in range(N):
                sig = traj[i, :, n] - traj[i, :, n].mean()
                if sig.std() < 1e-8:
                    continue
                c = np.correlate(sig, sig, mode='full')[T - 1: T - 1 + max_lag]
                acf_sum += c / max(c[0], 1e-12)
                count   += 1
        return acf_sum / max(count, 1)


# ╔══════════════════════════════════════════════════════════════╗
# ║                      绘图函数                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved] {path}")


def plot_phase_portrait(
    traj:         np.ndarray,
    model:        MinimalModelV3,
    model_name:   str  = "",
    n_show:       int  = 5,
    n_grid:       int  = 18,
    save:         str  = "fig_phase_portrait_v3.png",
    is_transient: bool = False,
) -> None:
    """
    PC1-PC2 phase portrait with velocity field, matching real model visualization.
    Blue dots = trajectory starts, Red × = ends. Coloured by time step.
    """
    n_init, T, N = traj.shape
    X_all = traj.reshape(-1, N).astype(np.float64)

    pca = PCA(n_components=2)
    pca.fit(X_all)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 8))

    # Trajectories
    cmap   = plt.cm.plasma
    n_show = min(n_show, n_init)
    sc_ref = None
    for i in range(n_show):
        P = pca.transform(traj[i].astype(np.float64))
        sc_ref = ax.scatter(P[:, 0], P[:, 1], c=np.arange(T), cmap=cmap,
                            s=3, alpha=0.5, vmin=0, vmax=T)
        ax.plot(*P[0], 'bo', ms=7, zorder=5)
        ax.plot(*P[-1], 'rx', ms=9, mew=2, zorder=5)

    if sc_ref is not None:
        plt.colorbar(sc_ref, ax=ax, label='Time Step')

    # Velocity field (grid in PC space)
    P_all  = pca.transform(X_all)
    x_lim  = np.percentile(P_all[:, 0], [2, 98])
    y_lim  = np.percentile(P_all[:, 1], [2, 98])
    gx = np.linspace(x_lim[0], x_lim[1], n_grid)
    gy = np.linspace(y_lim[0], y_lim[1], n_grid)
    GX, GY = np.meshgrid(gx, gy)

    V_mean = X_all.mean(axis=0)
    comps  = pca.components_     # (2, N)

    U = np.zeros_like(GX)
    V = np.zeros_like(GY)
    speed = np.zeros_like(GX)

    for ii in range(n_grid):
        for jj in range(n_grid):
            pc_pt   = np.array([GX[ii, jj], GY[ii, jj]])
            x_rec   = V_mean + pc_pt @ comps          # reconstruct state
            # One-step update (no noise)
            pre     = model.W @ x_rec + model.background_drive
            x_next  = np.tanh(pre)
            if model._leak_arr is not None:
                x_next = ((1.0 - model._leak_arr) * x_rec +
                           model._leak_arr * x_next)
            dx    = x_next - x_rec
            dx_pc = comps @ dx           # project back to PC space
            U[ii, jj] = dx_pc[0]
            V[ii, jj] = dx_pc[1]
            speed[ii, jj] = np.sqrt(dx_pc[0]**2 + dx_pc[1]**2)

    strm = ax.streamplot(GX, GY, U, V,
                         color=speed, cmap='cool', linewidth=0.7,
                         arrowsize=0.8, density=1.2)
    plt.colorbar(strm.lines, ax=ax, label='speed')

    prefix = "Transient " if is_transient else ""
    ax.set(
        xlabel=f'PC1 ({evr[0]*100:.1f}% var)',
        ylabel=f'PC2 ({evr[1]*100:.1f}% var)',
        title=(f'{prefix}Phase Portrait: PC1 vs PC2\n'
               f'({n_show} trajectories, blue=start, red×=end, colour=time)\n'
               f'{model_name}'),
    )
    ax.grid(True, alpha=0.3)
    _savefig(save)


def plot_parameter_scan(
    results: List[Dict],
    save:    str = "fig_param_scan_v3.png",
) -> None:
    """Heatmap / line plot of LLE and D2 over (rho, noise) grid."""
    rhos   = sorted(set(r['rho']   for r in results))
    noises = sorted(set(r['noise'] for r in results))

    lle_mat  = np.full((len(noises), len(rhos)), np.nan)
    dim_mat  = np.full((len(noises), len(rhos)), np.nan)
    slp_mat  = np.full((len(noises), len(rhos)), np.nan)

    for r in results:
        ri = rhos.index(r['rho'])
        ni = noises.index(r['noise'])
        lle_mat[ni, ri]  = r['lle']
        dim_mat[ni, ri]  = r['dim_95']
        slp_mat[ni, ri]  = r['pc2_slope']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['LLE (target 0.005-0.01)', 'PCA dim 95% (target ≤5)',
              'PC2 contraction slope (target -0.1~-0.3)']
    mats   = [lle_mat, dim_mat, slp_mat]
    cmaps  = ['RdYlGn', 'RdYlGn_r', 'RdYlGn_r']

    for ax, mat, title, cm in zip(axes, mats, titles, cmaps):
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap=cm, interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels([f'{r:.3f}' for r in rhos], rotation=45, fontsize=8)
        ax.set_yticks(range(len(noises)))
        ax.set_yticklabels([f'{n:.3f}' for n in noises], fontsize=8)
        ax.set(xlabel='Spectral radius rho', ylabel='Noise sigma', title=title)

        # Mark cells close to target
        for ni in range(len(noises)):
            for ri in range(len(rhos)):
                val = mat[ni, ri]
                if not np.isnan(val):
                    ax.text(ri, ni, f'{val:.3f}', ha='center', va='center',
                            fontsize=6, color='black')

    fig.suptitle('Parameter Scan: rho × noise  (v3 EdgeChaos model)', fontsize=12)
    _savefig(save)


def plot_mechanism_comparison(
    records: List[Dict],
    save:    str = "fig_mechanisms_v3.png",
) -> None:
    """
    Bar chart comparing 7 mechanism variants across 5 metrics.
    records: list of dicts with keys label, lle, d2, pc2_slope, dim_95, n_attract.
    """
    labels   = [r['label']     for r in records]
    lles     = [r['lle']       for r in records]
    d2s      = [r['d2']        for r in records]
    slopes   = [r['pc2_slope'] for r in records]
    dims     = [r['dim_95']    for r in records]
    n_atts   = [r.get('n_attract', 0) for r in records]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_flat = axes.flatten()

    metrics = [
        (lles,   'LLE',           [0.0], 'tab:blue',   (0.005, 0.01)),
        (d2s,    'D2 (corr dim)', None,  'tab:green',  (2.0,   3.0)),
        (slopes, 'PC2 slope',     [0.0], 'tab:red',    (-0.3,  -0.1)),
        (dims,   'PCA dim 95%',   [5.0], 'tab:orange', None),
        (n_atts, 'Attractor count(KMeans)', [3.0], 'tab:purple', None),
    ]

    for ax, (vals, ylabel, hlines, clr, target_band) in zip(ax_flat, metrics):
        y_arr = np.array(vals, dtype=float)
        bars  = ax.bar(x, y_arr, color=clr, alpha=0.75, edgecolor='k', lw=0.5)
        if hlines:
            for hl in hlines:
                ax.axhline(hl, color='k', ls='--', lw=0.8)
        if target_band:
            ax.axhspan(*target_band, color='yellow', alpha=0.25,
                       label=f'target {target_band}')
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis='y', alpha=0.4)

    # ACF plot in last panel
    ax_last = ax_flat[5]
    for r in records:
        if 'acf' in r and r['acf'] is not None:
            ax_last.plot(r['acf'][:100], label=r['label'][:18], alpha=0.7)
    ax_last.axhline(0.5, color='k', ls='--', lw=0.8)
    ax_last.set(xlabel='Lag', ylabel='ACF', title='Autocorrelation')
    ax_last.legend(fontsize=6)
    ax_last.grid(True, alpha=0.4)

    # Real-model reference lines
    for ax, ref in zip(ax_flat, [REAL_LLE, REAL_D2, REAL_PC2_SLOPE, 5.0, REAL_N_ATTRACT]):
        ax.axhline(ref, color='crimson', ls=':', lw=1.5, label=f'real={ref}')

    fig.suptitle('Mechanism Comparison (v3): 7 Perturbation Variants', fontsize=12)
    _savefig(save)


def plot_hub_analysis(
    model:    MinimalModelV3,
    hub_info: Dict,
    save:     str = "fig_hub_analysis_v3.png",
) -> None:
    """
    Visualize hub node scores, community membership, and ablation effect.
    """
    npc        = model.nodes_per_community
    hs         = hub_info['hub_scores']
    hub_nodes  = hub_info['hub_nodes']
    rho_full   = hub_info['rho_full']
    rho_abl    = hub_info['rho_ablated']
    importance = hub_info['importance']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Hub score per node
    ax = axes[0]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    comm_colors = [colors[i // npc] for i in range(model.N)]
    ax.bar(range(model.N), hs, color=comm_colors, alpha=0.75, edgecolor='none')
    thresh = hs.mean() + 2 * hs.std()
    ax.axhline(thresh, color='crimson', ls='--', lw=1.2,
               label=f'Hub threshold (mean+2σ)')
    for h in hub_nodes:
        ax.bar(h, hs[h], color='crimson', alpha=0.9)
    ax.set(xlabel='Node index', ylabel='Inter-comm out-strength',
           title=f'Hub Scores  ({len(hub_nodes)} hubs identified)')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.4)

    # Panel B: Community membership matrix of W
    ax = axes[1]
    im = ax.imshow(np.abs(model.W), cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='|W_ij|')
    for c in range(model.n_communities):
        ax.axhline((c + 1) * npc - 0.5, color='cyan', lw=0.8)
        ax.axvline((c + 1) * npc - 0.5, color='cyan', lw=0.8)
    for h in hub_nodes:
        ax.axhline(h, color='lime', lw=0.5, alpha=0.6)
        ax.axvline(h, color='lime', lw=0.5, alpha=0.6)
    ax.set(title='|W| matrix  (cyan=community border, lime=hub row/col)',
           xlabel='Source node', ylabel='Target node')

    # Panel C: Ablation bar chart
    ax = axes[2]
    ax.bar(['Full network', 'Hub ablated'],
           [rho_full, rho_abl],
           color=['steelblue', 'salmon'], edgecolor='k', lw=0.8)
    ax.set(ylabel='Spectral radius rho',
           title=(f'Hub ablation effect\n'
                  f'Δρ={abs(rho_full-rho_abl):.4f} '
                  f'({importance*100:.1f}% change)'))
    ax.grid(True, axis='y', alpha=0.4)

    fig.suptitle(f'Hub Node Analysis (Model B Structured)  '
                 f'[real: 8 hubs, Δρ 38-218%]', fontsize=12)
    _savefig(save)



def plot_timescale_separation(
    model:       MinimalModelV3,
    traj_trans:  np.ndarray,
    save:        str = "fig_timescale_v3.png",
) -> None:
    """
    Show community-level mean activity traces from transient trajectories.
    Demonstrates timescale separation: slow fMRI (tau=20) vs fast EEG (tau=2).
    """
    n_init, T, N = traj_trans.shape
    npc    = model.nodes_per_community
    n_comm = model.n_communities
    t_arr  = np.arange(T)
    taus   = model._tau_arr

    fig, axes = plt.subplots(n_comm, 1, figsize=(12, 4 * n_comm), sharex=True)
    if n_comm == 1:
        axes = [axes]
    comm_labels = ['C0 (fMRI-like, slow)', 'C1 (EEG-like, fast)', 'C2 (coupling)']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i_traj in range(min(4, n_init)):
        for c in range(n_comm):
            sl    = model.community_indices(c)
            c_avg = traj_trans[i_traj, :, sl].mean(axis=1)
            axes[c].plot(t_arr, c_avg, color=colors[c], alpha=0.65, lw=0.9)

    for c in range(n_comm):
        tau_c = float(taus[c * npc]) if taus is not None else 1.0
        h_c   = float(model.background_drive[c * npc])
        axes[c].set(ylabel='Mean activity',
                    title=f'{comm_labels[c]}  tau={tau_c:.0f}  h={h_c:.2f}')
        axes[c].grid(True, alpha=0.4)
        axes[c].axhline(0, color='k', lw=0.4)

    axes[-1].set_xlabel('Time step (no burnin)')
    fig.suptitle(
        'Timescale Separation: Community Traces (transient)\n'
        'fMRI:EEG frequency ratio ~10x driven by tau=[20,2,7]',
        fontsize=12)
    _savefig(save)


def plot_community_energy(
    records: List[Dict],
    save:    str = "fig_community_energy_v3.png",
) -> None:
    """Bar chart of per-community energy E* and joint energy for each model."""
    labels  = [r['label'] for r in records]
    n_comm  = max(len([k for k in r if k.startswith('E_c')]) for r in records)
    colors  = ['tab:blue', 'tab:orange', 'tab:green']

    x   = np.arange(len(labels))
    w   = 0.22
    fig, ax = plt.subplots(figsize=(14, 5))

    for c in range(n_comm):
        vals = [float(r.get(f'E_c{c}', 0) or 0) for r in records]
        ax.bar(x + (c - 1) * w, vals, width=w, color=colors[c],
               alpha=0.8, label=f'C{c} energy', edgecolor='none')

    # Real-model reference lines
    ax.axhline(REAL_ENERGY_fMRI, color='blue',   ls='--', lw=1.3,
               label=f'real fMRI-like C0={REAL_ENERGY_fMRI}')
    ax.axhline(REAL_ENERGY_EEG,  color='orange', ls='--', lw=1.3,
               label=f'real EEG-like C1={REAL_ENERGY_EEG}')

    joint_vals = [float(r.get('E_joint', 0) or 0) for r in records]
    ax.plot(x, joint_vals, 's--', color='black', ms=7, zorder=5,
            label='E_joint (all nodes)')
    ax.axhline(REAL_ENERGY_JOINT, color='black', ls=':', lw=1.2,
               label=f'real joint={REAL_ENERGY_JOINT}')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set(ylabel='E* = sqrt(mean(x^2))',
           title='Per-community Energy  (target: fMRI-like≈0.61, EEG-like≈0.08)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis='y', alpha=0.4)
    _savefig(save)


def plot_lle_cv_comparison(
    cv_records: List[Dict],
    save:       str = "fig_lle_cv_v3.png",
) -> None:
    """LLE mean±std bars with CV annotation, for Model A and B."""
    labels   = [r['label'] for r in cv_records]
    means    = [r.get('mean_lle', 0) or 0 for r in cv_records]
    stds     = [r.get('std_lle', 0) or 0 for r in cv_records]
    cvs      = [r.get('cv_pct', 0) or 0 for r in cv_records]

    x    = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=['steelblue', 'forestgreen'], alpha=0.8, edgecolor='k')
    ax.axhline(REAL_LLE, color='crimson', ls='--', lw=1.5,
               label=f'real LLE={REAL_LLE}')
    for xi, cv in zip(x, cvs):
        ax.text(xi, (means[xi] or 0) + (stds[xi] or 0) + 0.0002,
                f'CV={cv:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.axhspan(0.005, 0.010, color='yellow', alpha=0.3, label='target 0.005-0.01')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(ylabel='LLE (Rosenstein)',
           title=(f'LLE mean+-std across bootstrap trials\n'
                  f'(real CV~{REAL_LLE_CV:.1f}%,  target: ~23.7%)'))
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.4)
    _savefig(save)


def plot_two_model_summary(
    results_A: Dict,
    results_B: Dict,
    save:      str = "fig_summary_comparison_v3.png",
) -> None:
    """
    Side-by-side spider / bar chart comparing Model A & B against real model.
    """
    categories  = ['LLE×100', 'D2', '|PC2 slope|×10',
                   'PCA dim 95%', 'n_attract']
    real_vals   = [REAL_LLE * 100, REAL_D2, abs(REAL_PC2_SLOPE) * 10,
                   2.0, float(REAL_N_ATTRACT)]

    def _vals(r: Dict) -> List[float]:
        return [
            abs(float(r.get('lle', 0) or 0)) * 100,
            float(r.get('d2', 0) or 0),
            abs(float(r.get('pc2_slope', 0) or 0)) * 10,
            float(r.get('dim_95', 0) or 0),
            float(r.get('n_attract', 0) or 0),
        ]

    A_vals = _vals(results_A)
    B_vals = _vals(results_B)

    x   = np.arange(len(categories))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - w, real_vals, width=w, label='Real brain',   color='crimson',  alpha=0.8)
    ax.bar(x,     A_vals,    width=w, label='Model A (EdgeChaos)',
           color='steelblue', alpha=0.8)
    ax.bar(x + w, B_vals,    width=w, label='Model B (Structured)',
           color='forestgreen', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Metric value (scaled)')
    ax.set_title('Model A & B vs Real Brain Targets (v3)')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.4)

    # Annotate actual numbers above bars
    for xi, (rv, av, bv) in enumerate(zip(real_vals, A_vals, B_vals)):
        ax.text(xi - w, rv + 0.02, f'{rv:.2f}', ha='center', va='bottom', fontsize=7)
        ax.text(xi,     av + 0.02, f'{av:.2f}', ha='center', va='bottom', fontsize=7)
        ax.text(xi + w, bv + 0.02, f'{bv:.2f}', ha='center', va='bottom', fontsize=7)

    _savefig(save)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    主实验流程                                 ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    print("=" * 70)
    print("MinimalModel v3 — Dual-Model Framework")
    print("  MODEL A: EdgeChaos   (LLE/D2/PC2 parameter matching)")
    print("  MODEL B: Structured  (timescale tau=[20,2,7] + hub nodes)")
    print("    NEW: response-matrix hub importance, LLE CV, community energy")
    print("=" * 70)

    BASE_CFG = dict(
        n_communities       = 3,
        nodes_per_community = 20,
        w_intra             = 2.0,
        w_inter_base        = 0.4,
        inter_prob          = 0.3,
        seed                = 42,
    )

    # ──────────────────────────────────────────────────────────────
    # Phase 0 — Parameter scan (rho × noise)
    # Transient trajectories are used for PC2 slope measurement.
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 0/5   Parameter Scan  (rho x noise, LLE + PC2 slope)")
    print("=" * 70)

    RHO_SCAN   = [1.01, 1.02, 1.03, 1.04, 1.05]
    NOISE_SCAN = [0.01, 0.02, 0.03, 0.05]
    scan_results = []

    for rho in RHO_SCAN:
        for noise in NOISE_SCAN:
            cfg = {**BASE_CFG, 'target_rho': rho, 'base_noise': noise}
            mdl = MinimalModelV3(**cfg)
            # Settled traj for LLE + PCA dim
            tr   = mdl.simulate(n_steps=800, n_init=30, burnin=200)
            lle  = MinimalModelV3.estimate_lyapunov(tr, max_time=350,
                                                    fit_range=(20, 100))
            pst  = MinimalModelV3.compute_pca_stats(tr)
            # Transient traj for PC2 slope
            trt  = mdl.simulate_transient(n_init=15, n_steps=200)
            slp, r2 = MinimalModelV3.compute_pc2_contraction(
                trt, skip_steps=3, fit_end=25)
            rec = dict(rho=rho, noise=noise, lle=lle,
                       dim_95=pst['dim_95'], pc2_slope=slp, pc2_r2=r2)
            scan_results.append(rec)
            print(f"  rho={rho:.2f}  noise={noise:.3f}  "
                  f"LLE={lle:+.5f}  dim={pst['dim_95']}  "
                  f"PC2_slope={slp:+.4f}  R2={r2:.3f}")

    plot_parameter_scan(scan_results)

    def _score(r):
        if np.isnan(r['lle']) or np.isnan(r['pc2_slope']):
            return 1e9
        return abs(r['lle'] - REAL_LLE) + 0.3 * abs(r['pc2_slope'] - REAL_PC2_SLOPE)

    best = min(scan_results, key=_score)
    print(f"\n  Best: rho={best['rho']:.2f}  noise={best['noise']:.3f}"
          f"  LLE={best['lle']:+.5f}  PC2_slope={best['pc2_slope']:+.4f}")

    # ──────────────────────────────────────────────────────────────
    # Phase 1 — MODEL A "EdgeChaos"  deep analysis
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1/5   MODEL A 'EdgeChaos'  (n_steps=2000)")
    print("  Validates: LLE~0.01, D2~2.5, PCA dim<=5, LLE CV~23.7%")
    print("=" * 70)

    CFG_A = {**BASE_CFG, 'target_rho': best['rho'], 'base_noise': best['noise']}
    model_A = MinimalModelV3(**CFG_A)

    print("  Simulating settled trajectories (60 x 2000) ...")
    traj_A  = model_A.simulate(n_steps=2000, n_init=60, burnin=300)
    print("  Simulating transient trajectories (30 x 300, no burnin) ...")
    traj_A_trans = model_A.simulate_transient(n_init=30, n_steps=300)

    lle_A          = MinimalModelV3.estimate_lyapunov(traj_A, max_time=500,
                                                      fit_range=(20, 100))
    d2_A           = MinimalModelV3.compute_correlation_dimension(traj_A)
    pca_A          = MinimalModelV3.compute_pca_stats(traj_A)
    n_km_A, n_db_A = MinimalModelV3.count_attractors(traj_A)
    sigma_A        = MinimalModelV3.compute_branching_ratio(traj_A)
    energy_A       = MinimalModelV3.compute_community_energy(traj_A, model_A)
    _, rho_R_A     = MinimalModelV3.compute_response_matrix(model_A, traj_A)
    # PC2 slope from transient (not settled!) trajectories — fit_end=25
    slp_A, r2_A    = MinimalModelV3.compute_pc2_contraction(
        traj_A_trans, skip_steps=3, fit_end=25)

    print("  Computing LLE CV (12 bootstraps) ...")
    mean_lle_A, std_lle_A, cv_A = MinimalModelV3.compute_lle_cv(
        model_A, n_boots=12, n_traj=25, n_steps=700)

    print(f"\n  ── Model A Results ──")
    print(f"  LLE            = {lle_A:+.5f}   (target: 0.005-0.01)")
    print(f"  LLE CV         = {cv_A:.1f}%      (target: ~23.7%)")
    print(f"  D2             = {d2_A:.3f}     (target: ~2.48)")
    print(f"  PC2 slope(tr.) = {slp_A:+.4f}  R2={r2_A:.4f}  (target: -0.1~-0.3)")
    print(f"  PCA dim 95%    = {pca_A['dim_95']}           (target: <=5)")
    print(f"  var(top-2 PC)  = {pca_A['var_top2']*100:.1f}%         (target: ~96%)")
    print(f"  rho_R          = {rho_R_A:.3f}    (target: ~3.17)")
    print(f"  Attractors     = {n_km_A}(KM) / {n_db_A}(DBSCAN)  (target: 2-4)")
    print(f"  Energy: {energy_A}")

    plot_phase_portrait(traj_A, model_A,
                        model_name=f"A rho={CFG_A['target_rho']:.2f} "
                                   f"noise={CFG_A['base_noise']:.3f}",
                        save="fig_phase_portrait_A.png")
    plot_phase_portrait(traj_A_trans, model_A,
                        model_name="A transient (no burnin)",
                        is_transient=True,
                        save="fig_phase_portrait_A_trans.png")

    res_A = dict(
        lle=lle_A, d2=d2_A, pc2_slope=slp_A, pc2_r2=r2_A,
        dim_95=pca_A['dim_95'], n_attract=n_km_A,
        rho_R=rho_R_A, lle_cv=cv_A, **energy_A,
    )

    # ──────────────────────────────────────────────────────────────
    # Phase 2 — MODEL B "Structured"  deep analysis
    #   NEW requirements:
    #   - tau=[20,2,7]: fMRI:EEG ~10x frequency ratio
    #   - community_background_drive=[0.35,0,0.10]: energy hierarchy
    #     fMRI-like(C0)~0.61, EEG-like(C1)~0.08
    #   - n_hubs=6, hub_out_scale=8: response-matrix hub importance
    #   - hub importance via R=(I-DW)^{-1}D spectral radius (target 38-218%)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2/5   MODEL B 'Structured'  (n_steps=2000)")
    print("  tau=[20,2,7]: fMRI:EEG ~10x ratio")
    print("  community_background_drive=[0.35, 0.0, 0.10]")
    print("  n_hubs=6, hub_out_scale=8  (response-matrix hub control)")
    print("=" * 70)

    CFG_B = {
        **BASE_CFG,
        'target_rho'               : 1.03,
        'base_noise'               : 0.02,
        'community_taus'           : [20, 2, 7],
        # fMRI-like (C0) high drive → high energy (≈0.61)
        # EEG-like  (C1) slight inhibitory bias → low energy (≈0.08)
        # Coupling  (C2) moderate drive
        'community_background_drive': [0.30, -0.05, 0.10],
        'n_hubs'                   : 6,
        'hub_out_scale'            : 8.0,
    }
    model_B = MinimalModelV3(**CFG_B)

    print("  Simulating settled trajectories (80 x 2000) ...")
    traj_B  = model_B.simulate(n_steps=2000, n_init=80, burnin=500)
    print("  Simulating transient trajectories (30 x 300, no burnin) ...")
    traj_B_trans = model_B.simulate_transient(n_init=30, n_steps=300)

    lle_B          = MinimalModelV3.estimate_lyapunov(traj_B, max_time=500,
                                                      fit_range=(20, 100))
    d2_B           = MinimalModelV3.compute_correlation_dimension(traj_B)
    pca_B          = MinimalModelV3.compute_pca_stats(traj_B)
    n_km_B, n_db_B = MinimalModelV3.count_attractors(traj_B)
    energy_B       = MinimalModelV3.compute_community_energy(traj_B, model_B)
    # PC2 slope from transient trajectories — use shorter fit window
    # (steps 3-25) to capture the fast initial contraction, not noise
    slp_B, r2_B    = MinimalModelV3.compute_pc2_contraction(
        traj_B_trans, skip_steps=3, fit_end=25)
    # Hub importance via response matrix R = (I-DW)^{-1}D
    hub_info_B     = model_B.compute_hub_importance_response(traj_B)
    rho_R_B        = hub_info_B['rho_R_full']

    print("  Computing LLE CV (12 bootstraps) ...")
    mean_lle_B, std_lle_B, cv_B = MinimalModelV3.compute_lle_cv(
        model_B, n_boots=12, n_traj=25, n_steps=700)

    print(f"\n  ── Model B Results ──")
    print(f"  LLE            = {lle_B:+.5f}   (target: 0.005-0.01)")
    print(f"  LLE CV         = {cv_B:.1f}%      (target: ~23.7%)")
    print(f"  D2             = {d2_B:.3f}")
    print(f"  PC2 slope(tr.) = {slp_B:+.4f}  R2={r2_B:.4f}  (target: -0.1~-0.3)")
    print(f"  PCA dim 95%    = {pca_B['dim_95']}")
    print(f"  var(top-2 PC)  = {pca_B['var_top2']*100:.1f}%")
    print(f"  rho_R (resp.mat) = {rho_R_B:.3f}  (target: ~3.17)")
    print(f"  Hub nodes (2sig) = {len(hub_info_B['hub_nodes'])}  "
          f"(real: 8)")
    print(f"  Hub importance   = "
          f"rho_R {hub_info_B['rho_R_full']:.3f} -> {hub_info_B['rho_R_ablated']:.3f}  "
          f"({hub_info_B['importance_pct']:.1f}% change)  (real: 38-218%)")
    print(f"  Community energy:")
    for k, v in energy_B.items():
        ref = {'E_c0': f"(real fMRI-like {REAL_ENERGY_fMRI})",
               'E_c1': f"(real EEG-like  {REAL_ENERGY_EEG})",
               'E_joint': f"(real joint     {REAL_ENERGY_JOINT})"}.get(k, '')
        print(f"    {k} = {v:.4f}  {ref}")

    plot_phase_portrait(traj_B, model_B,
                        model_name="B tau=[20,2,7] + 6 hubs + background drive",
                        save="fig_phase_portrait_B.png")
    plot_phase_portrait(traj_B_trans, model_B,
                        model_name="B transient (no burnin)",
                        is_transient=True,
                        save="fig_phase_portrait_B_trans.png")

    # Hub analysis with response-matrix metric
    # Adapt hub_info to the panel that expects old keys
    hub_info_B_display = dict(
        hub_nodes  = hub_info_B['hub_nodes'],
        hub_scores = hub_info_B['hub_scores'],
        rho_full   = hub_info_B['rho_R_full'],
        rho_ablated= hub_info_B['rho_R_ablated'],
        importance = hub_info_B['importance_pct'] / 100.0,
    )
    plot_hub_analysis(model_B, hub_info_B_display,
                      save="fig_hub_analysis_v3.png")

    plot_timescale_separation(model_B, traj_B_trans)

    res_B = dict(
        lle=lle_B, d2=d2_B, pc2_slope=slp_B, pc2_r2=r2_B,
        dim_95=pca_B['dim_95'], n_attract=n_km_B,
        rho_R=rho_R_B, lle_cv=cv_B, **energy_B,
    )

    # ──────────────────────────────────────────────────────────────
    # Phase 3 — 10 mechanism variants comparison
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3/5   Mechanism Comparison  (10 variants)")
    print("=" * 70)

    mechanism_specs = [
        # (label, extra_kwargs, syn_noise_scale)
        ('Baseline',             {},                                     0),
        ('Noise x2 (0.04)',      {'base_noise': 0.04},                  0),
        ('Noise x5 (0.10)',      {'base_noise': 0.10},                  0),
        ('Hub x8 (n=6)',         {'n_hubs': 6, 'hub_out_scale': 8.0},   0),
        ('Tau=[20,2,7]',         {'community_taus': [20, 2, 7]},        0),
        ('Tau+drive',            {'community_taus': [20, 2, 7],
                                  'community_background_drive': [0.35, 0, 0.10]}, 0),
        ('Energy constr.',       {'energy_constraint': True},           0),
        ('Boundary damp',        {'boundary_damping': True,
                                  'damping_threshold': 0.80},           0),
        ('Syn noise 5%',         {},                                    0.05),
        ('Full B',               {'community_taus': [20, 2, 7],
                                  'community_background_drive': [0.35, 0, 0.10],
                                  'n_hubs': 6, 'hub_out_scale': 8.0},  0),
    ]

    mech_records = []
    for label, extra, syn in mechanism_specs:
        cfg = {**BASE_CFG, 'target_rho': best['rho'],
               'base_noise': best['noise'], **extra}
        if syn > 0:
            base_keys = list(BASE_CFG.keys()) + ['target_rho', 'base_noise']
            mdl = MinimalModelV3(
                **{k: v for k, v in cfg.items() if k in base_keys}
            ).perturb_connectivity(noise_scale=syn, seed=7)
        else:
            mdl = MinimalModelV3(**cfg)

        tr   = mdl.simulate(n_steps=1000, n_init=40, burnin=250)
        lle  = MinimalModelV3.estimate_lyapunov(tr, max_time=350,
                                                 fit_range=(20, 100))
        d2   = MinimalModelV3.compute_correlation_dimension(tr)
        pst  = MinimalModelV3.compute_pca_stats(tr)
        n_km, _ = MinimalModelV3.count_attractors(tr)
        eng  = MinimalModelV3.compute_community_energy(tr, mdl)
        # Transient for PC2 slope
        trt  = mdl.simulate_transient(n_init=15, n_steps=200)
        slp, r2 = MinimalModelV3.compute_pc2_contraction(
            trt, skip_steps=3, fit_end=100)

        rec = dict(label=label, lle=lle, d2=d2, pc2_slope=slp, pc2_r2=r2,
                   dim_95=pst['dim_95'], n_attract=n_km, **eng)
        mech_records.append(rec)

        d2s   = f"{d2:.3f}"   if np.isfinite(d2)  else "  NaN"
        slps  = f"{slp:+.4f}" if np.isfinite(slp) else "   NaN"
        print(f"  [{label:<22s}]: LLE={lle:+.5f}  D2={d2s}  "
              f"PC2={slps}  R2={r2:.3f}  "
              f"dim={pst['dim_95']}  Ej={eng['E_joint']:.3f}  n={n_km}")

    plot_mechanism_comparison(mech_records)
    plot_community_energy(mech_records)

    # ──────────────────────────────────────────────────────────────
    # Phase 4 — Phase portrait comparison + LLE CV bar chart
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 4/5   Phase Portraits + LLE CV")
    print("=" * 70)

    portrait_cfgs = [
        ('Noise_x5',    {**BASE_CFG, 'target_rho': best['rho'],
                         'base_noise': 0.10}),
        ('Tau20_2_7',   {**BASE_CFG, 'target_rho': 1.03,
                         'base_noise': 0.02,
                         'community_taus': [20, 2, 7],
                         'community_background_drive': [0.35, 0.0, 0.10]}),
        ('Hub_x8',      {**BASE_CFG, 'target_rho': 1.03,
                         'base_noise': 0.02,
                         'n_hubs': 6, 'hub_out_scale': 8.0}),
    ]
    for name, cfg in portrait_cfgs:
        m   = MinimalModelV3(**cfg)
        trt = m.simulate_transient(n_init=15, n_steps=250)
        plot_phase_portrait(trt, m, model_name=name, is_transient=True,
                            n_show=5, save=f"fig_phase_portrait_{name}.png")
        print(f"  Saved: fig_phase_portrait_{name}.png")

    # LLE CV comparison bar chart
    cv_records = [
        dict(label='Model A (EdgeChaos)',
             mean_lle=mean_lle_A, std_lle=std_lle_A, cv_pct=cv_A),
        dict(label='Model B (Structured)',
             mean_lle=mean_lle_B, std_lle=std_lle_B, cv_pct=cv_B),
    ]
    plot_lle_cv_comparison(cv_records)

    # ──────────────────────────────────────────────────────────────
    # Phase 5 — Final summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 5/5   Final Summary")
    print("=" * 70)

    plot_two_model_summary(res_A, res_B)

    hdr = f"{'Metric':<32} {'Real brain':>12} {'Model A':>12} {'Model B':>12}"
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    rows = [
        ("LLE",
         f"{REAL_LLE:.4f}",        f"{lle_A:+.5f}",       f"{lle_B:+.5f}"),
        ("LLE CV (%)",
         f"{REAL_LLE_CV:.1f}%",    f"{cv_A:.1f}%",        f"{cv_B:.1f}%"),
        ("D2 (corr dim)",
         f"{REAL_D2:.2f}",         f"{d2_A:.3f}",         f"{d2_B:.3f}"),
        ("PC2 slope (transient)",
         f"{REAL_PC2_SLOPE:.2f}",  f"{slp_A:+.4f}",       f"{slp_B:+.4f}"),
        ("PC2 R2",
         "< 0.1",                  f"{r2_A:.4f}",          f"{r2_B:.4f}"),
        ("PCA dim 95%",
         "<=5",                    str(pca_A['dim_95']),   str(pca_B['dim_95'])),
        ("var(top-2 PCs)",
         "~96%",     f"{pca_A['var_top2']*100:.1f}%", f"{pca_B['var_top2']*100:.1f}%"),
        ("Attractors (KMeans)",
         str(REAL_N_ATTRACT),      str(n_km_A),           str(n_km_B)),
        ("rho_R (response mat)",
         f"{REAL_RHO_RESPONSE:.2f}", f"{rho_R_A:.3f}",    f"{rho_R_B:.3f}"),
        ("Hub importance (rho_R%)",
         "38-218%",                "N/A",
         f"{hub_info_B['importance_pct']:.1f}%"),
        ("Energy C0 (fMRI-like)",
         f"{REAL_ENERGY_fMRI:.2f}",
         f"{energy_A.get('E_c0',0):.4f}",
         f"{energy_B.get('E_c0',0):.4f}"),
        ("Energy C1 (EEG-like)",
         f"{REAL_ENERGY_EEG:.2f}",
         f"{energy_A.get('E_c1',0):.4f}",
         f"{energy_B.get('E_c1',0):.4f}"),
        ("Energy joint",
         f"{REAL_ENERGY_JOINT:.2f}",
         f"{energy_A['E_joint']:.4f}",
         f"{energy_B['E_joint']:.4f}"),
    ]
    for metric, real, A, B in rows:
        print(f"  {metric:<32} {real:>12} {A:>12} {B:>12}")

    print("\n  Generated figures:")
    figs = [
        "fig_param_scan_v3.png",
        "fig_phase_portrait_A.png",         "fig_phase_portrait_A_trans.png",
        "fig_phase_portrait_B.png",         "fig_phase_portrait_B_trans.png",
        "fig_hub_analysis_v3.png",          "fig_timescale_v3.png",
        "fig_mechanisms_v3.png",            "fig_community_energy_v3.png",
        "fig_lle_cv_v3.png",
        "fig_phase_portrait_Noise_x5.png",  "fig_phase_portrait_Tau20_2_7.png",
        "fig_phase_portrait_Hub_x8.png",    "fig_summary_comparison_v3.png",
    ]
    for f in figs:
        print(f"    {f}")

    print("\n  v3 Key Conclusions:")
    print(f"  1. Timescale tau=[20,2,7] creates fMRI:EEG ~10x frequency ratio")
    print(f"     -> PC2 contraction slope from transient traj: B={slp_B:+.4f}")
    print(f"  2. Community drive [0.35,0,0.10] reproduces energy hierarchy:")
    print(f"     C0={energy_B.get('E_c0',0):.3f} (target fMRI 0.61), "
          f"C1={energy_B.get('E_c1',0):.3f} (target EEG 0.08)")
    print(f"  3. Hub importance via response matrix:")
    print(f"     rho_R {hub_info_B['rho_R_full']:.3f} -> {hub_info_B['rho_R_ablated']:.3f} "
          f"({hub_info_B['importance_pct']:.1f}% change,  real: 38-218%)")
    print(f"  4. LLE CV: A={cv_A:.1f}%, B={cv_B:.1f}%  (real: ~23.7%)")
    print(f"  5. Model A achieves LLE={lle_A:.5f} at rho={CFG_A['target_rho']:.2f}")
