import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, List, Tuple

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_SEED = 42


class MinimalModel:
    """
    极简社区网络模型，支持能量约束和扰动。
    """
    def __init__(
        self,
        n_communities: int = 3,
        nodes_per_community: int = 20,
        w_intra: float = 2.0,
        w_inter_base: float = 0.4,
        inter_prob: float = 0.3,
        target_rho: float = 1.02,
        base_noise: float = 0.01,
        asymmetric_bias: float = 0.0,
        boundary_damping: float = 0.0,
        boundary_threshold: float = 0.8,
        energy_penalty: float = 0.0,
        energy_threshold: float = 0.8,
        nonlinear: str = 'tanh',
        seed: int = DEFAULT_SEED,
    ):
        self.n_communities = n_communities
        self.nodes_per_community = nodes_per_community
        self.N = n_communities * nodes_per_community
        self.w_intra = w_intra
        self.w_inter_base = w_inter_base
        self.inter_prob = inter_prob
        self.target_rho = target_rho
        self.base_noise = base_noise
        self.asymmetric_bias = asymmetric_bias
        self.boundary_damping = boundary_damping
        self.boundary_threshold = boundary_threshold
        self.energy_penalty = energy_penalty
        self.energy_threshold = energy_threshold
        self.seed = seed
        np.random.seed(seed)

        self.nonlinear = np.tanh if nonlinear == 'tanh' else nonlinear
        self._build_connectivity()
        self._scale_to_target_rho()

    def _build_connectivity(self):
        N = self.N
        W = np.zeros((N, N))
        for c in range(self.n_communities):
            idx = slice(c * self.nodes_per_community, (c + 1) * self.nodes_per_community)
            W[idx, idx] = self.w_intra / self.nodes_per_community

        for i in range(N):
            for j in range(N):
                if i // self.nodes_per_community != j // self.nodes_per_community:
                    if np.random.rand() < self.inter_prob:
                        sign = np.random.choice([-1, 1])
                        strength = self.w_inter_base * np.random.uniform(0.5, 1.5)
                        W[i, j] = sign * strength / self.nodes_per_community

        # 孤立节点处理
        row_sums = np.sum(np.abs(W), axis=1)
        isolated = np.where(row_sums == 0)[0]
        for i in isolated:
            targets = np.random.choice(N, size=min(3, N - 1), replace=False)
            for j in targets:
                sign = np.random.choice([-1, 1])
                W[i, j] = sign * 0.01 / self.nodes_per_community

        self.W_raw = W

    def _scale_to_target_rho(self):
        eigvals = np.linalg.eigvals(self.W_raw)
        rho0 = np.max(np.abs(eigvals))
        if rho0 == 0:
            raise ValueError("原始矩阵谱半径为0，无法缩放")
        self.g = self.target_rho / rho0
        self.W = self.g * self.W_raw
        print(f"原始谱半径: {rho0:.4f}, 缩放因子: {self.g:.4f}, 目标谱半径: {self.target_rho:.2f}")

    def update(
        self,
        x: np.ndarray,
        noise: Optional[float] = None,
        asymmetric_bias: Optional[float] = None,
        boundary_damping: Optional[float] = None,
        boundary_threshold: Optional[float] = None,
        energy_penalty: Optional[float] = None,
        energy_threshold: Optional[float] = None,
    ) -> np.ndarray:
        noise = self.base_noise if noise is None else noise
        asym = self.asymmetric_bias if asymmetric_bias is None else asymmetric_bias
        damp = self.boundary_damping if boundary_damping is None else boundary_damping
        bthr = self.boundary_threshold if boundary_threshold is None else boundary_threshold
        ep = self.energy_penalty if energy_penalty is None else energy_penalty
        ethr = self.energy_threshold if energy_threshold is None else energy_threshold

        dx = self.nonlinear(self.W @ x)
        noise_term = noise * np.random.randn(len(x))

        if asym > 0:
            noise_term += asym * (np.random.randn(len(x)) + 0.5)

        if damp > 0:
            over = x > bthr
            dx[over] -= damp * (x[over] - bthr)
            under = x < -bthr
            dx[under] += damp * (-x[under] - bthr)

        if ep > 0 and np.mean(np.abs(x)) > ethr:
            dx -= ep * (x / (1 + np.abs(x)))

        return dx + noise_term

    def simulate(self, n_steps=2000, n_init=100, burnin=200, **kwargs):
        N = self.N
        traj = np.zeros((n_init, n_steps, N))
        for i in range(n_init):
            x = np.random.uniform(-1, 1, N)
            for _ in range(burnin):
                x = self.update(x, noise=0.0, asymmetric_bias=0.0, boundary_damping=0.0, energy_penalty=0.0)
            traj[i, 0] = x
            for t in range(1, n_steps):
                x = self.update(x, **kwargs)
                traj[i, t] = x
        return traj

    # ---------- 分析工具 ----------
    @staticmethod
    def estimate_lyapunov(traj, max_time=500, fit_range=(20, 150)):
        n_init, n_steps, N = traj.shape
        init = traj[:, 0, :]
        dist_mat = squareform(pdist(init))
        np.fill_diagonal(dist_mat, np.inf)
        pairs = [(i, np.argmin(dist_mat[i])) for i in range(n_init)]

        times = np.arange(min(max_time, n_steps))
        log_dist = np.zeros(len(times))
        for it, t in enumerate(times):
            d = [np.linalg.norm(traj[i, t] - traj[j, t]) for i, j in pairs]
            d = np.array(d)
            d[d < 1e-12] = 1e-12
            log_dist[it] = np.mean(np.log(d))

        t_fit = times[fit_range[0]:fit_range[1]]
        y_fit = log_dist[fit_range[0]:fit_range[1]]
        slope, *_ = linregress(t_fit, y_fit)
        return slope

    @staticmethod
    def compute_pca_dim(traj, threshold=0.95):
        X = traj.reshape(-1, traj.shape[2])
        pca = PCA().fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        return np.argmax(cum >= threshold) + 1

    def compute_community_activity(self, traj):
        n_init, n_steps, _ = traj.shape
        comm = np.zeros((n_init, n_steps, self.n_communities))
        for c in range(self.n_communities):
            idx = slice(c * self.nodes_per_community, (c + 1) * self.nodes_per_community)
            comm[:, :, c] = traj[:, :, idx].mean(axis=2)
        return comm

    def _project_to_2d_and_velocity(self, traj):
        """返回 (proj, d_proj, pos_mid)"""
        X = traj.reshape(-1, self.N)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(X)               # (n_points, 2)
        d_proj = proj[1:] - proj[:-1]              # 速度
        pos_mid = proj[:-1]                        # 中间位置
        return proj, d_proj, pos_mid

    def analyze_pc2_contraction(self, traj, manifold_thresh=0.5):
        proj, d_proj, pos_mid = self._project_to_2d_and_velocity(traj)
        if manifold_thresh is not None:
            on = np.abs(pos_mid[:, 1]) < manifold_thresh
            if not np.any(on):
                print(f"警告: PC2收缩无点满足 |PC2|<{manifold_thresh}，使用全部点。")
                on = np.ones(len(pos_mid), dtype=bool)
        else:
            on = np.ones(len(pos_mid), dtype=bool)

        pc2 = pos_mid[on, 1].reshape(-1, 1)
        dpc2 = d_proj[on, 1]
        if len(pc2) == 0:
            return 0.0, 0.0

        model = LinearRegression().fit(pc2, dpc2)
        return model.coef_[0], model.score(pc2, dpc2)

    def analyze_pc1_dynamics(self, traj, manifold_thresh=0.5, n_bins=10):
        proj, d_proj, pos_mid = self._project_to_2d_and_velocity(traj)
        if manifold_thresh is not None:
            on = np.abs(pos_mid[:, 1]) < manifold_thresh
            if not np.any(on):
                print(f"警告: PC1动力学无点满足 |PC2|<{manifold_thresh}，使用全部点。")
                on = np.ones(len(pos_mid), dtype=bool)
        else:
            on = np.ones(len(pos_mid), dtype=bool)

        pc1 = pos_mid[on, 0]
        dpc1 = d_proj[on, 0]
        if len(pc1) == 0:
            return np.array([]), np.array([])

        bins = np.linspace(pc1.min(), pc1.max(), n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        means = []
        for i in range(n_bins):
            mask = (pc1 >= bins[i]) & (pc1 < bins[i+1])
            means.append(np.mean(dpc1[mask]) if np.any(mask) else np.nan)
        return centers, np.array(means)

    def run_experiment(self, n_steps=2000, n_init=100, burnin=200,
                       noise=None, asymmetric_bias=None, boundary_damping=None,
                       boundary_threshold=None, energy_penalty=None,
                       energy_threshold=None, verbose=True, **kwargs):
        if verbose:
            print("="*50)
            print("运行实验：")
            print(f" 社区数: {self.n_communities}, 节点数/社区: {self.nodes_per_community}")
            print(f" 目标谱半径: {self.target_rho:.2f}")
            print(f" 基础噪声: {self.base_noise}")
            print(f" 不对称偏置: {self.asymmetric_bias}")
            print(f" 边界阻尼: {self.boundary_damping} (阈值 {self.boundary_threshold})")
            print(f" 能量约束: {self.energy_penalty} (阈值 {self.energy_threshold})")
            print("-"*50)

        # 合并参数，优先使用传入值
        sim_kwargs = {
            'noise': noise,
            'asymmetric_bias': asymmetric_bias,
            'boundary_damping': boundary_damping,
            'boundary_threshold': boundary_threshold,
            'energy_penalty': energy_penalty,
            'energy_threshold': energy_threshold,
            **kwargs
        }
        traj = self.simulate(n_steps, n_init, burnin, **sim_kwargs)
        if verbose:
            print(f"轨迹生成完成，形状 {traj.shape}")

        pca95 = self.compute_pca_dim(traj, 0.95)
        pca99 = self.compute_pca_dim(traj, 0.99)
        lle = self.estimate_lyapunov(traj)
        slope_pc2, r2_pc2 = self.analyze_pc2_contraction(traj)
        bin_c, mean_dpc1 = self.analyze_pc1_dynamics(traj)
        comm_avg = self.compute_community_activity(traj)
        pca_2 = PCA(n_components=2).fit_transform(traj[0])
        ev = PCA(n_components=2).fit(traj.reshape(-1, self.N)).explained_variance_ratio_

        if verbose:
            print(f"PCA 95% 维度: {pca95}, 99% 维度: {pca99}")
            print(f"最大Lyapunov指数: {lle:.6f}  → {'混沌边缘' if abs(lle)<0.01 else '弱混沌' if lle>0 else '稳定'}")
            print(f"PC2收缩 slope={slope_pc2:.4f}, R²={r2_pc2:.3f}")
            if len(bin_c) > 0:
                print("沿PC1的binned dPC1/dt (前5个):")
                for i in range(min(5, len(bin_c))):
                    print(f" PC1={bin_c[i]:.2f}: {mean_dpc1[i]:.4f}")

        return {
            'trajectories': traj,
            'pca_dim_95': pca95,
            'pca_dim_99': pca99,
            'lle': lle,
            'pc2_slope': slope_pc2,
            'pc2_r2': r2_pc2,
            'pc1_bin_centers': bin_c.tolist(),
            'pc1_mean_dpc1': mean_dpc1.tolist(),
            'community_avg': comm_avg,
            'pca_proj': pca_2,
            'pca_ev': ev,
            'params': {
                'n_communities': self.n_communities,
                'nodes_per_community': self.nodes_per_community,
                'w_intra': self.w_intra,
                'w_inter_base': self.w_inter_base,
                'inter_prob': self.inter_prob,
                'target_rho': self.target_rho,
                'base_noise': self.base_noise,
                'asymmetric_bias': self.asymmetric_bias if asymmetric_bias is None else asymmetric_bias,
                'boundary_damping': self.boundary_damping if boundary_damping is None else boundary_damping,
                'boundary_threshold': self.boundary_threshold if boundary_threshold is None else boundary_threshold,
                'energy_penalty': self.energy_penalty if energy_penalty is None else energy_penalty,
                'energy_threshold': self.energy_threshold if energy_threshold is None else energy_threshold,
                'seed': self.seed,
            }
        }

    def plot_results(self, results, save_path=None):
        traj = results['trajectories']
        comm_avg = results['community_avg']
        proj = results['pca_proj']
        ev = results['pca_ev']
        params = results['params']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 社区活动
        ax = axes[0, 0]
        for c in range(self.n_communities):
            ax.plot(comm_avg[0, :, c], label=f'社区 {c+1}')
        ax.set(xlabel='时间步', ylabel='平均活动', title='社区平均活动 (第一条轨迹)')
        ax.legend(); ax.grid(True)

        # 吸引子投影
        ax = axes[0, 1]
        ax.plot(proj[:, 0], proj[:, 1], lw=0.5, alpha=0.8)
        ax.set(xlabel=f'PC1 ({ev[0]*100:.1f}%)', ylabel=f'PC2 ({ev[1]*100:.1f}%)', title='吸引子投影 (PC1-PC2)')
        ax.grid(True)

        # 摘要
        ax = axes[1, 0]
        ax.axis('off')
        text = f"""
        实验结果摘要
        =============
        LLE: {results['lle']:.6f}
        PCA95: {results['pca_dim_95']}  PCA99: {results['pca_dim_99']}
        PC2收缩: slope={results['pc2_slope']:.4f}  R²={results['pc2_r2']:.3f}
        参数:
        社区数: {params['n_communities']}
        每社区节点数: {params['nodes_per_community']}
        w_intra: {params['w_intra']}
        w_inter_base: {params['w_inter_base']}
        inter_prob: {params['inter_prob']}
        目标谱半径: {params['target_rho']}
        基础噪声: {params['base_noise']}
        不对称偏置: {params['asymmetric_bias']}
        边界阻尼: {params['boundary_damping']} (阈值 {params['boundary_threshold']})
        能量约束: {params['energy_penalty']} (阈值 {params['energy_threshold']})
        """
        ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10, va='top')

        # 1D动力学
        ax = axes[1, 1]
        if results['pc1_bin_centers'] and results['pc1_mean_dpc1']:
            ax.plot(results['pc1_bin_centers'], results['pc1_mean_dpc1'], 'o-')
            ax.set(xlabel='PC1', ylabel='平均 dPC1/dt', title='沿流形的1D动力学')
            ax.grid(True)
        else:
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存至: {save_path}")
        plt.show()


# ---------- 辅助函数 ----------
def compute_energy_budget(traj, tail_ratio=0.3):
    tail_steps = int(traj.shape[1] * tail_ratio)
    tail = traj[:, -tail_steps:, :]
    return np.mean(np.abs(tail))


def scan_parameter(param_name, values, base_config, exp_kwargs=None, verbose=True):
    if exp_kwargs is None:
        exp_kwargs = {'n_steps': 1000, 'n_init': 50, 'verbose': False}
    results = []
    for v in values:
        cfg = base_config.copy()
        cfg[param_name] = v
        model = MinimalModel(**cfg)
        exp = model.run_experiment(**exp_kwargs)
        results.append((v, exp['lle'], exp['pca_dim_95'], exp['pc2_slope'], exp['pc2_r2']))
        if verbose:
            print(f"{param_name}={v:.4f} -> LLE={exp['lle']:.6f}, PCA_dim={exp['pca_dim_95']}, PC2_slope={exp['pc2_slope']:.4f}")
    return results


# ---------- 测试流程 ----------
if __name__ == "__main__":
    print("="*80)
    print("Step 1: 运行原始模型（无扰动），计算能量预算及PC2分布")
    print("="*80)

    base_cfg = {
        'n_communities': 3,
        'nodes_per_community': 20,
        'w_intra': 2.0,
        'w_inter_base': 0.5,
        'inter_prob': 0.3,
        'target_rho': 1.02,
        'base_noise': 0.001,
        'asymmetric_bias': 0.0,
        'boundary_damping': 0.0,
        'energy_penalty': 0.0,
        'seed': 42,
    }

    model_orig = MinimalModel(**base_cfg)
    traj_orig = model_orig.simulate(n_steps=2000, n_init=20, burnin=300, noise=0.001)

    energy_budget = compute_energy_budget(traj_orig, 0.3)
    pca_dim = MinimalModel.compute_pca_dim(traj_orig)
    lle = MinimalModel.estimate_lyapunov(traj_orig)
    print(f"原始模型尾部平均能量预算: {energy_budget:.4f}")
    print(f"原始模型 PCA 95% 维度: {pca_dim}")
    print(f"原始模型 LLE: {lle:.6f}")

    # PC2分布
    X = traj_orig.reshape(-1, 60)
    proj = PCA(n_components=2).fit_transform(X)
    pc2 = proj[:, 1]
    print(f"PC2 范围: [{pc2.min():.2f}, {pc2.max():.2f}], 标准差: {pc2.std():.4f}")
    rec_thresh = max(0.5, pc2.std() * 1.5)
    print(f"建议流形阈值: {rec_thresh:.2f}")
    manifold_thresh = rec_thresh  # 使用推荐阈值

    print("\n" + "="*80)
    print("Step 2: 扰动实验对比（减小扰动）")
    print("="*80)

    # 减小扰动参数
    # 扰动参数（中等强度）
    # 扰动参数（中等强度）
    perturb = {
        'noise': 0.02,
        'asymmetric_bias': 0.005,
        'boundary_damping': 0.005,
        'boundary_threshold': 1.2,
        'n_steps': 1000,
        'n_init': 10,
        'burnin': 200,
    }

    # 有能量约束（阈值=预算×1.0, penalty=0.15）
    print("扰动 + 有能量约束（阈值=预算×1.0, penalty=0.15）")
    cfg_with = base_cfg.copy()
    cfg_with['energy_penalty'] = 0.15
    cfg_with['energy_threshold'] = energy_budget * 1.0
    model_with = MinimalModel(**cfg_with)
    res_with = model_with.run_experiment(**perturb, verbose=True)

    # 用正确阈值重新分析PC2和PC1
    slope, r2 = model_with.analyze_pc2_contraction(res_with['trajectories'], manifold_thresh=manifold_thresh)
    res_with['pc2_slope'] = slope
    res_with['pc2_r2'] = r2
    bin_c, mean_d = model_with.analyze_pc1_dynamics(res_with['trajectories'], manifold_thresh=manifold_thresh)
    res_with['pc1_bin_centers'] = bin_c.tolist() if len(bin_c) > 0 else []
    res_with['pc1_mean_dpc1'] = mean_d.tolist() if len(mean_d) > 0 else []
    print(f"修正后PC2收缩 slope={slope:.4f}, R²={r2:.3f}")
    if len(bin_c) > 0:
        print("沿PC1的binned dPC1/dt (前5个):")
        for i in range(min(5, len(bin_c))):
            print(f" PC1={bin_c[i]:.2f}: {mean_d[i]:.4f}")
    model_with.plot_results(res_with, "with_energy.png")

    # 无能量约束（与之前相同）
    print("\n扰动 + 无能量约束")
    model_without = MinimalModel(**base_cfg)
    res_without = model_without.run_experiment(**perturb, verbose=True)
    slope, r2 = model_without.analyze_pc2_contraction(res_without['trajectories'], manifold_thresh=manifold_thresh)
    res_without['pc2_slope'] = slope
    res_without['pc2_r2'] = r2
    bin_c, mean_d = model_without.analyze_pc1_dynamics(res_without['trajectories'], manifold_thresh=manifold_thresh)
    res_without['pc1_bin_centers'] = bin_c.tolist() if len(bin_c) > 0 else []
    res_without['pc1_mean_dpc1'] = mean_d.tolist() if len(mean_d) > 0 else []
    print(f"修正后PC2收缩 slope={slope:.4f}, R²={r2:.3f}")
    if len(bin_c) > 0:
        print("沿PC1的binned dPC1/dt (前5个):")
        for i in range(min(5, len(bin_c))):
            print(f" PC1={bin_c[i]:.2f}: {mean_d[i]:.4f}")
    model_without.plot_results(res_without, "no_energy.png")