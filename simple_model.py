"""
极简图神经网络模型：三个社区，每个社区内强耦合，社区间随机稀疏连接。
通过调节参数，可复现低维临界动力学（维度≈2-3，LLE≈小正数）。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.decomposition import PCA


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ================== 参数设置 ==================
np.random.seed(42)                     # 可重复性

n_nodes_per_community = 20              # 每个社区节点数（增加以丰富内部动力学）
n_communities = 3                       # 社区数
N = n_nodes_per_community * n_communities

# 耦合强度
w_intra = 2.0                           # 社区内基础强度（强耦合，促使同步）
w_inter_base = 0.4                       # 社区间基础强度（调节以控制同步程度）
inter_prob = 0.3                         # 社区间连接概率

noise_level = 0.01                       # 过程噪声，帮助系统摆脱完全同步

# 构建连接矩阵 W (N x N)
W = np.zeros((N, N))

# 社区内全连接（包括自环）
for c in range(n_communities):
    idx = slice(c*n_nodes_per_community, (c+1)*n_nodes_per_community)
    # 强度除以节点数，使总输入规模与社区大小无关
    W[idx, idx] = w_intra / n_nodes_per_community

# 社区间随机连接（非对称，可正可负，模拟兴奋/抑制）
for i in range(N):
    for j in range(N):
        if i//n_nodes_per_community != j//n_nodes_per_community:  # 不同社区
            if np.random.rand() < inter_prob:
                # 随机符号，强度在一定范围内波动
                sign = np.random.choice([-1, 1])
                strength = w_inter_base * np.random.uniform(0.5, 1.5)
                W[i, j] = sign * strength / n_nodes_per_community

# 检查并修正孤立节点（行和为零则添加微小随机连接）
row_sums = np.sum(np.abs(W), axis=1)
isolated = np.where(row_sums == 0)[0]
for i in isolated:
    # 随机连接到其他节点
    targets = np.random.choice(N, size=3, replace=False)
    for j in targets:
        sign = np.random.choice([-1, 1])
        W[i, j] = sign * 0.01 / n_nodes_per_community

# 计算谱半径并缩放至略大于1，以进入混沌边缘
eigvals = np.linalg.eigvals(W)
rho0 = np.max(np.abs(eigvals))
target_rho = 1.02                        # 略大于1，诱发弱混沌
g = target_rho / rho0
W_scaled = g * W
print(f"原始谱半径: {rho0:.4f}, 缩放后谱半径: {target_rho:.2f}")

# ================== 定义动力学 ==================
def update(x, W, noise=0.0):
    """一步更新：x_{t+1} = tanh(W @ x) + 噪声"""
    return np.tanh(W @ x) + noise * np.random.randn(len(x))

def simulate(W, n_steps=2000, n_init=100, burnin=200, noise=0.0):
    """生成多条轨迹"""
    N = W.shape[0]
    trajectories = np.zeros((n_init, n_steps, N))
    for i in range(n_init):
        # 随机初始状态，均匀分布在[-1,1]
        x = np.random.uniform(-1, 1, N)
        # 预热，让系统进入吸引子
        for _ in range(burnin):
            x = update(x, W, noise=0)      # 预热期不加噪声，确保进入固有吸引子
        trajectories[i, 0, :] = x
        for t in range(1, n_steps):
            x = update(x, W, noise=noise)
            trajectories[i, t, :] = x
    return trajectories

# ================== 运行模拟 ==================
print("生成轨迹中...")
traj = simulate(W_scaled, n_steps=2000, n_init=100, burnin=200, noise=noise_level)
print(f"轨迹形状: {traj.shape}")

# ================== 分析 ==================
# 1. PCA 维度
X = traj.reshape(-1, N)
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.argmax(cumsum >= 0.90) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
n_99 = np.argmax(cumsum >= 0.99) + 1
print(f"PCA 前3个主成分解释方差比例: {pca.explained_variance_ratio_[:3]}")
print(f"达到90%方差所需主成分数: {n_90}")
print(f"达到95%方差所需主成分数: {n_95}")
print(f"达到99%方差所需主成分数: {n_99}")

# 2. 最大Lyapunov指数估计 (Rosenstein方法简化版)
def estimate_lyapunov(traj, dt=1, max_time=500):
    n_init, n_steps, N = traj.shape
    # 计算所有初始状态之间的距离矩阵
    init_states = traj[:, 0, :]
    dist_mat = squareform(pdist(init_states))
    np.fill_diagonal(dist_mat, np.inf)
    # 为每条轨迹找到最近邻
    pairs = []
    for i in range(n_init):
        j = np.argmin(dist_mat[i])
        pairs.append((i, j))
    # 跟踪距离随时间演化
    times = np.arange(min(max_time, n_steps))
    avg_log_dist = np.zeros(len(times))
    for idx_t, t in enumerate(times):
        log_dists = []
        for i, j in pairs:
            d = np.linalg.norm(traj[i, t, :] - traj[j, t, :])
            if d > 1e-12:
                log_dists.append(np.log(d))
            else:
                log_dists.append(-20)  # 极小距离
        avg_log_dist[idx_t] = np.mean(log_dists)
    # 对线性区域拟合斜率 (通常取10~100步)
    fit_start, fit_end = 20, 150
    t_fit = times[fit_start:fit_end]
    y_fit = avg_log_dist[fit_start:fit_end]
    slope, intercept, r_value, p_value, std_err = linregress(t_fit, y_fit)
    return slope, avg_log_dist, times

lle, log_dist_curve, times = estimate_lyapunov(traj, max_time=500)
print(f"估计最大Lyapunov指数: {lle:.6f}")
if abs(lle) < 0.01:
    print("→ 系统处于混沌边缘 (临界态)")
elif lle > 0:
    print("→ 系统弱混沌")
else:
    print("→ 系统稳定")

# 3. 可视化社区平均活动
community_avg = np.zeros((traj.shape[0], traj.shape[1], n_communities))
for c in range(n_communities):
    idx = slice(c*n_nodes_per_community, (c+1)*n_nodes_per_community)
    community_avg[:, :, c] = traj[:, :, idx].mean(axis=2)

plt.figure(figsize=(12, 4))
for c in range(n_communities):
    plt.plot(community_avg[0, :, c], label=f'社区 {c+1}')
plt.xlabel('时间步')
plt.ylabel('平均活动')
plt.title('三个社区的平均活动 (第一条轨迹)')
plt.legend()
plt.grid(True)
plt.show()

# 4. 吸引子投影 (PC1-PC2)
# 为清晰起见，仅显示第一条轨迹
pca_2 = PCA(n_components=2)
X_traj0 = traj[0, :, :]  # (2000, N)
proj = pca_2.fit_transform(X_traj0)
plt.figure(figsize=(6, 6))
plt.plot(proj[:, 0], proj[:, 1], linewidth=0.5, alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('吸引子投影 (PC1-PC2)')
plt.grid(True)
plt.show()

# 5. 打印摘要
print("\n=== 极简模型最终结果 ===")
print(f"有效维度 (95%方差): {n_95}")
print(f"最大Lyapunov指数: {lle:.4f}")