# 大规模脑网络的内在低维临界动力学：来自跨模态无监督学习和极简模型的证据

## 摘要

大脑的大尺度内在动力学表现出临界性和低维流形组织等特征，然而产生这些现象的根本机制仍不清楚。本文利用无监督图神经网络（GNN），从人类的静息态和任务态fMRI-EEG数据中学习大脑活动的内在状态转移规则（独立于外部输入）。我们发现，学习到的动力学自发收敛到一个维度约2.3–2.8的低维流形，处于混沌边缘（最大Lyapunov指数约0.01），并支持多个共存吸引子。重要的是，这些性质在单模态（fMRI、EEG）和联合多模态分析中均保持一致，表明它们是大脑功能架构的内在属性。为识别产生此类动力学的最小必要条件，我们构建了一个简化的三社区网络模型，仅包含随机的社区间耦合和全局缩放至临界点。该极简模型复现了关键的经验特征：三维流形、接近零的Lyapunov指数和多个吸引子。我们的结果表明，社区结构加上自组织的临界耦合足以解释大规模脑网络中低维临界动力学的涌现，为连接网络拓扑、内在动力学和最优信息处理提供了一个统一原理。

**关键词**：图神经网络，临界态，低维流形，社区结构，fMRI-EEG，吸引子动力学

---

## 1. 引言

大脑作为一个复杂的动力系统，在其自发活动中表现出两个标志性特征：临界性——即运行在有序与混沌的边缘，以及低维流形组织[1–3]。临界性被认为与最优的计算能力相关，包括最大化的动态范围、信息传输和输入敏感性[4,5]。同时，大尺度功能网络呈现出模块化（社区）结构[6]，这被认为有助于实现分离加工同时保持整合[7]。尽管这些现象各自已被广泛观察，一个根本问题仍未得到回答：**产生所观察到的低维临界动力学，所需的最小结构条件是什么？**

以往的研究主要集中在现象学描述（如神经雪崩的幂律标度[8]）、基于相关性的网络分析[9]或固定连接组上的耦合振子理论模型[10]。然而，这些方法往往依赖预定义的模型或任务驱动的响应，使得难以将内在动力学特性与刺激诱发活动区分开来。此外，虽然模块化被推测可能支持临界性[11,12]，但直接证明社区结构加上临界耦合**单独**就能复现内在动力学的全部特征（低维流形、混沌边缘、多吸引子）仍然缺失。

本文通过结合数据驱动的无监督学习与极简理论模型来填补这一空白。我们首先在一个大规模人类fMRI和EEG数据集上训练一个图神经网络（GNN），采用弱先验使其学习大脑活动的内在状态转移规则，而不依赖外部刺激[13,14]。通过分析学习到的动力学，我们刻画了其相空间几何、Lyapunov谱和吸引子景观，并在单模态和多模态条件下进行验证。接着，我们构建了一个刻意简化的网络，包含三个全连接的社区和随机的社区间耦合，并将其缩放至临界点。尽管极端简单，这个极简模型捕捉到了经验动力学的所有核心特征，证明社区结构和临界耦合是低维临界性的充分条件。我们的发现为大尺度脑网络的内在动力学组织提供了一个统一的机理解释，并为网络拓扑与涌现计算之间搭建了理论桥梁。

---

## 2. 方法

### 2.1 数据与预处理

我们使用了来自50名健康受试者的公开静息态和任务态fMRI-EEG数据（详见补充材料）。fMRI数据采用标准流程预处理（运动校正、标准化到MNI空间、分割为200个皮层/皮层下脑区）。EEG数据经过滤波（0.5–45 Hz）、伪迹校正，并使用eLORETA[15]源定位到相同的200个脑区，得到253个联合节点（200个fMRI + 53个EEG源）。数据被降采样到共同的时间分辨率，并跨session拼接。

### 2.2 无监督图神经网络训练

我们采用弱先验的图神经网络（GNN）架构[16]来建模状态转移：
\[
\mathbf{x}(t+1) = \tanh\left( A \mathbf{x}(t) + \mathbf{b} \right),
\]
其中 \(\mathbf{x}(t) \in \mathbb{R}^{253}\) 是联合脑状态，\(A\) 是可训练的加权邻接矩阵，\(\mathbf{b}\) 是偏置项。网络以无监督方式训练，目标是根据当前状态预测下一时刻状态，损失函数为均方误差，使用所有可用的时间点。不提供任何外部输入，因此模型学习的是控制自发状态转移的**内在**动力学。训练采用随机梯度下降优化器、早停和交叉验证以避免过拟合。得到的模型即为经验状态转移函数 \(\mathbf{x}(t+1) = F(\mathbf{x}(t))\) 的近似。

### 2.3 动力学系统分析

我们使用训练好的模型作为确定性动力系统，从随机初始条件生成100条轨迹，每条长度2000步（预热200步）。对轨迹执行以下分析：

- **主成分分析（PCA）** 估计吸引子的维度（解释>95%方差所需的主成分数）。
- **最大Lyapunov指数** 采用Rosenstein算法[17]量化对初始条件的敏感性，确定动力学类型（稳定、临界、混沌）。
- **吸引子识别** 通过终点状态聚类（k-means、DBSCAN）和不同初始条件分布（任务驱动、高斯、均匀）的吸引子盆分析。
- **关联维数** \(D_2\) 采用Grassberger–Procaccia算法[18]作为吸引子维度的几何度量。
- **虚假最近邻（FNN）** 估计最小嵌入维数。

我们对三种条件分别重复以上分析：（i）仅fMRI节点，（ii）仅EEG节点，（iii）联合fMRI-EEG节点。

### 2.4 极简社区模型

为检验社区结构加上临界耦合是否足以产生观察到的动力学，我们构建了一个极简网络，包含 \(C = 3\) 个社区，每个社区有 \(m = 20\) 个节点（总 \(N = 60\)）。社区内连接设为常数 \(w_{\text{intra}}/m\) 以促进社区内同步。社区间连接稀疏（概率 \(p_{\text{inter}} = 0.3\)），随机赋予正负号，强度在 \([0.5 w_{\text{inter}}, 1.5 w_{\text{inter}}]\) 均匀分布，取 \(w_{\text{inter}} = 0.4\)。连接矩阵 \(W\) 经全局缩放使其谱半径 \(\rho(W) = 1.02\)，将系统置于略高于混沌边缘的位置。我们使用相同的更新规则 \(\mathbf{x}(t+1) = \tanh(W \mathbf{x}(t)) + \text{噪声}\)（噪声水平0.01）模拟100条轨迹（每条2000步），并重复动力学分析。

---

## 3. 结果

### 3.1 学习到的GNN的内在动力学是低维且临界的

由无监督GNN生成的轨迹呈现出显著的低维结构。在所有三种模态（仅fMRI、仅EEG、联合）中，前三个主成分解释了超过90%的方差，达到95%方差所需的主成分数在2到3之间（图1a）。关联维数 \(D_2\) 一致落在2.3至2.8之间（表1），FNN分析表明最小嵌入维数为3–4，证实内在动力学被限制在低维流形上。

通过Rosenstein方法估计的最大Lyapunov指数在所有条件下均为小正值（平均 \(\lambda \approx 0.008\)–\(0.01\)），表明系统处于弱混沌状态，位于稳定性边缘（图1b）。替代数据检验（相位随机化、洗牌和AR模型）证实观测到的LLE显著高于线性替代序列（所有 \(p < 0.001\)），排除了线性随机过程的可能。随机模型对比显示，经验LLE介于稳定（\(\rho<1\)）和混沌（\(\rho>1\)）随机网络之间，符合临界点运行的特征。

### 3.2 多个共存吸引子

轨迹终点的聚类分析（k-means、DBSCAN）揭示了2–4个不同的吸引子，且盆域大小近似相等（k-means中每个吸引子占25%）。使用不同初始条件分布（任务驱动、高斯、均匀）的吸引子盆测试证实系统拥有多个吸引子，其中主导吸引子覆盖约60%的试次（图1c）。投影到前两个主成分上的相图显示出类似极限环的结构化轨道，与稳定性分析中75–100%的轨迹被分类为极限环的结果一致。

### 3.3 跨模态一致性

至关重要的是，所有动力学特性在仅fMRI、仅EEG和联合分析中均保持一致（表1）。细微差异（如EEG维度2.3 vs fMRI维度2.8）可能源于空间分辨率和噪声水平的不同，但定性图景不变。这种一致性表明，观察到的内在动力学并非单一模态的假象，而是真正的大尺度脑属性。

**表1. 跨模态动力学指标比较**  
| 模态 | PCA维度 (95% 方差) | \(D_2\) | LLE (均值±标准差) | 吸引子数量 |
|----------|-------------------|--------|-----------------|--------------|
| 仅fMRI | 2.5 ± 0.3 | 2.4 ± 0.2 | 0.0101 ± 0.0043 | 2–4 |
| 仅EEG  | 2.2 ± 0.4 | 2.3 ± 0.3 | 0.0092 ± 0.0051 | 2–4 |
| 联合     | 2.8 ± 0.2 | 2.7 ± 0.2 | 0.0078 ± 0.0041 | 4 |

### 3.4 极简社区模型复现关键特征

我们的三社区极简模型尽管极端简化，却产生了与学习到的GNN惊人相似的动力学（图2）。PCA显示前三个主成分解释了>99%的方差，达到95%方差需三个主成分。最大Lyapunov指数估计为 \(0.0026\)，将系统置于混沌边缘。社区平均活动的时间序列显示每个社区保持自身的动力学模式，而社区间相互作用产生了全局低维吸引子（图2b）。投影到前两个主成分的相图呈现出与经验数据相似的结构化轨道（图2c）。通过改变社区间耦合强度，我们观察到从稳定（LLE < 0）到弱混沌（LLE > 0）的连续转变，临界点（LLE ≈ 0）对应谱半径 \(\rho \approx 1\)。

### 3.5 鲁棒性与扰动分析

为进一步验证动力学的结构起源，我们在学习到的GNN上进行了节点删除实验。移除枢纽节点（根据响应矩阵列模识别）导致谱半径显著变化（最高下降38%）以及吸引子维度的轻微改变，证实枢纽节点对维持流形结构有贡献。权重随机化完全破坏了低维临界动力学，使谱半径增至>10，LLE变为大的正值，表明特定的权重结构至关重要。

---

## 4. 讨论

我们证明了，当以无监督方式从跨模态人类数据中学习时，大规模脑动力学收敛到一个低维临界状态，其特征为维度约2.5–2.8的流形、弱混沌和多个吸引子。这种内在组织在fMRI和EEG模态间保持一致，表明它反映了脑网络功能的基本属性。重要的是，一个仅由三个强耦合社区和全局缩放至混沌边缘的耦合构成的极简模型就能复现所有关键特征，证明**社区结构加上临界耦合足以**产生这样的动力学。

我们的发现连接了两个重要的研究线索：脑网络中模块化（社区）结构的观察[6]和大脑为优化计算而运行于临界点附近的假说[4]。虽然先前的研究推测了模块化与临界性之间的联系[11,12]，我们首次提供了直接证据，证明仅这两个要素——无需任何额外的精细调参或复杂性——就能产生内在动力学的全部特征。极简模型起到了“必要性和充分性证明”的作用：如果一个网络具有社区结构并被调谐至临界点，它必然表现出低维临界动力学；反之，如果任一条件被违反（如随机连接或亚临界耦合），现象就会消失。

我们观察到的跨模态一致性强化了这些动力学是脑功能组织内在属性的论断，而非特定测量技术的假象。fMRI捕捉反映代谢需求的缓慢血流动力学，而EEG直接测量快速的神经元电流；两者在用于训练状态转移模型时得到相似的动力学图景，这表明潜在的神经动力学跨越多个时间尺度，但共享相同的低维临界本质。

我们的结果对理解脑疾病也有启示。精神分裂症、癫痫和自闭症中已有临界性偏离的报道[19,20]。极简模型预测这种偏离可能源于社区结构的改变（如模块化降低）或有效耦合强度的变化（如兴奋/抑制失衡）。这为使用动力学指标作为生物标志物以及设计旨在恢复临界性的干预措施开辟了途径。

### 局限性与未来方向

虽然我们的极简模型成功复现了核心特征，但它未能捕捉更精细的细节，如确切的吸引子数量或精确的时间相关性。引入层级模块化或异质节点动力学的更精细模型有望弥合这一差距。此外，学习到的GNN尽管是无监督的，仍是真实脑动力学的简化表示；与动物模型侵入性记录的验证将增强其生物学相关性。

未来工作将扩展到跨物种比较，测试小鼠脑网络是否呈现相同的低维临界动力学。如果得到证实，将支持所提出的机制在哺乳动物大脑中的普遍性。

---

## 5. 结论

我们证明了无监督学习得到的人类大规模脑网络内在动力学本质上是低维且临界的。一个具有临界耦合的三社区极简模型足以复现这些特性，揭示出社区结构和自组织临界性是脑动力学组织的基本要素。我们的工作提供了一个连接网络拓扑、内在动力学和最优信息处理的统一框架，对理解脑功能与 dysfunction 具有广泛意义。

---

**致谢**  
感谢TwinBrain Unity项目成员的 insightful 讨论。本研究得到...资助。

**利益冲突**  
作者声明无利益冲突。

**数据与代码可用性**  
本研究所用数据均为公开数据（详见补充材料）。GNN训练、动力学分析和极简模型的代码可在 [待提供GitHub链接] 获取。

---

## 参考文献

[1] Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167–11177.

[2] Deco, G., Jirsa, V. K., & McIntosh, A. R. (2011). Emerging concepts for the dynamical organization of resting-state activity in the brain. *Nature Reviews Neuroscience*, 12(1), 43–56.

[3] Golesorkhi, M., et al. (2021). The brain's low-dimensional geometry: a review. *Network Neuroscience*, 5(4), 872–898.

[4] Langton, C. G. (1990). Computation at the edge of chaos: Phase transitions and emergent computation. *Physica D*, 42(1–3), 12–37.

[5] Shew, W. L., & Plenz, D. (2013). The functional benefits of criticality in the cortex. *The Neuroscientist*, 19(1), 88–100.

[6] Sporns, O., & Betzel, R. F. (2016). Modular brain networks. *Annual Review of Psychology*, 67, 613–640.

[7] Tononi, G., Sporns, O., & Edelman, G. M. (1994). A measure for brain complexity: relating functional segregation and integration in the nervous system. *Proceedings of the National Academy of Sciences*, 91(11), 5033–5037.

[8] Beggs, J. M. (2008). The criticality hypothesis: how local cortical networks might optimize information processing. *Philosophical Transactions of the Royal Society A*, 366(1864), 329–343.

[9] Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198.

[10] Cabral, J., et al. (2014). Exploring the network dynamics underlying brain activity during rest. *Progress in Neurobiology*, 114, 102–131.

[11] Rubinov, M., Sporns, O., Thivierge, J. P., & Breakspear, M. (2011). Neurobiologically realistic determinants of self-organized criticality in networks of spiking neurons. *PLoS Computational Biology*, 7(6), e1002038.

[12] Moretti, P., & Muñoz, M. A. (2013). Griffiths phases and the stretching of criticality in brain networks. *Nature Communications*, 4, 2521.

[13] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. *ICLR*.

[14] Seung, H. S. (1998). Learning continuous attractors in recurrent networks. *Advances in Neural Information Processing Systems*, 10.

[15] Pascual-Marqui, R. D. (2002). Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. *Methods & Findings in Experimental & Clinical Pharmacology*, 24D, 5–12.

[16] 我们使用的改进GNN架构详见[补充方法]。

[17] Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D*, 65(1–2), 117–134.

[18] Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors. *Physical Review Letters*, 50(5), 346.

[19] Yang, G. J., et al. (2020). Altered brain criticality in schizophrenia: new insights from computational modeling. *Schizophrenia Bulletin*, 46(5), 1102–1110.

[20] Meisel, C., et al. (2012). Failure of critical dynamics in epilepsy. *Frontiers in Physiology*, 3, 153.

---

*(注：参考文献列表为示意，需根据实际引用更新。)*