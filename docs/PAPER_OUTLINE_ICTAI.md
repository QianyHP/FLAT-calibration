# FLAT · IEEE ICTAI 2026 论文大纲（Short · 5 页）

> 写作总路线图。实验数字以 [`EXPERIMENT.md`](EXPERIMENT.md) 为准；LaTeX 见 [`paper/root.tex`](../paper/root.tex)。
> 配套 skill：`ccf-paper-writer`（起草）、`research-paper-writing`（逐段）、`ccf-paper-reviewer`（自审）、`ccfa-paper-figures`（作图）。

---

## 0. 定位与命题

| 项 | 值 |
|----|----|
| 会议 | [IEEE ICTAI 2026](https://ictai.computer.org/2026/)（Short ≤ 5 页，IEEE 双栏 10pt，双盲） |
| 议题锚点 | decision / optimization under cost · sample-efficient learning；交通是高价值 application |
| 贡献定位 | 一个方法（FLAT）+ 一个可迁移结论（目标几何与序贯预算分配，是预算受限标定样本效率的支配因素） |

**Thesis（全文只证这一句）：**

> 在硬仿真预算（$B\approx10^2$）下，*如何塑造目标函数* 与 *每一步序贯仿真投在哪*，是决定标定质量的支配因素。FLAT 用低维**行为指纹**把崎岖的「参数→误差」曲面平滑到 40 个样本即可被廉价代理拟合（$R^2\ge0.94$），使黑盒目标对廉价代理**可学（learnable）**，再以退火 LCB 序贯采点把每次昂贵仿真投到最能降误差处；由此在 100 次预算内一致且显著地优于梯度、种群、演化策略、密度估计四大优化族。

---

## 1. 标题（已定）

> **FLAT: Smoothing the Rugged Landscape for Learnable, Sample-Efficient Traffic Calibration**

`FLAT` = **F**ingerprint-guided **L**earnable **A**cquisition for **T**raffic Calibration。全称把论文核心论点（使黑盒目标对廉价代理 *learnable*）直接写入 acronym；标题以 "smoothing the rugged landscape" 呼应核心机制，`sample-efficient` 直击 ICTAI 议题。

---

## 2. Storyline Blueprint（起草前先成立）

```text
Task      : 微观交通仿真（SUMO–TraCI）跟驰/换道参数标定 = 10 维昂贵黑盒优化，单次评估 = 一次完整仿真。
Value     : 数字孪生可信度；ITS 决策与仿真评估都依赖标定保真度。
Existing  : 直接优化（SPSA/GA/CMA-ES）与密度估计式 BO（TPE）。
Gap       : 硬预算下它们或样本饥渴、或在崎岖噪声曲面上失稳，且缺少「公平预算 + 多场景 + 统计检验」的系统证据。
Root      : 它们直接在原始、崎岖、高方差的匹配目标上搜索，没有可复用的廉价目标模型，每次昂贵仿真的信息增益低。
Insight   : 先改目标的几何（领域指纹→平滑、低方差、可学标量），再让廉价代理建模它、用序贯 LCB 把稀缺预算投到最能降误差处。
Method    : FLAT = 行为指纹目标 J_b + LHS 初始设计 + 退火 LCB 序贯采集（信赖域 + 全局候选）；代理可插拔（RF 默认 / MLP）。
Evidence  : 表I 六场景均值最优；表II 30 配对 Wilcoxon 全 p<0.05；图3 N_init 消融（纯 LHS 最差）；R²≥0.94@40；RF≈MLP。
Scope     : 增益集中在中等崎岖场景；平底场景存在由问题本身决定的不可约下界。
```

**故事弧**：任务重要 → 四族在硬预算下不足 → 根因「直接搜崎岖原始目标、无廉价模型」 → 洞察「先平滑目标使其可学，再序贯花预算」 → FLAT 实现 → 证据逐条验证 → 适用域。

---

## 3. 贡献（3 条，每条有证据）

1. **FLAT 方法**：将*平滑化的领域行为指纹目标*与*退火 LCB 序贯代理采集*显式耦合、代理模块化；这一耦合使黑盒目标对廉价代理可学，在硬预算下最大化每次昂贵仿真的信息增益（§II；Algorithm 1）。
2. **可迁移发现 + 机理**：「平滑目标 + 序贯预算分配」在四个机制异构的优化族上一致且统计显著地领先（表I/II）；并由三条证据揭示机理——指纹使曲面可学（$R^2\ge0.94$@40）、序贯采集有因果贡献（图3）、增益结构性（RF≈MLP）。即目标几何与预算分配、而非优化器族选择，是支配因素。
3. **公平可复现协议**：六异构场景 × 五 seed、统一 100 预算、30 个 scene×seed 配对显著性检验，连同目标与场景设定一并开源，可作为预算受限标定的评测基线。

---

## 4. 摘要（~120 词，6 拍）

1. 任务：SUMO 标定 = 10 维昂贵黑盒，工程仅 $B\approx10^2$ 次仿真。
2. 缺口/根因：现有优化族直接搜崎岖原始目标，预算内样本饥渴或失稳。
3. 方法：we present **FLAT**——行为指纹平滑目标（使其对廉价代理可学）+ 退火 LCB 代理序贯采点（代理可插拔）。
4. 证据：六场景五 seed，对 SPSA/GA/CMA-ES/TPE 均值最优；**30 配对 Wilcoxon 全 $p<0.05$**。
5. 机理：增益来自序贯采集而非初始设计，且 RF≈MLP（结构性、与模型容量无关）。
6. 影响 + 开源：为预算受限仿真标定给出可迁移结论与可复现基线。

**Index Terms**：sample-efficient decision optimization; surrogate-assisted optimization; simulation-based calibration; behavioral fingerprint; SUMO.

---

## 5. 章节级写作计划（5 页四节）

> 规则：一段一义、首句点题、术语稳定、Claim 必有 Evidence。

### §I Introduction（~1.0 页，5 段；文献内嵌，不设独立 Related Work）

| 段 | 角色 | 要点 |
|----|------|------|
| 1 | 任务 + 应用 + 预算钩子 | 数字孪生需标定 SUMO 10 维参数；一次评估 = 一次完整仿真；工程预算 $B\approx10^2$ → 昂贵黑盒优化。 |
| 2 | 现有方法 + 根因（内嵌文献） | 直接优化样本饥渴/失稳；密度估计式 BO 建模 $p(x\mid y)$。根因：都在崎岖高方差的原始目标上直接搜，缺廉价目标模型。 |
| 3 | **洞察段（全文锚点，单独成段）** | 样本效率取决于两个被低估的决策：① 优化什么表示（指纹平滑目标）② 下一步仿真投哪（序贯 LCB）。 |
| 4 | 方法一句话 + Fig.1 | 命名 FLAT；指向 Fig.1（目标地形），一句话讲清「平滑目标 + 序贯采点」如何咬合。 |
| 5 | 贡献枚举 + 结构 | §3 三条贡献；§II 方法，§III 实验，§IV 结论。 |

纪律：不用「先给朴素解再改进」式写法；不堆 SUMO 参数名；未定义 $J_b$ 不报数字。

### §II FLAT Method（~1.3 页）

- **II-A 目标函数 $J_b$**（式1）：8 维指纹一句列出；点明平滑 + 低方差 + 场景自适应权重 = 领域先验。**Fig.1** 置于本节：XAM-N6 上 raw 逐秒速度 RMSE vs. $J_b$ 双 3-D 曲面对比（见 §6.1）。
- **II-B 固定预算协议**：$B{=}100$；Phase A 40 LHS；Phase B 60 序贯；TraCI 写入 10 维 $\theta$；报告 `error_at_budget`。
- **II-C 代理 + 退火 LCB**：RF（500 树，树间离散度 = σ）默认；MLP（10 成员集成）可选；$\mathrm{LCB}=\mu-\kappa\sigma$，$\kappa$ 2.0→0.5；信赖域 + 全局候选 + 去重。一句机理：σ 让 LCB 在不确定处探索、确定处收敛。
- **II-D Algorithm 1**（压缩伪代码）。

### §III Experiments（~2.0 页）

- **III-A 协议**：6 场景、$B{=}100$、5 seed、四族基线；所有方法共享同一场景边界与同一目标 $J_b$；30 配对单侧 Wilcoxon。
- **III-B 主对比**：表I 六场景均值（FLAT 0.241；TPE +6.7%、GA +14.7%、CMA-ES +16.3%、SPSA +59.3%）含显著性标记；Fig.2 对比组合图。叙事句：领先且 seed 带最窄（稳健）。
- **III-C 机理与消融**：Fig.3 $N_{\mathrm{init}}$ 扫描——纯 LHS 几乎总最差 ⇒ 序贯采集是增益来源；RF≈MLP，MLP 仅难场景兑现（RML 0.434→0.408）⇒ 增益结构性；40 样本即 $R^2\ge0.94$ 佐证指纹平滑了目标。
- **III-D 适用域**：增益集中在中等崎岖场景；平底场景（西安，六法≈0.43）存在由问题本身决定的不可约下界——界定方法适用域。

### §IV Conclusion（~0.3 页）

- 一句重述 Thesis；一句总结证据（跨四族显著领先 + 序贯消融）。
- 一句展望：路网级标定与在线增量更新。

---

## 6. 图表

### 6.1 Fig.1 目标地形（已入稿）

**状态**：已写入 `paper/root.tex` §II-A（Behavioral Fingerprint Objective），caption 仅保留技术要点（切片、归一化、Left/Right 标签）。

**资产**：

- 图：`outputs/figures/landscape_dual_XAM-N6.{pdf,png}`
- 数据：`outputs/results/landscape_dual_XAM-N6.json`（$20\times20$ 网格，400 次 SUMO）
- 脚本：`run_landscape_dual.py` → `plot_landscape_dual.py`

**信息**：左 raw 逐秒平均速度 RMSE（崎岖）；右指纹 $J_b$（更平滑）。各曲面 min–max 归一化到 $[0,1]$，便于比较形状而非绝对量纲。

**正文一句**：在 XAM-N6 上对 `(accel, tau)` 切片可见，指纹目标显著平滑参数–误差响应面（Fig.1）。

> 注：早期规划的 3 面板流程 teaser（A 问题 / B 方法 / C 收敛）已弃用；方法节以真实地形扫描替代示意曲线。`export_fig1_teaser_assets.py` 导出的 Part C 收敛/柱图元素仍可用于 poster 或补充材料。

### 6.2 数据图/表（已优化）

- **Fig.2 `method_comparison_panel`**：已删 score-proxy 子图（$J_b$ 的线性变换，冗余），改为「上=收敛、下=$J_b$@100 柱状」两行。
- **Table I（主结果）**：已并入显著性 `†` 列（`Method | mean $J_b$ ↓ | Δ% vs FLAT | sig.`）。
- **Fig.3 `calib_conv_*`**（§III-C）：六个独立 16:9 子图（subfloat (a)–(f)），逐场景 RF vs MLP 收敛 + 逐场景 $R^2$ 标注。**learning 证据主图**：佐证「RF≈MLP（结构性）」与「指纹使目标可学（$R^2\ge0.94$@40）」。
- **Fig.4 `n_init_*`**：六子图 (a)–(f)，每子图均带 x 轴标签 `$N_{\mathrm{init}}$`；确保纯 LHS（$N_{\mathrm{init}}{=}100$）视觉最差、RF/MLP 两线可分。
- **场景显示名**：`Xian`→**Xi'an**、`XAM-N6`→**XAM**（仅显示名；数据键/文件名保持不变）。
- **补充材料**：`multiscene_method_convergence`、分场景全表（EXPERIMENT 表3）。

---

## 7. Claim → Evidence（投稿前逐条核对）

| Claim | 出处 | 证据 |
|------|------|------|
| C1 「组合」固定预算下显著优于四族 | §III-B | 表I + 30 配对 Wilcoxon |
| C2 增益来自序贯采集而非初始设计 | §III-C | 图3：纯 LHS 最差 |
| C3 增益结构性、与模型容量无关 | §III-C | RF≈MLP；MLP 仅难场景兑现 |
| C4 指纹平滑了目标 | §II/§III-C、Fig.1 | 40 样本 $R^2\ge0.94$；地形双曲面对比 |
| C5 适用域清晰 | §III-D | 西安平底≈0.43（不可约下界） |

---

## 8. 写作顺序

1. 表I + 图3 caption（数字已齐，最稳）。
2. §II 方法 + Algorithm 1（与 `unified_calibration.py` 符号对齐）；Fig.1 已就位。
3. §III 实验正文。
4. **§I Introduction 最后写**（任务 → 现有方法 + 根因 → 洞察 → 方法 → 贡献）。
5. Abstract → Conclusion → 通读对照 §7 → 砍页。
6. 自审：`ccf-paper-reviewer` 预投稿评审。

---

## 9. 待办

- [x] 标题、Fig.2（去 score-proxy）、Table I（并显著性列）已定稿并同步 `paper/root.tex`。
- [x] **Fig.1 目标地形**：`landscape_dual_XAM-N6` 已入稿 §II-A。
- [ ] 用 `ccf-paper-writer` 起草 §II/§III 正文 → 编译核对 ≤ 5 页。

---

## 10. 代码/证据映射

| 论文元素 | 仓库位置 |
|----------|----------|
| $J_b$、8 维指纹、场景权重、收紧边界 | `code/calibration/unified_calibration.py`（`FEATURE_*`、`get_param_bounds`、`get_feature_weights`） |
| 退火 LCB、信赖域、候选去重 | 同上（`_kappa_at_step`、`_propose_candidates`、`TRUST_FRAC`） |
| 六方法公平对比、预算协议 | `code/experiments/run_comparison.py`（`MAIN_METHODS`、`BUDGET_FAIR`） |
| 30 配对 Wilcoxon | `code/experiments/analyze_significance.py` → `comparison_significance.csv` |
| Fig.1 地形扫描 | `run_landscape_dual.py`、`plot_landscape_dual.py` → `landscape_dual_XAM-N6.*` |
| 其余图 | `code/experiments/plot_all_figures.py` → `outputs/figures/` |

---

## 11. 自审清单（投稿前必过）

- [ ] Intro 第 3 段（洞察）能脱离方法独立成立、且可证伪。
- [ ] 三条贡献相互独立、删一条不塌其余。
- [ ] §7 每条 Claim 有 Evidence，Abstract 无 overclaim。
- [ ] 全程不出现「碾压」「全面超越」；TPE 处用「20/30 配对，$p{=}0.009$」式表述。
- [ ] 术语全程稳定（fingerprint / surrogate / LCB / budget / FLAT）。
- [ ] PDF ≤ 5 页含参考文献。

---

*本大纲随 `EXPERIMENT.md` 与 `paper/root.tex` 同步；定稿前以实验 CSV 更新表号数字。*
