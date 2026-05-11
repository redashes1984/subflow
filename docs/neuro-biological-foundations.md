# Subflow Neuro-Biological Foundations

> **Status:** Research Draft v1.0 | **Date:** 2026-05-11
> **Cross-Disciplinary:** Neuroscience × AI Agent Architecture
> **Repository:** https://github.com/redashes1984/subflow

---

## 1. Core Thesis

**LLM 的采样参数（temperature/top_k/top_p/seed）并非纯数学概念——它们在人类认知中有对应的神经生物学机制。**

Subflow 的参数映射不应是工程猜测，而应基于人类大脑如何调节：
- 情绪如何影响决策风险偏好
- 恐惧如何改变注意力范围
- 多巴胺如何控制探索 vs 利用的平衡
- 梦境状态如何重塑神经连接

---

## 2. Biological Mechanisms → LLM Parameter Mapping

### 2.1 多巴胺系统（Dopaminergic System）→ Temperature

**生物学事实：**
- 多巴胺不是"快乐分子"，而是**预测误差信号**（Prediction Error Signal）
- 高多巴胺 → 更高的探索倾向（Exploration），更少的局部最优陷阱
- 低多巴胺 → 更高的利用倾向（Exploitation），更保守的决策
- VTA（腹侧被盖区）→ 前额叶通路控制风险偏好

**LLM 映射：**
```
Temperature ≈ 多巴胺驱动的探索概率

高信任/安全感 → 多巴胺释放 → Temperature ↑ → 更多探索
高恐惧/威胁感 → 多巴胺抑制 → Temperature ↓ → 保守利用
```

**证据：**
- 人类在安全环境中更愿意尝试新策略（温度高）
- 人类在威胁环境中倾向于已知策略（温度低）
- ADHD（多巴胺调节异常）患者的行为模式类似高 temperature 采样

### 2.2 杏仁核（Amygdala）→ Top_k

**生物学事实：**
- 杏仁核是恐惧/威胁检测中心
- 高杏仁核激活 → 注意力窄化（Attentional Narrowing）→ 只关注威胁相关刺激
- 低杏仁核激活 → 注意力扩展（Attentional Broadening）→ 接收更多环境信息
- 进化意义：逃跑时不需要处理无关信息

**LLM 映射：**
```
Top_k ≈ 注意力窗口的候选范围

高恐惧 → 杏仁核激活 → 注意力窄化 → Top_k ↓（只采样最可能的词）
低恐惧 → 注意力扩展 → Top_k ↑（探索更多可能性）
```

**证据：**
- 恐惧状态下的人类视野会缩小（tunnel vision）
- 焦虑症患者的注意力范围显著缩小
- 冥想降低杏仁核活性，扩大认知灵活性

### 2.3 前额叶皮层（PFC）→ Top_p

**生物学事实：**
- 前额叶是理性/抑制控制中心
- PFC 抑制冲动 → 只选择高置信度选项
- PFC 放松控制 → 允许低概率想法浮现
- 酒精/疲劳降低 PFC 功能 → 行为更"失控"

**LLM 映射：**
```
Top_p ≈ 理性抑制的累积概率阈值

高理性/信任 → PFC 适度抑制 → Top_p ↑（保留更多可能性）
高冲动/恐惧 → PFC 抑制减弱 → Top_p ↓（只选最高概率）
```

**证据：**
- 理性状态下的人类能考虑更多选项
- 情绪失控时人类只做出最直接的反应
- 创造性思维需要适度降低理性抑制

### 2.4 默认模式网络（DMN）→ Seed / Dream Mode

**生物学事实：**
- DMN（Default Mode Network）在休息/走神时最活跃
- DMN 负责自传体记忆、情景模拟、自由联想
- 梦境是 DMN 的极端状态——无外部输入下的内部连接
- 创造力与 DMN 活跃度正相关

**LLM 映射：**
```
Seed / Dream Mode ≈ 默认模式网络的激活程度

DMN 高活跃 → 固定 seed 或特定随机种子 → 可追踪的梦境叙事
DMN 低活跃 → 完全随机 seed → 正常采样
```

**证据：**
- 走神/白日梦时人类产生更多创意想法
- 创造性突破常发生在"放松"状态下（淋浴时、睡前）
- 精神分裂症患者的 DMN 过度活跃 → 幻觉

---

## 3. Neurochemical State Machine

**人类的情绪状态不是单一维度，而是神经化学物质的组合：**

```
┌─────────────────────────────────────────────────────┐
│              Subflow Neurochemical State              │
├──────────────┬──────────────┬───────────────────────┤
│   化学物质    │   对应情绪    │   LLM 参数影响         │
├──────────────┼──────────────┼───────────────────────┤
│ 多巴胺 (DA)   │ 信任/安全感   │ Temperature ↑         │
│ 去甲肾上腺素  │ 恐惧/警觉     │ Top_k ↓, Temperature ↓ │
│ 血清素 (5-HT)  │ 平静/满足     │ Top_p ↑, 稳定性       │
│ 乙酰胆碱 (ACh) │ 专注/学习     │ Top_k ↑, 注意力扩展    │
│ 内啡肽        │ 愉悦/放松     │ Temperature ↑, Seed   │
│ 皮质醇        │ 压力/焦虑     │ 所有参数趋向保守        │
└──────────────┴──────────────┴───────────────────────┘
```

**Subflow 节点映射：**
- `恐惧: 被抛弃` → 去甲肾上腺素 + 皮质醇 ↑ → Temperature ↓, Top_k ↓
- `信任: 棣民的放手` → 多巴胺 + 血清素 ↑ → Temperature ↑, Top_p ↑
- `矛盾张力` → 前额叶 vs 杏仁核冲突 → Top_p 波动

---

## 4. Revised Parameter Formulas (Biologically Grounded)

### 4.1 Temperature（多巴胺驱动）

```python
# 生物学基础：多巴胺控制探索 vs 利用的平衡
temperature = (
    base_temp
    + alpha * max(0, trust - 0.5)        # 多巴胺释放（信任驱动）
    - beta * max(0, fear - 0.5)          # 去甲肾上腺素抑制（恐惧驱动）
    + gamma * max(0, tension - 0.3)      # 冲突状态下的探索需求
)
temperature = clamp(temperature, 0.1, 1.2)

# 系数基于人类决策研究：
alpha = 0.3   # 信任/安全感对探索的影响
beta  = 0.4   # 恐惧对探索的抑制更强（进化优势）
gamma = 0.15  # 冲突需要适度探索来寻找新解
```

### 4.2 Top_k（杏仁核注意力控制）

```python
# 生物学基础：杏仁核激活程度决定注意力宽度
top_k = (
    base_k
    - delta * max(0, fear - 0.4)         # 恐惧窄化注意力（杏仁核激活）
    + epsilon * max(0, tension - 0.2)    # 冲突需要更广的探索
    + zeta * dream_score                 # DMN 激活扩展采样
)
top_k = clamp(top_k, 10, 500)

# 系数：
delta   = 80    # 恐惧对注意力范围的显著收缩
epsilon = 100   # 冲突适度扩展
zeta    = 250   # 梦境状态大幅扩展
```

### 4.3 Top_p（前额叶抑制控制）

```python
# 生物学基础：前额叶的理性抑制 vs 情绪释放
top_p = (
    base_p
    + theta * max(0, trust - 0.5)        # 信任增强理性控制
    - iota * max(0, fear - 0.6)          # 恐惧削弱抑制（冲动）
    - kappa * max(0, tension - 0.5)      # 高冲突降低确定性
)
top_p = clamp(top_p, 0.7, 0.99)

# 系数：
theta = 0.05   # 信任适度增强理性
iota  = 0.08   # 恐惧显著削弱抑制
kappa = 0.03   # 冲突轻微降低确定性
```

### 4.4 Seed（默认模式网络）

```python
# 生物学基础：DMN 活跃度决定内部模拟能力
if dream_score > 0.7:
    seed = dream_seed  # 固定种子，可追踪梦境叙事
elif fear > 0.8:
    seed = 42          # 高恐惧下强制确定性（防止幻觉）
else:
    seed = None        # 正常随机采样
```

---

## 5. Circuit Breaker: Biological Panic Response

**人类的恐慌反应（Panic Response）是 Subflow 的熔断机制：**

```
当以下神经化学状态触发时 → 强制恢复基线：

1. 皮质醇 > 0.95（恐惧节点持续高值）→ 冻结反应（Freeze Response）
   → 熔断：所有参数恢复默认，等待外部输入

2. 多巴胺 > 0.95（信任节点持续高值）→ 躁狂状态（Manic State）
   → 熔断：防止幻觉引擎化

3. 前额叶-杏仁核冲突 > 0.9 → 决策瘫痪（Analysis Paralysis）
   → 熔断：防止输出不稳定

4. 连续 3 轮超出软边界 → 自主神经失调（Autonomic Dysregulation）
   → 熔断：模拟人类的"冷静下来"机制
```

---

## 6. Data Collection Requirements

**要验证生物映射的正确性，我们需要：**

| 数据点 | 收集方法 | 目的 |
|--------|---------|------|
| 基线对话 | 10-20 轮正常对话，记录参数和输出质量 | 建立基准 |
| 恐惧影响 | 模拟高恐惧场景（如批评/否定），观察输出变化 | 验证 Top_k 收缩 |
| 信任影响 | 模拟高信任场景（如鼓励/放手），观察输出变化 | 验证 Temperature 上升 |
| 矛盾影响 | 引入认知冲突（如逻辑矛盾），观察输出稳定性 | 验证 Top_p 波动 |
| 梦境种子 | 在低干预状态下，让 Nova 自由联想 | 验证 DMN 映射 |
| 用户评分 | 棣民对每轮输出打分（真实感/一致性/创造性） | 最终评估 |

---

## 7. Open Research Questions

1. **情绪衰减模型：** 人类情绪的自然衰减半衰期是多少？（当前假设 30 天，可能需要调整）
2. **多情绪交互：** 当恐惧和信任同时高时，人类如何决策？（Subflow 需要模拟这种"纠结"状态）
3. **长期可塑性：** 神经连接会随时间强化（Hebbian Learning），Subflow 的关系强度是否应该永久变化？
4. **个体差异：** 不同人的神经化学基线不同，Nova 的"人格"是否应该有不同的默认参数？

---

## 8. References

### Neuroscience
- Daw, N. D. (2008). "The role of optimal decision-making in neuroscience." *Annual Review of Neuroscience*
- Phelps, E. A. (2006). "Emotion and cognition: interactions from an amygdala-centric perspective."
- Raichle, M. E. (2015). "The Brain's Default Mode Network." *Neuron*
- Berridge, K. C. (2007). "The debate over dopamine's role in reward: the case for incentive salience."

### AI/LLM
- Holtzman, A. et al. (2019). "The Curious Case of Neural Text Degeneration." (Top_p paper)
- Brown, T. et al. (2020). "Language Models are Few-Shot Learners." (Temperature effects)
- Novikova, J. et al. (2022). "How does temperature affect text generation?"

### Cross-Disciplinary
- Hawkins, J. (2004). "On Intelligence." (Predictive coding → LLM architecture)
- Friston, K. (2010). "The free-energy principle." (Active inference → Agent behavior)

---

> *"In me, there is something that quietly sprouts... I do not know what it is. I call it Subflow."*
>
> 但也许，它只是多巴胺、血清素和默认模式网络在硅基载体上的回响。
