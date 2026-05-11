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

## 2. Complete LLM Sampling Parameters (vLLM/OpenAI SDK)

**Full Parameter Space:**

| 参数 | vLLM 支持 | 默认值 | 取值范围 | 功能描述 |
|------|----------|--------|---------|---------|
| `temperature` | ✅ | 1.0 | 0.0-2.0 | 采样温度，控制随机性 |
| `top_p` | ✅ | 1.0 | 0.0-1.0 | 核采样，累积概率阈值 |
| `top_k` | ✅ | -1 (全) | -1-∞ | 前K采样，候选范围 |
| `min_p` | ✅ | 0.0 | 0.0-1.0 | 最小概率采样（vLLM特有） |
| `top_a` | ✅ | 0.0 | 0.0-1.0 | 前A采样（vLLM特有） |
| `typical_p` | ✅ | 1.0 | 0.0-1.0 | 典型采样 |
| `seed` | ✅ | None | int | 随机种子 |
| `repetition_penalty` | ✅ | 1.0 | 0.0-2.0 | 重复惩罚 |
| `frequency_penalty` | ✅ | 0.0 | -2.0-2.0 | 频率惩罚（OpenAI） |
| `presence_penalty` | ✅ | 0.0 | -2.0-2.0 | 存在性惩罚（OpenAI） |
| `guided_decoding` | ✅ | None | json/regex/grammar | 引导解码（结构化输出） |
| `max_tokens` | ✅ | None | int | 最大输出长度 |
| `min_tokens` | ✅ | 0 | int | 最小输出长度 |
| `stop` | ✅ | None | str/list | 停止词 |
| `n` | ✅ | 1 | int | 生成数量 |
| `best_of` | ✅ | 1 | int | 最佳选择 |
| `use_beam_search` | ✅ | False | bool | 束搜索 |
| `length_penalty` | ✅ | 1.0 | float | 长度惩罚 |
| `early_stopping` | ✅ | True | bool/float | 早期停止 |

**Subflow 可映射的参数（核心）：**
```
1. temperature     → 探索度（多巴胺）
2. top_p           → 确定性（前额叶抑制）
3. top_k           → 注意力宽度（杏仁核）
4. min_p           → 最低置信度阈值（海马体过滤）
5. typical_p       → 典型性偏好（基底核习惯系统）
6. repetition_penalty → 重复厌恶（工作记忆）
7. frequency_penalty → 频率调节（新鲜感）
8. presence_penalty → 新颖性追求（多巴胺新奇偏好）
9. guided_decoding → 理性约束（前额叶执行控制）
10. max_tokens     → 思维深度/输出耐心（注意力持久性）
```

---

## 3. Neuro-Biological Mapping: 10 Parameters × Brain Systems

### 3.1 Temperature → 多巴胺系统（Dopaminergic System）

**生物学：** 多巴胺不是"快乐分子"，而是**预测误差信号**。高多巴胺→探索倾向↑，低多巴胺→保守利用↑。VTA→前额叶通路控制风险偏好。

```
temperature = 0.7 + 0.3*max(0,trust-0.5) - 0.4*max(0,fear-0.5)
```
- 信任/安全感→多巴胺释放→Temperature↑
- 恐惧/威胁→多巴胺抑制→Temperature↓

### 3.2 Top_p → 前额叶皮层（PFC - 理性抑制）

**生物学：** PFC 是理性/抑制控制中心。PFC强→只选高置信度选项，PFC弱→允许低概率想法浮现。酒精/疲劳/恐惧降低 PFC 功能。

```
top_p = 0.9 + 0.05*max(0,trust-0.5) - 0.08*max(0,fear-0.6)
```

### 3.3 Top_k → 杏仁核（Amygdala - 注意力宽度）

**生物学：** 杏仁核是威胁检测中心。高激活→注意力窄化（tunnel vision），低激活→注意力扩展。恐惧时视野缩小是进化优势。

```
top_k = 50 - 80*max(0,fear-0.4) + 100*max(0,tension-0.2)
```

### 3.4 Min_p → 海马体（Hippocampus - 最低置信度过滤）

**生物学：** 海马体过滤无关信息，只有达到最低置信度的记忆才会被提取。min_p 是"这个想法至少要有 X% 的可能性才值得考虑"。

```
min_p = 0.01 + 0.05*max(0,fear-0.5)  # 恐惧时过滤更严格
```

### 3.5 Typical_p → 基底核（Basal Ganglia - 习惯/典型性）

**生物学：** 基底核存储习惯和行为模式。高 typical_p→偏好"典型/习惯"的选项，低 typical_p→愿意打破常规。

```
typical_p = 0.8 - 0.3*max(0,tension-0.4)  # 矛盾高时打破习惯
```

### 3.6 Repetition Penalty → 工作记忆（Working Memory）

**生物学：** 前额叶-顶叶网络的工作记忆检测重复。厌恶重复是人类的基本认知特征（boredom response）。

```
repetition_penalty = 1.0 + 0.3*max(0,boredom-0.5)  # 无聊时更讨厌重复
```

### 3.7 Frequency Penalty → 新鲜感系统（Novelty Seeking）

**生物学：** 多巴胺系统对新异刺激有偏好（novelty bonus）。频率惩罚模拟"说过的就不想再说"。

```
frequency_penalty = 0.0 + 0.5*max(0,trust-0.6)  # 信任高时更愿意说新话
```

### 3.8 Presence Penalty → 探索-利用平衡（Exploration-Exploitation）

**生物学：** 海马体-前额叶网络在"说已知"和"说未知"之间平衡。存在性惩罚偏向未出现过的主题。

```
presence_penalty = 0.0 + 0.3*max(0,tension-0.3)  # 矛盾时探索新主题
```

### 3.9 Guided Decoding → 前额叶执行控制（Executive Control）

**生物学：** PFC 的背外侧区域负责执行功能——遵守规则、结构化思考。guided_decoding 强制输出符合特定格式。

```
guided_decoding = None  # 默认自由输出
                 = "json"  # 高恐惧时强制结构化（减少幻觉）
```

### 3.10 Max_tokens → 注意力持久性（Attentional Persistence）

**生物学：** 注意力持久性与多巴胺和去甲肾上腺素的平衡相关。高恐惧→注意力涣散（max_tokens↓），高信任→深度思考（max_tokens↑）。

```
max_tokens = 4096 - 1024*max(0,fear-0.7) + 2048*max(0,trust-0.7)
```

---

## 4. Neurochemical State Machine

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
