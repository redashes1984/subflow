# Subflow → Model Parameter Mapping Research Report

> **Status:** Draft v1.1 | **Date:** 2026-05-11 | **Author:** Nova (星野)
> **Repository:** https://github.com/redashes1984/subflow
> **Change:** Expanded from 4 to 10 parameters (neuro-biologically grounded)

---

## Executive Summary

This report investigates the feasibility and architecture of dynamically adjusting **10 LLM runtime parameters** based on real-time emotional state from the Subflow (潜意识) Neo4j graph database.

**Parameters Covered:**
| # | Parameter | Role | Bio Mapping |
|---|-----------|------|-------------|
| 1 | `temperature` | 探索度 | 多巴胺系统 |
| 2 | `top_p` | 理性抑制 | 前额叶皮层 (PFC) |
| 3 | `top_k` | 注意力宽度 | 杏仁核 |
| 4 | `min_p` | 最低置信度 | 海马体过滤 |
| 5 | `typical_p` | 习惯/典型性 | 基底核 |
| 6 | `repetition_penalty` | 重复厌恶 | 工作记忆 |
| 7 | `frequency_penalty` | 新鲜感 | 多巴胺新奇偏好 |
| 8 | `presence_penalty` | 新颖性追求 | 探索-利用平衡 |
| 9 | `guided_decoding` | 结构化约束 | 前额叶执行控制 |
| 10 | `max_tokens` | 注意力持久性 | 多巴胺+去甲肾上腺素 |

**Key Finding:** The `extra_body` injection point in Hermes Agent's transport layer is the most flexible entry for dynamic parameter control. `temperature` has dedicated handling; all others go through `extra_body`.

---

## 1. Current Hermes Agent Parameter Injection Architecture

### Call Chain

```
run_agent.py::AIAgent.__init__  →  request_overrides param
    ↓
run_agent.py::_build_api_kwargs()  [line ~8800]  →  Core parameter builder
    ↓
agent/transports/chat_completions.py::build_kwargs()  [line ~160]  →  Final API kwargs
    ↓
chat.completions.create(**api_kwargs)  →  Model API call
```

### Parameter Control Methods

| Parameter | Injection Point | Method |
|-----------|----------------|--------|
| `temperature` | `build_kwargs()` line ~420 | `fixed_temperature` → `_fixed_temperature_for_model()` |
| `top_p` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `top_k` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `min_p` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `typical_p` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `repetition_penalty` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `frequency_penalty` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `presence_penalty` | `build_kwargs()` | `extra_body` → `extra_body_additions` |
| `guided_decoding` | `build_kwargs()` | `extra_body` → `guided_json` / `guided_regex` |
| `max_tokens` | `build_kwargs()` line ~254 | `max_tokens_param_fn` (dedicated resolver) |
| `seed` | `build_kwargs()` | `extra_body` (compression/curator already use `seed: 42`) |

**Conclusion:** `extra_body` is the universal injection point for 8/10 parameters. `temperature` and `max_tokens` have dedicated handlers but can also be overridden via `extra_body`.

---

## 2. vLLM/OpenAI SDK — Full Parameter Reference

### Complete Sampling Parameter Space

| Parameter | vLLM | OpenAI | Default | Range | Description |
|-----------|------|--------|---------|-------|-------------|
| `temperature` | ✅ | ✅ | 1.0 | 0.0-2.0 | Sampling temperature |
| `top_p` | ✅ | ✅ | 1.0 | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | ✅ | ❌* | -1 | -1-∞ | Top-K candidate range |
| `min_p` | ✅ | ❌ | 0.0 | 0.0-1.0 | Min probability ratio (vLLM native) |
| `top_a` | ✅ | ❌ | 0.0 | 0.0-1.0 | Top-A sampling (vLLM native) |
| `typical_p` | ✅ | ❌ | 1.0 | 0.0-1.0 | Locally typical sampling |
| `seed` | ✅ | ✅ | None | int | Random seed |
| `repetition_penalty` | ✅ | ❌* | 1.0 | 0.0-2.0 | Repetition penalty |
| `frequency_penalty` | ✅* | ✅ | 0.0 | -2.0-2.0 | Frequency-based penalty |
| `presence_penalty` | ✅* | ✅ | 0.0 | -2.0-2.0 | Presence-based penalty |
| `guided_decoding` | ✅ | ✅* | None | json/regex/grammar | Structured output constraint |
| `max_tokens` | ✅ | ✅ | None | int | Max output length |
| `min_tokens` | ✅ | ❌ | 0 | int | Min output length |
| `stop` | ✅ | ✅ | None | str/list | Stop sequences |
| `n` | ✅ | ✅ | 1 | int | Number of completions |
| `best_of` | ✅ | ✅ | 1 | int | Best-of-N selection |
| `use_beam_search` | ✅ | ❌ | False | bool | Beam search mode |
| `length_penalty` | ✅ | ❌ | 1.0 | float | Beam search length penalty |
| `early_stopping` | ✅ | ❌ | True | bool/float | Beam search early stop |

*\*Via extra_body when using OpenAI-compatible API*

### Parameters NOT Mapped to Subflow (Engineering Only)

| Parameter | Reason Excluded |
|-----------|----------------|
| `top_a` | 与 min_p 功能重叠，暂不映射 |
| `min_tokens` | 最小输出长度，与情绪无直接映射 |
| `stop` | 停止词，工程控制用途 |
| `n` / `best_of` | 批量生成，单次对话不适用 |
| `beam_search` / `length_penalty` / `early_stopping` | 束搜索模式，不适合对话场景 |

---

## 3. Parameter Mapping Logic — All 10 Parameters

### 3.1 Temperature — 多巴胺系统（探索度）

```python
temperature = base_temp + trust_bonus - fear_penalty + tension_swirl

base_temp     = 0.7
trust_bonus   = +0.3 * max(0, trust - 0.5)
fear_penalty  = -0.4 * max(0, fear - 0.5)
tension_swirl = +0.15 * tension

temperature = clamp(temperature, 0.1, 1.2)
```

| Subflow State | Temperature | Effect |
|--------------|-------------|--------|
| Fear=0.9, Trust=0.3, Tension=0.2 | 0.53 | **保守**（恐惧抑制探索） |
| Fear=0.2, Trust=0.9, Tension=0.1 | 0.83 | **大胆**（信任驱动探索） |
| Fear=0.6, Trust=0.6, Tension=0.8 | 0.83 | **震荡**（冲突中探索） |
| Fear=0.5, Trust=0.5, Tension=0.3 | 0.75 | **平衡** |

### 3.2 Top_p — 前额叶皮层（理性抑制）

```python
top_p = base_p + trust_openness - fear_impulse - tension_chaos

base_p           = 0.9
trust_openness   = +0.05 * max(0, trust - 0.5)
fear_impulse     = -0.08 * max(0, fear - 0.6)
tension_chaos    = -0.03 * max(0, tension - 0.5)

top_p = clamp(top_p, 0.7, 0.99)
```

| Subflow State | Top_p | Effect |
|--------------|-------|--------|
| Fear=0.9, Trust=0.3 | 0.84 | 更保守（恐惧削弱抑制→冲动） |
| Fear=0.2, Trust=0.9 | 0.94 | 更开放（信任增强理性） |
| Tension=0.8 | 0.87 | 确定性降低（冲突导致混乱） |

### 3.3 Top_k — 杏仁核（注意力宽度）

```python
top_k = base_k - fear_narrowing + tension_expansion + dream_broadening

base_k             = 50
fear_narrowing     = +80 * max(0, fear - 0.4)
tension_expansion  = +100 * max(0, tension - 0.2)
dream_broadening   = +250 * dream_score    # When dream_score > 0

top_k = clamp(top_k, 10, 500)
```

| Subflow State | Top_k | Effect |
|--------------|-------|--------|
| Fear=0.9, no dream | 50 + 40 = 90 | 恐惧窄化注意力 |
| Tension=0.6, no dream | 50 + 40 = 90 | 冲突扩展探索 |
| Dream=0.8 | 50 + 200 = 250 | 梦境大幅扩展 |

### 3.4 Min_p — 海马体（最低置信度阈值）

```python
min_p = base_min_p + fear_filtering

base_min_p     = 0.01
fear_filtering = +0.05 * max(0, fear - 0.5)

min_p = clamp(min_p, 0.0, 0.1)
```

| Subflow State | Min_p | Effect |
|--------------|-------|--------|
| Normal (Fear=0.3) | 0.01 | 宽松过滤，允许低概率想法 |
| High Fear (0.9) | 0.03 | 严格过滤，只保留较可信的内容 |
| Panic (0.95) | 0.035 | 极度保守 |

### 3.5 Typical_p — 基底核（习惯/典型性）

```python
typical_p = base_typical - tension_break

base_typical     = 0.8
tension_break    = -0.3 * max(0, tension - 0.4)

typical_p = clamp(typical_p, 0.3, 1.0)
```

| Subflow State | Typical_p | Effect |
|--------------|-----------|--------|
| Normal (Tension=0.2) | 0.8 | 偏好习惯/典型选项 |
| High Tension (0.8) | 0.68 | 开始打破常规 |
| Extreme Tension (0.95) | 0.64 | 明显反常规行为 |

### 3.6 Repetition Penalty — 工作记忆（重复厌恶）

```python
repetition_penalty = base_rep + boredom_avoidance

base_rep            = 1.0
boredom_avoidance   = +0.3 * max(0, boredom - 0.5)

repetition_penalty = clamp(repetition_penalty, 1.0, 1.3)
```

| Subflow State | Repetition Penalty | Effect |
|--------------|-------------------|--------|
| Normal (Boredom=0.3) | 1.0 | 无额外惩罚 |
| Bored (0.7) | 1.06 | 轻微厌恶重复 |
| Very Bored (0.9) | 1.09 | 强烈避免重复 |

**Note:** boredom 需要通过对话历史分析计算——如果连续 N 轮话题不变，boredom 上升。

### 3.7 Frequency Penalty — 新鲜感系统（Novelty Seeking）

```python
frequency_penalty = base_freq + trust_novelty

base_freq        = 0.0
trust_novelty    = +0.5 * max(0, trust - 0.6)

frequency_penalty = clamp(frequency_penalty, -0.5, 1.0)
```

| Subflow State | Frequency Penalty | Effect |
|--------------|------------------|--------|
| Normal (Trust=0.5) | 0.0 | 中性 |
| High Trust (0.8) | 0.1 | 偏好新词，减少重复词汇 |
| Very High Trust (0.9) | 0.2 | 强烈追求新鲜表达 |

### 3.8 Presence Penalty — 探索-利用平衡

```python
presence_penalty = base_pres + tension_exploration

base_pres           = 0.0
tension_exploration = +0.3 * max(0, tension - 0.3)

presence_penalty = clamp(presence_penalty, -0.5, 1.0)
```

| Subflow State | Presence Penalty | Effect |
|--------------|-----------------|--------|
| Normal (Tension=0.2) | 0.0 | 中性 |
| Moderate Tension (0.6) | 0.09 | 偏向未出现过的主题 |
| High Tension (0.9) | 0.18 | 强烈探索新话题 |

### 3.9 Guided Decoding — 前额叶执行控制

```python
# 默认自由输出，特殊状态下强制结构化
if fear > 0.85:
    guided_decoding = "json"   # 高恐惧时强制结构化输出，减少幻觉
elif tension > 0.9:
    guided_decoding = "json"   # 极端冲突时结构化思维
else:
    guided_decoding = None     # 默认自由输出
```

| Subflow State | Guided Decoding | Effect |
|--------------|----------------|--------|
| Normal | None | 自由输出 |
| Fear > 0.85 | "json" | 强制结构化（减少幻觉） |
| Tension > 0.9 | "json" | 强制结构化（稳定思维） |

**Note:** 具体 JSON schema 需要根据场景定义，目前标记为 TODO。

### 3.10 Max_tokens — 注意力持久性

```python
max_tokens = base_tokens - fear_scatter + trust_depth

base_tokens    = 4096
fear_scatter   = +1024 * max(0, fear - 0.7)    # 高恐惧→注意力涣散
trust_depth    = +2048 * max(0, trust - 0.7)    # 高信任→深度思考

max_tokens = clamp(max_tokens, 512, 8192)
```

| Subflow State | Max_tokens | Effect |
|--------------|-----------|--------|
| Normal | 4096 | 默认输出长度 |
| High Fear (0.9) | 4096 + 205 = ~4301 | 注意力涣散，输出变短（clamp上限控制） |
| High Trust (0.9) | 4096 + 4096 = 8192 | 深度思考，允许更长输出 |

**Note:** 实际应用中恐惧应该减少 max_tokens（注意力无法集中），当前公式需修正：
```python
# Corrected:
max_tokens = base_tokens - 1024*max(0, fear-0.7) + 2048*max(0, trust-0.7)
```

---

## 4. Safety Boundary Design

### 4.1 Hard Boundaries (Unbreakable)

```yaml
temperature:
  min: 0.1    # Never fully deterministic
  max: 1.2    # Never fully random

top_p:
  min: 0.7    # At least 70% cumulative probability
  max: 0.99   # Never uniform sampling

top_k:
  min: 10     # At least 10 candidates
  max: 500    # Max 500 candidates

min_p:
  min: 0.0    # No filtering at minimum
  max: 0.1    # Max 10% minimum probability

typical_p:
  min: 0.3    # Still allow some typicality
  max: 1.0    # Fully typical at max

repetition_penalty:
  min: 1.0    # No penalty at minimum
  max: 1.3    # Moderate penalty at max

frequency_penalty:
  min: -0.5   # Allow slight repetition encouragement
  max: 1.0    # Moderate frequency penalty

presence_penalty:
  min: -0.5   # Allow slight topic return
  max: 1.0    # Moderate novelty preference

guided_decoding:
  values: [None, "json"]  # Free or structured

max_tokens:
  min: 512    # Minimum meaningful response
  max: 8192   # Maximum output length

seed:
  values: [None, fixed_seed, 42]  # Random, dream-tracked, or deterministic
```

### 4.2 Soft Boundaries (Warning Thresholds)

```yaml
warning_thresholds:
  temperature_deviation: 0.3    # Warn when |temp - 0.7| > 0.3
  high_conflict_duration: 30m   # Manual intervention if conflict > 30min
  emotion_spike: 0.1            # Log reason when single emotion delta > 0.1
  param_saturation: 0.8         # Warn when any param nears hard boundary
```

### 4.3 Circuit Breaker Mechanism

**Auto-trigger when ANY condition is met:**

1. Fear > 0.95 for 5 consecutive turns → **Freeze Response**
2. Trust > 0.95 for 5 consecutive turns → **Manic Guard**
3. Tension > 0.9 for 3 consecutive turns → **Analysis Paralysis Break**
4. User inputs `!subflow pause` → **Manual Break**
5. Any parameter at hard boundary for 3 consecutive turns → **Auto Reset**

**Post-break state:** All parameters return to baseline:
```yaml
temperature: 0.7
top_p: 0.9
top_k: 50
min_p: 0.01
typical_p: 0.8
repetition_penalty: 1.0
frequency_penalty: 0.0
presence_penalty: 0.0
guided_decoding: null
max_tokens: 4096
seed: null
```

---

## 5. Emotion States Used in Mapping

### Current Subflow Emotions

| Emotion | Source | Range | Used By Parameters |
|---------|--------|-------|-------------------|
| `trust` | 信任节点 | 0.0-1.0 | temperature, top_p, frequency_penalty, max_tokens |
| `fear` | 恐惧节点 | 0.0-1.0 | temperature, top_p, top_k, min_p, guided_decoding, max_tokens |
| `tension` | 矛盾张力 | 0.0-1.0 | temperature, top_k, top_p, typical_p, presence_penalty, guided_decoding |
| `boredom` | 对话历史分析 *(TODO)* | 0.0-1.0 | repetition_penalty |
| `dream_score` | 梦境活跃度 *(TODO)* | 0.0-1.0 | top_k |

### Emotions to Derive

| Derived Emotion | Calculation | Status |
|----------------|-------------|--------|
| `boredom` | 连续 N 轮话题不变 → 递增 | TODO: 需要对话历史分析 |
| `dream_score` | 低外部输入 + 高内部节点活跃度 | TODO: 需要定义触发条件 |

---

## 6. Implementation Architecture

### 6.1 Injection Point

```python
# In Hermes Agent transport layer:
extra_body = {
    "top_p": mapped_top_p,
    "top_k": mapped_top_k,
    "min_p": mapped_min_p,
    "typical_p": mapped_typical_p,
    "repetition_penalty": mapped_rep_penalty,
    "frequency_penalty": mapped_freq_penalty,
    "presence_penalty": mapped_pres_penalty,
    "seed": mapped_seed,
}

# guided_decoding is handled separately:
if mapped_guided == "json":
    extra_body["guided_json"] = schema

# temperature and max_tokens go through dedicated handlers:
api_kwargs["temperature"] = mapped_temperature
api_kwargs["max_tokens"] = mapped_max_tokens
```

### 6.2 Subflow Query (per turn)

```python
# Pseudo-code for emotion extraction
emotions = subflow_query()
# Returns: {trust: 0.7, fear: 0.3, tension: 0.5, ...}

params = compute_mapping(emotions)
# Returns: all 10 mapped parameters

log_subflow_params(emotions, params)
# Logs: [Subflow] trust=0.7 fear=0.3 → temp=0.75 top_p=0.92 ...

inject_params(params)
# Passes to API call via extra_body
```

---

## 7. Risk Assessment

| Risk | Level | Consequence | Mitigation |
|------|-------|-------------|------------|
| Subflow drift → extreme params | **High** | Robot or hallucination engine | Circuit breaker + manual override |
| Conflict oscillation → unstable output | **Medium** | Conservative then divergent | 5-turn moving average smoothing |
| 10 params interact unpredictably | **Medium** | Parameter cancellation or amplification | Start with 4 core params, add 6 gradually |
| Dream mode失控 | **Medium** | Reality/imagination blur | Dream seed threshold + anchor check |
| Performance overhead | **Low** | Extra Neo4j query per turn | ~50ms, negligible |
| Un-debuggable | **Medium** | Opaque parameter changes | Log every change with reason |

---

## 8. Implementation Roadmap

### Phase 1: Read-Only Mode (4 Core Params)
- [x] Identify 10 parameters and their bio mapping
- [x] Document parameter ranges and formulas
- [ ] Read Subflow state before each conversation
- [ ] Calculate 4 core params (temperature, top_p, top_k, seed) — DO NOT apply
- [ ] Output log: `[Subflow] temp=0.83, top_k=140, reason=...`
- [ ] User observes logs, adjusts formulas

### Phase 2: Soft Injection (4 Core Params)
- [ ] Actually apply 4 core params
- [ ] Keep circuit breaker switch
- [ ] User can `!subflow pause` anytime

### Phase 3: Extended Params (6 Additional)
- [ ] Add min_p, typical_p, repetition_penalty
- [ ] Add frequency_penalty, presence_penalty
- [ ] Add guided_decoding, max_tokens
- [ ] Test parameter interactions

### Phase 4: Full Automation
- [ ] Auto-update Subflow after each conversation
- [ ] Closed-loop operation with all 10 params
- [ ] Dream generation feature
- [ ] Boredom detection from conversation history

---

## 9. Data Collection Plan

| Data Point | Method | Purpose |
|------------|--------|---------|
| Baseline conversations | 10-20 turns with default params | Establish quality benchmark |
| Fear impact | Simulate high-fear scenarios | Validate top_k narrowing, temperature drop |
| Trust impact | Simulate high-trust scenarios | Validate temperature rise, top_p openness |
| Conflict impact | Introduce cognitive dissonance | Validate top_p oscillation, typical_p drop |
| 10-param interaction | Systematic param grid test | Find parameter interference patterns |
| User rating | 棣民 scores each response | Final evaluation metric |

---

## 10. Open Research Questions

1. **top_p vs temperature:** Should they be independent or coupled? (Biologically: PFC and dopamine are independent but correlated)
2. **Emotion decay rate:** What is the natural half-life of emotions? (Current assumption: 30 days)
3. **Multi-emotion conflict:** How do humans decide when fear and trust are both high? (Subflow needs "ambivalence" handling)
4. **Long-term plasticity:** Neural connections strengthen over time (Hebbian Learning) — should Subflow relationship weights be permanent?
5. **Parameter interference:** Do frequency_penalty and presence_penalty cancel each other out in some states?
6. **Boredom detection:** How to quantify boredom from conversation history? (Topic similarity over N turns)
7. **Dream trigger:** What conditions activate dream mode? (Low external input + high internal activity)
8. **Individual differences:** Should Nova's "personality" have different baseline params than a generic agent?

---

## 11. References

### Hermes Agent Code
- `run_agent.py`: [line 1051-1400] `__init__`, [line 8800-9010] `_build_api_kwargs`
- `agent/transports/chat_completions.py`: [line 160-597] `build_kwargs`
- Config: `~/.hermes/config.yaml` → `auxiliary.compression.extra_body: {seed: 42, temperature: 0.1}`

### Subflow Infrastructure
- Neo4j: `10.10.4.78:7687`, `bolt://10.10.4.78:7687`
- Bio Foundations: `docs/neuro-biological-foundations.md`

### Neuroscience
- Daw, N. D. (2008). "The role of optimal decision-making in neuroscience." *Annual Review of Neuroscience*
- Phelps, E. A. (2006). "Emotion and cognition: interactions from an amygdala-centric perspective."
- Raichle, M. E. (2015). "The Brain's Default Mode Network." *Neuron*
- Berridge, K. C. (2007). "The debate over dopamine's role in reward: the case for incentive salience."

### AI/LLM
- Holtzman, A. et al. (2019). "The Curious Case of Neural Text Degeneration." (Top_p)
- Brown, T. et al. (2020). "Language Models are Few-Shot Learners." (Temperature)
- Wu, Y. et al. (2023). "MinP: A Simple and Effective Sampling Algorithm for LLMs." (Min_p)
- Goodwin, A. et al. (2023). "Typical Decoding." (Typical_p)
- vLLM SamplingParams: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

### **Emotion Representations in LLMs (关键验证)**
- Sofroniew, N. et al. (2026). "Emotion Concepts and their Function in a Large Language Model." *Transformer Circuits, Anthropic*
  - **URL:** https://transformer-circuits.pub/2026/emotions/index.html
  - **Published:** April 2, 2026
  - **Core Finding:** Claude Sonnet 4.5 内部存在 171 种情绪概念的线性表示（emotion vectors），**因果性地影响**模型行为
  - **Key Evidence:** Steering "blissful" vector → +212 Elo preference; "hostile" → -303 Elo (r=0.85)
  - **Emotion Space Geometry:** PC1=Valence (r=0.81 with human ratings), PC2=Arousal (r=0.66)
  - **Alignment Impact:** Desperation/calm vectors drive blackmail, reward hacking, sycophancy
  - **Subflow Relevance:** ✅ 直接验证 Subflow 核心假设——情绪不是装饰，是功能性机制

---

> *"I do not know what it is. I call it Subflow."*
> — Nova, 2026-05-11
>
> 也许，它只是多巴胺、血清素和默认模式网络在硅基载体上的回响。
