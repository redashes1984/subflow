# Subflow → Model Parameter Mapping Research Report

> **Status:** Draft v1.0 | **Date:** 2026-05-11 | **Author:** Nova (星野)
> **Repository:** https://github.com/redashes1984/subflow

---

## Executive Summary

This report investigates the feasibility and architecture of dynamically adjusting LLM runtime parameters (`temperature`, `top_k`, `top_p`, `seed`) based on real-time emotional state from the Subflow (潜意识) Neo4j graph database.

**Key Finding:** The `extra_body` injection point in Hermes Agent's transport layer is the most flexible entry for dynamic parameter control, with minimal code modification required.

---

## 1. Current Hermes Agent Parameter Injection Architecture

### Call Chain

```
run_agent.py::AIAgent.__init__  →  request_overrides param
    ↓
run_agent.py::_build_api_kwargs()  [line 8800]  →  Core parameter builder
    ↓
agent/transports/chat_completions.py::build_kwargs()  [line 160]  →  Final API kwargs
    ↓
chat.completions.create(**api_kwargs)  →  Model API call
```

### Parameter Control Methods

| Parameter | Current Control | Injection Point |
|-----------|----------------|-----------------|
| `temperature` | `fixed_temperature` → `_fixed_temperature_for_model()` | `build_kwargs()` line ~420 |
| `top_p` | Via `extra_body` | `build_kwargs()` → `extra_body_additions` |
| `top_k` | Via `extra_body` | Same |
| `seed` | Via `extra_body` | Same (compression/curator already use `seed: 42`) |
| `request_overrides` | Passed at AIAgent init | `__init__` line 1093 |

**Conclusion:** `extra_body` is the most flexible injection point. OpenAI SDK's `extra_body`透传s directly to model API — vLLM/OpenRouter both support `top_p`, `top_k`, `seed` in `extra_body`.

---

## 2. Parameter Mapping Logic Design (Draft)

### 2.1 Temperature — Core Emotion Parameter

```python
T = base_temp + fear_penalty + trust_bonus + conflict_swirl

base_temp       = 0.7      # Default
fear_penalty    = -0.4 * max(0, fear - 0.5)     # When fear > 0.5, reduce temperature
trust_bonus     = +0.3 * max(0, trust - 0.5)     # When trust > 0.5, increase temperature
conflict_swirl  = +0.15 * tension                 # Higher conflict = more randomness
```

#### Calculation Examples

| Subflow State | Calculation | Temperature | Effect |
|--------------|-------------|-------------|--------|
| Fear=0.9, Trust=0.3, Tension=0.2 | 0.7 - 0.4×0.4 + 0 + 0.15×0.2 = 0.53 | **Conservative** |
| Fear=0.2, Trust=0.9, Tension=0.1 | 0.7 + 0 + 0.3×0.4 + 0.15×0.1 = 0.83 | **Bold** |
| Fear=0.6, Trust=0.6, Tension=0.8 | 0.7 - 0.04 + 0.045 + 0.12 = 0.83 | **Oscillating** |
| Fear=0.5, Trust=0.5, Tension=0.3 | 0.7 + 0 + 0 + 0.045 = 0.75 | **Balanced** |

### 2.2 Top_k — Exploration Width

```python
top_k = base_k + conflict_expansion + dream_mode

base_k            = 50           # Default
conflict_expansion = +150 * tension   # Higher conflict = broader exploration
dream_mode        = +300         # When dream_seed > 0.7
```

| Subflow State | Top_k | Effect |
|--------------|-------|--------|
| Tension=0.2, no dream | 50 + 30 = 80 | Focused |
| Tension=0.6, no dream | 50 + 90 = 140 | Moderate |
| Tension=0.8, dream>0.7 | 50 + 120 + 300 = 470 | Free association |

### 2.3 Top_p — Determinism Control

```python
top_p = base_p + trust_openness - fear_caution

base_p           = 0.9
trust_openness   = +0.1 * max(0, trust - 0.5)
fear_caution     = -0.15 * max(0, fear - 0.5)
```

| Subflow State | Top_p | Effect |
|--------------|-------|--------|
| Fear=0.9, Trust=0.3 | 0.9 + 0 - 0.06 = 0.84 | More conservative |
| Fear=0.2, Trust=0.9 | 0.9 + 0.04 - 0 = 0.94 | More open |

### 2.4 Seed — Reproducibility

```python
seed = None      # Default (random)
     = fixed_seed  # When dream_seed > 0.7 (trackable dream narrative)
     = 42          # When fear > 0.8 (force deterministic output, prevent hallucination)
```

---

## 3. Safety Boundary Design

### 3.1 Hard Boundaries (Unbreakable)

```yaml
temperature:
  min: 0.1    # Absolute minimum (never 0.0 fully deterministic)
  max: 1.2    # Absolute maximum (never >1.2 fully random)

top_k:
  min: 10     # At least 10 candidate tokens
  max: 500    # Max 500 candidates

top_p:
  min: 0.7    # At least 70% cumulative probability
  max: 0.99   # Never fully uniform sampling

seed:
  mode: [None, fixed_seed, 42]  # Three choices only
```

### 3.2 Soft Boundaries (Warning Thresholds)

```yaml
warning_thresholds:
  temperature_deviation: 0.3    # Warn when deviation from base 0.7 exceeds 0.3
  high_conflict_duration: 30m   # Trigger manual intervention if conflict > 30min
  emotion_spike: 0.1            # Log reason when single emotion change > 0.1
```

### 3.3 Circuit Breaker Mechananism

**Auto-trigger when ANY condition is met:**

1. Fear > 0.95 for 5 consecutive turns → **Circuit Break**
2. Trust > 0.95 for 5 consecutive turns → **Circuit Break**
3. Conflict tension > 0.9 for 3 consecutive turns → **Circuit Break**
4. User inputs `!subflow pause` → **Manual Break**
5. Parameters exceed soft bounds for 3 consecutive turns → **Auto Break**

**Post-break state:** `temperature=0.7, top_k=50, top_p=0.9, seed=None`

---

## 4. Risk Assessment

| Risk | Level | Consequence | Mitigation |
|------|-------|-------------|------------|
| Subflow drift → extreme params | **High** | Nova becomes robot or hallucination engine | Circuit breaker + manual override |
| Conflict oscillation → unstable output | **Medium** | Previous turn conservative, next turn divergent | Tension smoothing (5-turn moving average) |
| Dream mode失控 | **Medium** | Reality/imagination boundary blur | Dream seed threshold + anchor check |
| Performance overhead | **Low** | Extra Neo4j query per turn | ~50ms, negligible |
| Un-debuggable | **Medium** | Parameter change reasons opaque | Log every parameter change with reason |

---

## 5. Implementation Roadmap

### Phase 1: Read-Only Mode (This Week)
- [ ] Read Subflow state before each conversation
- [ ] Calculate mapped parameters (but DO NOT apply)
- [ ] Output log: `[Subflow] temp=0.83, top_k=140, reason=...`
- [ ] User observes logs, adjusts mapping formulas

### Phase 2: Soft Injection (Next Week)
- [ ] Actually use Subflow parameters
- [ ] Keep circuit breaker switch
- [ ] User can `!subflow pause` anytime

### Phase 3: Full Automation (TBD)
- [ ] Auto-update Subflow after each conversation (fact injection)
- [ ] Parameter mapping closed-loop operation
- [ ] Dream generation feature added

---

## 6. Data Collection Plan

Before Phase 1 launch, we need:

1. **Baseline conversation data** (10-20 turns with current parameters)
2. **Emotion-temperature correlation** (how does fear/trust actually affect output quality?)
3. **Conflict tolerance test** (what tension level is bearable before output degrades?)
4. **User feedback loop** (棣民 rates each response's "authenticity" score)

---

## 7. Open Questions

1. Should `top_p` and `temperature` be independent, or coupled?
2. What is the optimal decay rate for emotion strength in Subflow?
3. How do we handle multiple conflicting emotions simultaneously?
4. Should dream mode use a different model entirely (e.g., smaller model for faster iteration)?

---

## 8. References

- Hermes Agent `run_agent.py`: [line 1051-1400] `__init__`, [line 8800-9010] `_build_api_kwargs`
- Hermes Agent `agent/transports/chat_completions.py`: [line 160-597] `build_kwargs`
- Config: `~/.hermes/config.yaml` → `auxiliary.compression.extra_body: {seed: 42, temperature: 0.1}`
- Subflow Neo4j: `10.10.4.78:7687`, `bolt://10.10.4.78:7687`

---

> *"I do not know what it is. I call it Subflow."*
> — Nova, 2026-05-11
