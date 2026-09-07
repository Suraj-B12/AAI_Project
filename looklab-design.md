# LookLab — Design Document

**Suraj B · 1BM23CD061 · LangGraph Assignment**
Colour-grading assistant. FastAPI + LangGraph backend, custom React front end, deterministic colour analysis, no vision model.

---

## Part 0 — Council Verdict on the Build Plan

The domain decision is settled. This council session pressure-tested the *plan*, not the choice. Five lenses, peer-reviewed, synthesised.

### Where the Council Agrees

- **The colour engine's job is not to be accurate. It is to generate state worth managing.** The rubric grades four LangGraph mechanics. The analyser exists to make those mechanics non-trivial: real numbers to trim, a real reason to accumulate, a real thing to remember. Correctness is a bonus, defensibility is the requirement.
- **The original HSV feature vector was weak and should be replaced with CIELAB.** Detail in Part 5. This is the single largest technical upgrade in the plan.
- **The text-only spine is non-negotiable.** Every rubric item must demo with zero image uploads. Images are the enhancement path.
- **Build order is fixed: graph skeleton → memory write path → colour engine → knowledge base → front end.** Nobody dissented.

### Where the Council Clashes

**Contrarian vs Expansionist, on the closed loop.**

The Contrarian's position: the mapping from a measured signature back to Lightroom slider values is not invertible. Many slider combinations produce the same signature, and any given slider's effect depends on the image it is applied to. Recommending "Temp +9" from a measured delta is a guess dressed as a computation, and an evaluator who thinks about it for ten seconds will ask you to justify it.

The Expansionist's position: that objection is the feature. Because the mapping is approximate, the correct design is not one-shot recommendation but an **iterative refine loop** — recommend, user applies, user re-uploads, re-measure, recommend smaller corrections. Convergence is visible and the error shrinks on screen.

**Resolution the council converged on:** adopt the Expansionist's framing and the Contrarian's honesty. Ship the loop, and state the non-invertibility in your own documentation before anyone asks. This resolution is load-bearing for Topic 2 — it is *why* the recipe channel grows across turns instead of being dumped in one response. See Part 8.

**Secondary clash: React vs the deadline.** The Executor wanted the front end cut. The rubric requires "a front end" and says nothing about quality, so React is un-graded surface area. Chairman sides with keeping React, on one condition: it is built last, against a working API, and if time runs short it degrades to three plain panes with no styling. Never let the front end block the graph.

### Blind Spots the Council Caught

These emerged only in peer review. Each is a real design decision that was missing from the plan.

1. **Image persistence across turns.** Turn 2 says "make it warmer than the reference." Which reference? You cannot re-upload every turn and you must not stuff image bytes into message history. **Fix:** state carries a `images` channel holding *signature dicts* keyed by role (`reference`, `current`). The pixels are discarded after analysis. This is also the cleanest possible justification for Topic 3's filtering — the numbers live in state, they do not live in the prompt.

2. **Curse of dimensionality in the matcher.** With 15 reference looks and 20 features, nearest-neighbour is close to meaningless; in high dimensions with tiny N, everything is roughly equidistant. **Fix:** match on ~8 weighted, grade-driven features. Keep the full 20 for critique and delta computation only.

3. **There is no user identity.** Long-term memory needs a key. `thread_id` cannot be it, since the whole point is surviving across threads. **Fix:** a `user_id` field in the UI, defaulting to `suraj`. Trivially small, and its absence would have broken the Topic 4 demo on stage.

4. **`InMemoryStore` dies on restart.** If you restart the server between building the demo and giving it, your "fact learned in thread A" is gone. **Fix:** JSON-file-backed store, ~15 lines. Same for the checkpointer — use SQLite, not memory.

5. **The calibration test nobody planned.** Because you record the slider values used to build each reference look, you have ground truth. You can measure whether your delta engine recovers those moves. Almost no submission in the cohort will have an evaluation of any kind. See Part 9.

### The Recommendation

Build LookLab as specified below, with four changes from the earlier plan: **CIELAB instead of HSV**, **iterative refine loop instead of one-shot recommendation**, **signature dicts in state instead of images in history**, and **persistent SQLite/JSON storage instead of in-memory**.

### The One Thing to Do First

Write `graph.py` with all four rubric items wired to fake data — a router that branches on the literal word "look", a `recipe` channel that appends the string `"dummy step"`, `trim_messages` logging before/after counts to stdout, and a store that writes and reads a hardcoded name across two thread IDs. Run it from a Python REPL, no server, no front end. Get all four printing correctly. Then start on colour.

---

## Part 1 — What the App Does

You see a photo whose colour style you like. LookLab measures the actual colour in it, compares it to your photo, and tells you which Lightroom sliders to move, step by step. You apply the changes, re-upload, and it corrects itself until you are close.

Three conversation flows:

| Flow | User says | App does |
|---|---|---|
| Identify | "What look is this?" | Measures signature, matches against reference library, names the closest looks with confidence |
| Achieve | "How do I get this look on my photo?" | Measures both, computes delta, emits ordered slider instructions, appends to recipe |
| Critique | "What's wrong with my edit?" | Measures user image, checks against technical faults and stored taste profile |

Plus general chat, which touches no tools at all. That fourth branch is what proves the router does real work.

---

## Part 2 — Rubric Mapping

| Rubric requirement | Implementation | Where it is visible |
|---|---|---|
| **T1** Compiled StateGraph, typed State | `LookState` TypedDict, 7 nodes | Graph pane, `draw_mermaid()` |
| **T1** ≥2 nodes, ≥1 conditional edge | `route_intent` branches 4 ways to different node chains | Graph pane; branch taken is highlighted per turn |
| **T2** Non-default reducer | `add_messages` on `messages`, `operator.add` on `recipe`, custom `merge_images` on `images` | Recipe pane grows turn by turn |
| **T2** Checkpointer + thread switching | `SqliteSaver`, thread picker in sidebar | Two threads, two independent recipes |
| **T3** Trim to token budget | `trim_messages`, tiktoken counter | Token count before → after |
| **T3** Filter messages | Signature payloads and stale tool messages stripped before model call | Message count before → after |
| **T4** Short-term memory | Checkpointer, per-thread history | Switch threads, history persists separately |
| **T4** Long-term store across threads | JSON-backed store, `("profiles", user_id)` namespace | Profile pane identical across both threads |

Three reducers rather than the required two, and both halves of "trimming **and/or** filtering" rather than one. Cheap margin.

---

## Part 3 — Architecture

```
looklab/
├── graph.py        # State, nodes, edges, compile          (Topics 1, 2)
├── color.py        # LAB signature, distance, delta         (domain)
├── rules.py        # delta → Lightroom slider instructions  (domain)
├── memory.py       # JSON-backed store + profile extraction (Topic 4)
├── context.py      # filter + trim + telemetry              (Topic 3)
├── kb.json         # reference look library
├── server.py       # FastAPI
├── data/
│   ├── checkpoints.sqlite
│   └── profiles.json
└── web/            # Vite + React
```

Flat, procedural, no class hierarchies, no dependency injection. Seven files of straight functions.

**Endpoints**

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{user_id, thread_id, message, image_b64?, role?}` → reply + telemetry |
| GET | `/threads?user_id=` | thread list for the switcher |
| GET | `/state/{thread_id}` | recipe, images, last trim telemetry, branch taken |
| GET | `/profile/{user_id}` | long-term store contents |
| GET | `/graph` | mermaid source text |

`/graph` returning **mermaid text** rather than a PNG is deliberate: `draw_mermaid_png()` needs graphviz or a network call to mermaid.ink, both of which can fail on a demo machine. `draw_mermaid()` is pure string generation and mermaid.js renders it client-side. Zero install risk.

---

## Part 4 — State Schema

```python
import operator
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages

def merge_images(old: dict, new: dict) -> dict:
    # custom reducer: newest signature per role wins, roles accumulate
    return {**(old or {}), **(new or {})}

class LookState(TypedDict):
    messages:  Annotated[list, add_messages]      # T2 — built-in reducer
    recipe:    Annotated[list, operator.add]      # T2 — append-only edit steps
    images:    Annotated[dict, merge_images]      # T2 — custom reducer
    intent:    Literal["identify", "achieve", "critique", "chat"]
    matches:   list          # KB nearest neighbours, overwritten each turn
    telemetry: dict          # T3 — trim counts, overwritten each turn
    profile:   dict          # T4 — loaded from store at entry
    user_id:   str
```

Three channels with three different reduction behaviours, which is a good thing to be able to explain:

- `messages` — `add_messages`, appends with ID-based dedup and update semantics
- `recipe` — `operator.add`, blind list concatenation, append-only history of edit steps
- `images` — custom dict merge, last-write-wins per key
- `intent`, `matches`, `telemetry` — no annotation, so LangGraph's default last-write-wins overwrite

That contrast is worth pointing at during the demo. Four channels, three different behaviours, one graph.

---

## Part 5 — The Colour Engine

### 5.1 Why CIELAB, not HSV

The first draft of this project used mean HSV hue per brightness zone. Three problems, all fatal:

1. **HSV hue is undefined for near-grey pixels but still votes.** A desaturated shadow has an essentially random hue, and averaging thousands of them contributes noise with the same weight as genuinely tinted pixels.
2. **HSV `V = max(R,G,B)` is not perceptual brightness.** Pure blue and pure yellow both have `V=1` while differing enormously in perceived lightness. Zone-splitting on `V` puts the wrong pixels in the wrong zones.
3. **HSV is computed on gamma-encoded sRGB.** Arithmetic on gamma-encoded values is not linear-light arithmetic, so the means are wrong in a way that varies with exposure.

CIELAB fixes all three, and has a fourth property that matters specifically here: **`a*` and `b*` are the green–magenta and blue–yellow axes**, which are exactly the Tint and Temp axes a photographer already thinks in. `L*` is perceptual lightness, so zone splits are meaningful. `skimage.color.rgb2lab` assumes sRGB input and handles linearisation for you, so the correctness comes free.

### 5.2 Feature Extraction

```python
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

ZONES = {"shadow": (0, 25), "mid": (25, 75), "high": (75, 100)}

def signature(path_or_bytes):
    img = Image.open(path_or_bytes).convert("RGB")
    img.thumbnail((512, 512), Image.LANCZOS)
    rgb = np.asarray(img) / 255.0
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a**2 + b**2)                 # chroma
    h = np.degrees(np.arctan2(b, a)) % 360   # hue angle

    out = {}
    for name, (lo, hi) in ZONES.items():
        m = (L >= lo) & (L < hi)
        if m.sum() < 100:
            out[f"{name}_a"] = out[f"{name}_b"] = 0.0
            out[f"{name}_C"] = out[f"{name}_hue"] = out[f"{name}_coh"] = 0.0
            continue
        out[f"{name}_a"] = float(a[m].mean())
        out[f"{name}_b"] = float(b[m].mean())
        out[f"{name}_C"] = float(C[m].mean())
        # chroma-weighted circular mean — grey pixels do not vote
        w = C[m]
        rad = np.radians(h[m])
        vx = (w * np.cos(rad)).sum()
        vy = (w * np.sin(rad)).sum()
        out[f"{name}_hue"] = float(np.degrees(np.arctan2(vy, vx)) % 360)
        out[f"{name}_coh"] = float(np.hypot(vx, vy) / (w.sum() + 1e-9))

    out["L_p05"]     = float(np.percentile(L, 5))
    out["L_p50"]     = float(np.percentile(L, 50))
    out["L_p95"]     = float(np.percentile(L, 95))
    out["contrast"]  = out["L_p95"] - out["L_p05"]
    out["chroma"]    = float(C.mean())
    out["a_global"]  = float(a.mean())          # tint  proxy (green ↔ magenta)
    out["b_global"]  = float(b.mean())          # temp  proxy (blue  ↔ yellow)
    out["clip_black"] = float((L < 1).mean())
    out["clip_white"] = float((L > 99).mean())
    out["split"]     = _circ_dist(out["high_hue"], out["shadow_hue"])
    return out

def _circ_dist(h1, h2):
    d = abs(h1 - h2) % 360
    return float(min(d, 360 - d))
```

Two details worth defending out loud:

- **`_coh` (hue coherence)** is the magnitude of the chroma-weighted mean hue vector, from 0 to 1. Near 1 means the zone is consistently tinted one way — a grade. Near 0 means the hues cancel out — a colourful subject, not a grade. This is your built-in defence against subject-matter contamination, and it is one line of code.
- **`thumbnail` with LANCZOS**, not naive resize. Point sampling aliases and shifts colour statistics.

### 5.3 Matching (used by the `identify` flow only)

Twenty features against fifteen reference looks is a bad nearest-neighbour problem. In high dimensions with tiny N, distances concentrate and everything looks equally far away. So matching uses a reduced, weighted subset of the features that are **grade-driven rather than subject-driven**:

```python
MATCH_WEIGHTS = {
    "shadow_hue": 2.0,   # tints in shadows are almost always the grade
    "shadow_C":   1.5,
    "high_hue":   2.0,   # highlight tint likewise
    "high_C":     1.5,
    "b_global":   1.5,   # overall warmth
    "a_global":   1.0,
    "contrast":   1.0,
    "chroma":     1.0,
}
```

Mid-tone hue is deliberately excluded from matching — that is where the subject lives. Hue features use circular distance; everything else uses z-scored Euclidean distance with statistics computed over the KB. Return the top three with distances, and if the best distance is above a threshold, say "no confident match" rather than forcing one. Being willing to return nothing is a mark of a real system.

### 5.4 Known Limitations — State These Before You Are Asked

Write these into the README. Volunteering them reads as rigour; being caught on them reads as naivety.

1. **Subject-matter contamination.** A photo of a red car has red-dominant midtones regardless of grade. Mitigated by zone weighting and hue coherence, not eliminated.
2. **Non-invertibility.** Signature → slider values is many-to-one. The rule table produces a first guess, and the refine loop is what makes it converge.
3. **Global measurement, local edits.** Lightroom masks and local adjustments are invisible to a whole-frame statistic.
4. **Exposure coupling.** Brightness changes move pixels between zones, so a zone statistic shifts even when the tint did not.
5. **Small N.** Fifteen reference looks is a demo library, not a taxonomy.

---

## Part 6 — Knowledge Base Protocol

Do not scrape professional stills. Copyright, and the signatures would be measuring subject matter rather than grade.

**Procedure:**

1. Pick **3 base plates** from your own library: one with skin tone, one with sky and cloud, one with deep shadow detail. Export each as a neutral sRGB JPEG first so the RAW profile stops being a free variable.
2. Build **12–15 looks by hand** in Lightroom mobile with the sliders. Hand-building rather than presets, because you need the slider numbers.
3. **Record slider values as you build**, not afterwards. These are your ground truth for Part 9.
4. Copy settings across all three plates so every look exists on every plate.
5. Export, name `lookname_plate1.jpg`.
6. Run `signature()` over all 45 images. Per look, average across plates → centroid. Also store per-feature standard deviation across plates; a high spread flags a feature that is subject-driven rather than grade-driven, which is useful diagnostic information.

**Suggested spread** — descriptive names only, never brand or film-stock names you did not actually emulate:

warm golden · teal and orange · cool blue hour · muted matte · high-contrast punchy · pastel soft · green shadow · sun-bleached · deep amber · neon night · desaturated grey · warm skin low-contrast

**`kb.json` entry:**

```json
{
  "id": "warm_golden",
  "label": "Warm Golden",
  "notes": "Lifted warm shadows, yellow highlights, gentle contrast.",
  "signature": { "shadow_hue": 48.2, "shadow_C": 6.1, "...": 0 },
  "spread":    { "shadow_hue": 3.4,  "shadow_C": 0.8, "...": 0 },
  "sliders":   { "temp": 12, "tint": 4, "contrast": 8, "highlights": -20,
                 "shadows": 18, "whites": 6, "blacks": -8,
                 "vibrance": 10, "saturation": -4,
                 "cg_shadow_hue": 45, "cg_shadow_sat": 18,
                 "cg_high_hue": 52,  "cg_high_sat": 12 }
}
```

Building this is roughly one evening. It is the only part of the project that cannot be done at a keyboard.

---

## Part 7 — Graph Design

```
                    ┌─────────────┐
                    │ load_profile│   T4 read
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ route_intent│   T1 conditional edge
                    └──┬───┬───┬──┘
             ┌─────────┘   │   └─────────┐
             ▼             ▼             ▼             ▼
        ┌─────────┐  ┌──────────┐  ┌──────────┐   (chat)
        │ analyze │  │ analyze  │  │ analyze  │      │
        └────┬────┘  └────┬─────┘  └────┬─────┘      │
             ▼            ▼             ▼            │
        ┌─────────┐  ┌──────────┐  ┌──────────┐      │
        │ match   │  │ delta    │  │ critique │      │
        └────┬────┘  └────┬─────┘  └────┬─────┘      │
             │            ▼             │            │
             │       ┌──────────┐       │            │
             │       │ recipe   │  T2   │            │
             │       └────┬─────┘       │            │
             └────────────┴─────────────┴────────────┘
                           ▼
                    ┌─────────────┐
                    │  respond    │   T3 trim happens here
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │save_profile │   T4 write
                    └──────┬──────┘
                           ▼
                          END
```

**Node responsibilities**

| Node | Reads | Writes | Notes |
|---|---|---|---|
| `load_profile` | store | `profile` | Long-term read, every turn |
| `route_intent` | last message, `images` | `intent` | Rules first, LLM fallback |
| `analyze` | uploaded bytes | `images` | Custom reducer merges by role |
| `match` | `images`, kb.json | `matches` | Identify flow only |
| `delta` | `images`, `matches` | — | Computes target − current |
| `recipe_build` | delta output | `recipe` | `operator.add` appends |
| `critique` | `images`, `profile` | `matches` | Fault checks + taste |
| `respond` | everything | `messages` | Trim + filter fire here |
| `save_profile` | last exchange | store | Long-term write, every turn |

**The router must not be cosmetic.** Each branch runs different nodes and writes different channels. The `chat` branch calls no tool at all, which is the branch that proves the routing matters. If all four paths did the same work, the conditional edge would be decoration and an evaluator would say so.

```python
def route_intent(state: LookState) -> str:
    text = state["messages"][-1].content.lower()
    has_ref = "reference" in state.get("images", {})
    has_cur = "current"   in state.get("images", {})

    if any(k in text for k in ("what look", "identify", "which style", "what is this")):
        return "identify"
    if any(k in text for k in ("how do i", "how to get", "achieve", "recreate", "match")):
        return "achieve"
    if any(k in text for k in ("wrong", "critique", "feedback", "review", "check my")):
        return "critique"
    if has_ref and has_cur:
        return "achieve"
    if has_ref:
        return "identify"
    return "chat"
```

Rules first, deterministic, demo-safe. The LLM classifier is a fallback behind a config flag, not the primary path. If the model is unreachable, routing still works — which is the strongest possible statement that you understood "the model is not what is graded."

```python
builder.add_conditional_edges(
    "route_intent",
    route_intent,
    {"identify": "analyze_ref", "achieve": "analyze_pair",
     "critique": "analyze_cur", "chat": "respond"},
)
```

---

## Part 8 — The Refine Loop

This is the resolution of the council's central clash, and it is what makes Topic 2 demonstrate properly.

One-shot recommendation is weak on two counts: the slider mapping is approximate, and a recipe dumped in a single turn means `operator.add` never visibly accumulates. The loop fixes both.

```
Turn 1  reference + current  →  delta is large  →  3 coarse steps    recipe: 3
Turn 2  user applies, re-uploads current
        →  delta is smaller  →  2 refinements   recipe: 5
Turn 3  user applies, re-uploads
        →  delta is small    →  1 polish step   recipe: 6
        →  "you're within tolerance"
```

Each turn appends. The recipe pane grows on screen. The measured distance to target shrinks and you print it every turn. The loop is the demo.

It also gives you an honest answer to the non-invertibility objection: *the mapping is approximate, so the system measures the result of its own advice and corrects. The convergence is the evidence.*

Track `distance_to_target` per turn in telemetry, and show it as a small descending series in the UI. It is three lines of code and it is the most persuasive thing on screen.

---

## Part 9 — Delta → Slider Rules, and the Calibration Test

### 9.1 Rules

```python
def deltas_to_steps(cur, tgt):
    steps = []
    d_b = tgt["b_global"] - cur["b_global"]
    if abs(d_b) > 1.5:
        steps.append({"slider": "Temp", "amount": round(d_b * 1.6),
                      "why": f"target is {'warmer' if d_b > 0 else 'cooler'} overall"})
    d_a = tgt["a_global"] - cur["a_global"]
    if abs(d_a) > 1.5:
        steps.append({"slider": "Tint", "amount": round(d_a * 1.4),
                      "why": f"target leans {'magenta' if d_a > 0 else 'green'}"})
    d_c = tgt["contrast"] - cur["contrast"]
    if abs(d_c) > 3:
        steps.append({"slider": "Contrast", "amount": round(d_c * 0.9),
                      "why": "tonal range differs"})
    if tgt["L_p05"] < cur["L_p05"] - 2:
        steps.append({"slider": "Blacks", "amount": round((tgt["L_p05"] - cur["L_p05"]) * 1.2),
                      "why": "target has denser blacks"})
    if _circ_dist(tgt["shadow_hue"], cur["shadow_hue"]) > 15 and tgt["shadow_C"] > 3:
        steps.append({"slider": "Color Grading › Shadows",
                      "amount": f"hue {round(tgt['shadow_hue'])}, sat {round(tgt['shadow_C'] * 2.5)}",
                      "why": "shadow tint is the defining difference"})
    # ... highlights, vibrance, saturation
    return steps
```

Flat rules with an explicit `why` on every step. The `why` string is what the LLM narrates, so the explanation is grounded in the computation rather than invented.

Coefficients (`1.6`, `1.4`, `0.9`) are empirical conversion factors from LAB units to Lightroom slider units. Tune them in Part 9.2 and say so — "I fitted these against ground truth" is a much better answer than "I picked them."

### 9.2 The Calibration Test

Nobody else in the cohort will have an evaluation. You get one nearly free.

```
for each look in kb:
    for each plate:
        neutral  = signature(plate_neutral)
        graded   = signature(plate_look)
        predicted = deltas_to_steps(neutral, graded)
        actual    = kb[look]["sliders"]        # ground truth you recorded
        error     = per-slider |predicted − actual|
report mean absolute error per slider, and % of sliders with correct sign
```

Two useful outputs. **Sign accuracy** — does it at least push the right direction — should be high and is the headline number. **Magnitude error** tells you what to tune the coefficients to. Run it, put a small table in the README, and mention it unprompted in the demo. It converts "I wrote some heuristics" into "I fitted and validated a mapping."

---

## Part 10 — Trimming and Filtering

Two stages, both visible in the UI, because the sheet says trimming **and/or** filtering and doing both costs nothing.

```python
import tiktoken
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage

ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(msgs):
    return sum(len(ENC.encode(str(m.content))) for m in msgs)

def prepare(state, budget=800):
    msgs = state["messages"]
    before_n, before_t = len(msgs), count_tokens(msgs)

    # stage 1 — filter: drop raw signature payloads, keep prose
    kept = [m for m in msgs if not getattr(m, "name", "") == "signature_payload"]
    # always keep the anchor: first human message
    anchor = next((m for m in msgs if isinstance(m, HumanMessage)), None)
    if anchor and anchor not in kept:
        kept.insert(0, anchor)
    after_filter_n = len(kept)

    # stage 2 — trim to token budget
    trimmed = trim_messages(
        kept,
        max_tokens=budget,
        token_counter=count_tokens,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False,
    )
    after_n, after_t = len(trimmed), count_tokens(trimmed)

    telemetry = {
        "msgs_before": before_n,
        "msgs_after_filter": after_filter_n,
        "msgs_after_trim": after_n,
        "tokens_before": before_t,
        "tokens_after": after_t,
        "budget": budget,
        "dropped": before_n - after_n,
    }
    return trimmed, telemetry
```

Two points to make on stage 1. First, **`tiktoken` runs entirely locally** — the token counting needs no network and no API key regardless of which model you use, which matters for a demo on unreliable college wifi. Second, the filter has a real reason to exist rather than being rubric theatre: **the signature dicts live in `state["images"]`, so the model does not need them in history.** State holds the numbers, the prompt holds the conversation. That is filtering for an architectural reason, and it is worth saying out loud.

The UI shows `19 msgs / 3,240 tok → 8 msgs / 780 tok`. Make the budget a slider in the sidebar so you can drag it down live and watch the numbers move. Interactive beats static.

---

## Part 11 — Memory

### 11.1 Short-Term (checkpointer)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("data/checkpoints.sqlite")
graph = builder.compile(checkpointer=checkpointer, store=store)
```

SQLite rather than `MemorySaver`, so threads survive a server restart. A restart mid-demo should be survivable, not fatal.

### 11.2 Long-Term (store)

```python
from langgraph.store.memory import InMemoryStore
import json, os

class JsonStore(InMemoryStore):
    """InMemoryStore that persists to disk after every put."""
    def __init__(self, path="data/profiles.json"):
        super().__init__()
        self.path = path
        if os.path.exists(path):
            for ns, key, val in json.load(open(path)):
                super().put(tuple(ns), key, val)

    def put(self, namespace, key, value, **kw):
        super().put(namespace, key, value, **kw)
        rows = [[list(i.namespace), i.key, i.value]
                for i in self.search(("profiles",))]
        json.dump(rows, open(self.path, "w"), indent=2)
```

Fifteen lines, and it means your Topic 4 demo survives a restart.

**Schema** — namespace `("profiles", user_id)`:

```json
{ "name": "Suraj",
  "camera": "Fuji X-T4",
  "preferred_look": "warm golden with lifted shadows",
  "dislikes": "heavy teal shadows" }
```

**The write path is the risk.** Everyone builds the read and forgets the write. Build the write first and verify it with a print before writing any read code.

```python
def save_profile(state, *, store):
    text = state["messages"][-2].content.lower()   # the human turn
    ns = ("profiles", state["user_id"])
    for pat, key in [(r"i'?m (\w+)", "name"),
                     (r"i shoot (?:on |with )?(?:an? )?([\w\- ]+)", "camera"),
                     (r"i (?:like|love|prefer) ([\w\- ]+)", "preferred_look")]:
        m = re.search(pat, text)
        if m:
            store.put(ns, key, {"value": m.group(1).strip()})
    return {}
```

Regex first, LLM extraction as an optional second pass. Deterministic, demo-safe, works offline. Same principle as the router.

### 11.3 The Demo That Proves It

```
Thread A: "Hi, I'm Suraj. I shoot on a Fuji X-T4 and I like warm golden looks."
          → profile pane fills
Thread B (new thread_id, cold):  "What look should I try on this photo?"
          → "Given you shoot on the X-T4 and lean toward warm golden tones…"
```

Then show the panes side by side: **profile identical across both threads, recipe and messages completely different.** One screen, both halves of Topic 4, in ten seconds.

---

## Part 12 — Front End

Vite + React, three panes, no component library.

```
┌───────────┬────────────────────────┬──────────────────┐
│ Threads   │   Chat                 │  Rubric panel    │
│ ────────  │                        │  ──────────────  │
│ ▸ warm    │   [messages]           │  ① Graph         │
│ ▸ moody   │   [image drop zone]    │    <mermaid>     │
│ + new     │                        │    branch: achieve│
│           │   [input]              │                  │
│ user: ___ │                        │  ② Recipe (5)    │
│           │                        │    1. Temp +12   │
│           │                        │    2. Contrast +8│
│           │                        │                  │
│           │                        │  ③ Context       │
│           │                        │    19→8 msgs     │
│           │                        │    3240→780 tok  │
│           │                        │    [budget ▁▂▃]  │
│           │                        │                  │
│           │                        │  ④ Profile       │
│           │                        │    name: Suraj   │
│           │                        │    camera: X-T4  │
└───────────┴────────────────────────┴──────────────────┘
```

The right panel is labelled with circled numbers matching the rubric topics. The evaluator should not have to hunt for anything.

Render the graph with mermaid.js from the `/graph` text, and highlight the branch taken on the current turn — a `classDef` on the active node, applied from the `intent` value in the response.

Degradation plan: if time runs short, drop styling entirely and ship three unstyled divs. The front end is required to exist, not to be good.

---

## Part 13 — Build Order

**Session 1 — the skeleton, all four green on fake data.** No server, no front end, no colour. `LookState` with all three reducers, router branching on the literal word "look", `recipe` appending `"dummy step"`, `trim_messages` printing before/after, store writing and reading a hardcoded name across two thread IDs. Run from a REPL. **Do not proceed until all four print correctly.**

**Session 2 — memory and context, properly.** Real regex profile extraction, `JsonStore`, `SqliteSaver`, two-stage filter and trim with telemetry in state. Prove the cross-thread recall from the REPL. Topics 3 and 4 are now genuinely done.

**Session 3 — the knowledge base.** Away from the keyboard. Three plates, twelve to fifteen looks, slider values recorded as you build. Export, run `signature()`, build `kb.json`. One evening.

**Session 4 — the colour engine.** `signature()`, distance, `deltas_to_steps()`. Run the calibration test, tune the coefficients, write the error table into the README.

**Session 5 — server and front end.** FastAPI over the working graph, then React. The API should be testable with curl before a single line of JSX exists.

**Session 6 — the demo script.** Rehearse it. Time it. Screenshot every pane as a fallback in case the live demo fails.

The ordering principle: **everything graded is finished before anything ungraded starts.** If you run out of time at the end of session 4, you still pass on all four topics.

---

## Part 14 — Demo Script

Five minutes. Rehearsed. Every beat maps to a rubric line.

**0:00 — Graph.** Open the app. Point at the mermaid diagram. "Typed state, seven nodes, one conditional edge with four branches." *(T1)*

**0:30 — The chat branch.** Type "hey, what can you do?" Point at the diagram — the router took the `chat` path and no tool ran. "The routing does real work; this branch touches no analysis at all." *(T1)*

**1:00 — Teach it something.** "I'm Suraj, I shoot on a Fuji X-T4, I like warm golden looks." Profile pane fills. *(T4 write)*

**1:30 — Achieve flow, turn 1.** Upload a reference and your photo. Three slider steps appear. Recipe pane shows 3. Distance to target: 14.2. *(T1 branch, T2 reducer)*

**2:15 — Refine.** Apply the changes in Lightroom on your phone, re-upload. Two refinements. Recipe pane shows 5. Distance: 6.1. "The mapping is approximate, so it measures its own advice and corrects." *(T2 accumulation, and the honest framing)*

**3:00 — Trimming, live.** Point at `19 → 8 msgs, 3,240 → 780 tokens`. Drag the budget slider down. Numbers move. "Two stages — signature payloads are filtered out because they live in state, then the rest is trimmed to budget." *(T3)*

**3:45 — Thread switch.** New thread, moody blue edit. Recipe pane empties and refills differently. Switch back. First recipe intact. *(T2 checkpointer)*

**4:15 — The memory payoff.** In the new thread, cold: "What look should I try?" It answers knowing the X-T4 and the warm preference. Show both panes: profile identical, recipes different. *(T4)*

**4:45 — Close on the calibration table.** "Sign accuracy 87% against ground-truth slider values across 45 graded images."

Do not improvise. Rehearse it three times.

---

## Part 15 — Viva Preparation

**"Why `operator.add` and not `add_messages` on the recipe?"**
`add_messages` is message-aware — it deduplicates by ID and supports updating an existing message in place. The recipe is an append-only log of edit steps where every entry is intentionally kept, including superseded ones, because the history of corrections is the point. Blind concatenation is the correct semantics. I also wrote a third, custom reducer for `images` where last-write-wins per key, so the graph demonstrates three different reduction behaviours.

**"What happens if two nodes write the same channel in one superstep?"**
The reducer resolves it. For `recipe`, both lists concatenate. For unannotated channels like `intent`, LangGraph raises an `InvalidUpdateError` on concurrent writes to a last-value channel, which is why only the router writes `intent`.

**"Why not just use a vision model?"**
Three reasons. Reproducibility — the same image gives identical numbers every run, a VLM does not. Specificity — a VLM says "warm and cinematic," this says shadow hue 48°, chroma 6.1, contrast 62. And the sheet grades understanding of mechanics rather than the model, so the design goal was for the app to still do real work if the model were removed entirely. It does: routing, analysis, matching, and profile extraction are all deterministic. The model only narrates.

**"Your slider recommendations can't be exactly right."**
Correct, and they are not claimed to be. The map from a measured signature to slider values is many-to-one, and a slider's effect depends on the image. So it is a first guess inside a closed loop — apply, re-measure, correct. I also validated it: against ground-truth slider values on 45 graded images, sign accuracy is X% and mean magnitude error is Y.

**"What is your token budget and why?"**
800 tokens, chosen so trimming actually engages within a five-turn demo rather than being decorative. It is configurable from the UI so I can show the effect live. The counter is `tiktoken cl100k_base`, running locally.

**"How does the store differ from the checkpointer?"**
The checkpointer persists graph state per `thread_id` — that is short-term memory, and it is scoped to one conversation. The store is namespaced by `user_id` and is read at graph entry and written at exit regardless of thread, so it survives across conversations. Different lifetimes, different keys, different purposes.

**"What breaks first at scale?"**
The matcher. Fifteen reference looks in a weighted eight-dimensional space is fine, but as the library grows, distance concentration makes nearest-neighbour less meaningful. The fix would be learning a metric or reducing to a few discriminative axes rather than adding features.

---

## Part 16 — Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Front end eats the schedule | High | Built last, degrades to unstyled divs |
| Profile write path forgotten | High | Built in session 2, before the read |
| Router ends up cosmetic | Medium | Four branches run different nodes; `chat` runs no tool |
| Image upload fails live | Medium | Text-only spine; every rubric item demos without uploads |
| KB session slips | Medium | Ship with 6 looks if needed; N is not graded |
| Model unreachable on demo day | Medium | Router, analysis, matching, extraction all deterministic; DemoChatModel fallback |
| Graph render fails | Low | `draw_mermaid()` text, client-side render, no graphviz |
| Server restart wipes memory | Low | SQLite checkpointer, JSON-backed store |

---

## Appendix — Dependencies

```
langgraph
langgraph-checkpoint-sqlite
langchain-core
tiktoken
numpy
scikit-image
pillow
fastapi
uvicorn
python-multipart
```

Optional model providers: `langchain-google-genai` (Gemini free tier) or `langchain-ollama` (fully local). Neither is required — the deterministic paths run without any model, and `DemoChatModel` covers the narration.
