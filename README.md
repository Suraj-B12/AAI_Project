# LookLab

**Suraj B · 1BM23CD061 · LangGraph Assignment**

A colour-grading assistant. You name a look, or upload a reference photo and your
own; LookLab measures the actual colour in CIELAB, computes the difference, and
tells you which Lightroom sliders to move. You apply the changes, re-upload, and
it corrects itself until you converge.

FastAPI + LangGraph backend, deterministic colour analysis, **no vision model**.
Every number on screen is measured. The language model, when enabled, only
rewrites the wording — remove it entirely and the app still does all of its work.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m uvicorn looklab.server:app --port 8077
```

Open <http://127.0.0.1:8077>. No API key, no network and no build step are
required — the knowledge base is committed and the narrator is deterministic.

```bash
pip install -r requirements-dev.txt
pytest -q                       # 209 tests
python -m tools.stress          # 49 load / fuzz / soak checks against a live server
python -m tools.calibrate       # the evaluation table below
```

To deploy it somewhere your marker can open, see **[DEPLOY.md](DEPLOY.md)**.
Short version: it is one container — the front end is a single HTML file that
FastAPI serves at `/`, so there is nothing to deploy separately.

```bash
docker build -t looklab . && docker run -p 8077:8077 -e PORT=8077 looklab
```

---

## Rubric map

Every row below is backed by a named test. `pytest -q` runs all of them.

| Requirement | Implementation | Test |
|---|---|---|
| **T1** compiled `StateGraph`, typed state | `LookState` TypedDict, 11 nodes | `test_t1_graph_compiles_with_typed_state` |
| **T1** ≥2 nodes, ≥1 conditional edge | 4-way conditional edge into 4 different chains | `test_t1_conditional_edge_has_four_distinct_branches` |
| **T1** routing does real work | the `chat` branch runs no analysis node at all | `test_t1_chat_branch_runs_no_analysis` |
| **T2** non-default reducers | `add_messages` / `operator.add` / custom `merge_images` | `test_t2_three_channels_use_three_different_reducers` |
| **T2** reducer accumulates | recipe grows per turn and does not double | `test_t2_recipe_accumulates_across_turns_and_does_not_double` |
| **T2** checkpointer + thread switching | `SqliteSaver`, with a **negative** assertion | `test_t2_thread_switching_keeps_recipes_separate` |
| **T3** trim to a token budget | `trim_messages` + tiktoken, telemetry in state | `test_t3_trim_reduces_tokens_to_the_budget` |
| **T3** filter messages | signature payloads dropped before the model call | `test_t3_filter_drops_payloads_but_keeps_prose` |
| **T4** short-term memory | per-thread checkpoint, survives a fresh graph | `test_t2_checkpointer_survives_a_fresh_graph_over_the_same_file` |
| **T4** long-term store across threads | `SqliteStore`, `("profiles", user_id)` | `test_t4_profile_survives_a_completely_fresh_store` |
| **T4** memory changes the answer | cold thread recommends from stored taste | `test_t4_cold_thread_recommends_from_the_stored_profile` |

Three reducers rather than the required two, and both halves of "trimming
**and/or** filtering" rather than one.

### Three channels, three reduction behaviours

```python
class LookState(TypedDict):
    messages:  Annotated[list, add_messages]   # append with ID-based dedup
    recipe:    Annotated[list, operator.add]   # blind concatenation, append-only
    images:    Annotated[dict, merge_images]   # custom: last-write-wins per key
    intent:    str                             # unannotated -> last-value overwrite
```

`recipe` uses `operator.add` rather than `add_messages` deliberately: it is an
append-only log of edit steps where superseded entries are kept on purpose,
because the history of corrections *is* the point. Blind concatenation is the
correct semantics there.

---

## What it can tell you

The advice covers the panels a photographer actually uses, not just the Basic
sliders:

| Panel | What it advises on |
|---|---|
| **Basic** | Temp, Tint, Contrast, Blacks, Whites, Shadows, Highlights, Vibrance/Saturation |
| **Colour Grading** | the Shadows, Midtones, Highlights and Global wheels, each with a hue and a saturation |
| **Colour Mixer (HSL)** | per-colour Hue, Saturation and Luminance across eight families &mdash; red, orange, yellow, green, aqua, blue, purple, magenta |

Every step says *what to move and why* in plain language, and keeps the
measurement on a separate line so the reasoning is readable without knowing
what CIELAB is:

```
1. Color Grading > Shadows  hue 185, sat 36
   Tint the dark areas toward teal. This is the biggest single thing
   that makes the look recognisable.
   (the dark areas are about 157 degrees of hue apart)

2. Temp +11
   The look you want is noticeably warmer than your photo. Push toward yellow.
   (blue-yellow axis +1.6 -> +8.3)
```

HSL families are measured by bucketing pixels by hue around eight measured
CIELAB centres, weighted by chroma so near-grey pixels do not vote, and a
family is only advised on when it occupies at least 4% of **both** frames.
Otherwise the engine would confidently tell you to move the Purple slider
because of a few dozen pixels.

---

## Answering questions

The chat branch is not a dead end. A free-form message is classified
deterministically -- no model is consulted to decide what kind of message it is
-- and each class gets a real answer:

| Class | Answered with |
|---|---|
| **glossary** | a cited definition from the 15-term colour-science glossary, with its Wikipedia source and CC BY-SA licence |
| **library** | the reference looks, listed from `kb.json` |
| **meta** ("are you hallucinating?") | a plain explanation of what is measured, what is arithmetic, and what it gets wrong |
| **social** | a short reply, not a capabilities dump |
| **domain question** | the model, if one is configured; otherwise an honest "I cannot answer that without guessing" |
| **out of scope** | declined, with a pointer to what it can do |

Model-written answers are **visibly marked**:

> *Answered by a language model, not measured. LookLab's slider advice and every
> number it quotes come from measuring your image; this reply does not.*

That marking is what keeps the honesty claim intact. A measurement and a
model's opinion are different kinds of thing, and the app says which is which.
Out-of-scope questions are never sent to the model at all.

---

## Memory you control

Say **"remember that ..."**, "note that ...", "don't forget ..." or "save this
to memory" and the fact is written to the long-term store, which is keyed by
user rather than by conversation. It then applies in every future thread. Free
notes are capped at ten and the oldest fall off, because memory that grows
without limit is a storage leak.

You can also ask it to **forget**: "forget my camera", "delete my profile",
"forget that I print on matte". A deletion is confirmed by listing what is
left, because "done" is a claim and the remaining list is evidence. Memory a
user cannot remove is not memory they control.

Conversations can be **deleted permanently**, including their uploaded images.
Deleting rows is not enough: SQLite in WAL mode keeps the pages in the sidecar
until a checkpoint and in the freelist until a VACUUM, so the delete path runs
checkpoint &rarr; VACUUM &rarr; checkpoint and reports the bytes actually freed
(measured: 5.5 MB &rarr; 52 KB on a thread holding ten image turns). The
per-thread lock is dropped too, and a deleted demo thread is not re-seeded on
the next restart. The long-term profile is deliberately kept &mdash; it belongs
to you, not to one conversation.

---

## The graph

```
load_profile ──▶ route_intent ──┬──▶ analyze_ref  ──▶ match        ──┐
   (T4 read)      (T1 edge)     ├──▶ analyze_pair ──▶ delta ──▶ recipe_build ──┤
                                ├──▶ analyze_cur  ──▶ critique     ──┤
                                └──▶ (chat, no analysis at all)  ────┤
                                                                     ▼
                                                        respond (T3 trim + filter)
                                                                     ▼
                                                        save_profile (T4 write) ──▶ END
```

The diagram in the UI is not drawn by hand — it is the string
`compiled_graph.get_graph().draw_mermaid()` emits about itself, served from
`GET /graph`. It is evidence about the artefact rather than a picture of one,
and it needs no graphviz and no network.

The router is rules-first and deterministic: if every model provider were
unreachable, routing, colour analysis, matching and profile extraction would all
still work.

---

## Evaluation

### Identification: which reference look is this?

Selected by leave-2-plates-out cross-validation (28 folds) over the eight
knowledge-base plates with a one-standard-error rule. **The held-out plates were
never consulted during selection and are scored exactly once.**

| | top-1 | top-3 |
|---|---|---|
| Held-out plates (135 frames, 9 photographs the KB never saw) | **54.1%** | **80.0%** |
| Random chance (15 looks) | 6.7% | 20.0% |
| | **8.1× chance** | **4.0× chance** |

The nominal CV winner was a five-feature set chosen directly from each feature's
measured plate-spread ratio. It led by 1.6 points against a standard error of
2.7 — less than noise — and leaned on clipping fractions that are structurally
zero for most photographs. The one-standard-error rule preferred a broader,
chromaticity-only set instead, which then scored 54.1% on held-out plates
against that candidate's 37.0%. Reproduce with `python -m tools.tune_matcher`.

### Delta engine: does it recover the grading move?

Reported on the same held-out plates. `deltas_to_steps` was **frozen before
`tools/calibrate.py` was ever run** and has not been tuned against its output.

| Slider | n | Sign agreement | Recall | Mean abs. error |
|---|---|---|---|---|
| Temp | 126 | 81.0% | 92.1% | 10.1 |
| Tint | 108 | **98.5%** | 63.0% | 4.6 |
| Contrast | 135 | 89.0% | 67.4% | 11.5 |
| Blacks | 135 | 87.3% | 81.5% | 12.6 |
| Whites | 108 | **68.4%** | **35.2%** | 8.5 |
| Vibrance / Saturation | 117 | 89.7% | 82.9% | 16.1 |
| **Overall** | **617** | **87.2%** | 72.9% | |

**Monotonicity: 36/36 sweeps.** A larger true move always produced a larger
recommendation, on every held-out plate and every swept slider. That is the
property the refine loop actually needs — the engine does not have to be
numerically right, it has to not reverse direction as the gap grows.

Colour-grading hue landed within 45° of the true wheel value 49.8% of the time
(mean error 36°). That is the weakest result here and it is reported as such.

**Whites is the weak slider**: 68.4% sign agreement and only 35.2% recall. The
L\* 95th percentile moves very little for large Whites values on images that
have few highlight pixels to begin with, so the engine frequently declines to
emit a step at all. Stated rather than buried.

### What this validates and what it does not

The reference looks in `looklab/kb.json` are produced by `looklab/grading.py`, a
documented numpy model of Lightroom-style controls applied to real photographs.
They are **not** exports from Adobe Lightroom.

So the calibration measures **sign agreement and monotonicity** — when the engine
says "Temp up", was the true move up — and deliberately does **not** claim
magnitude accuracy against Adobe, which would be meaningless against a
non-Adobe forward model.

The partial circularity is real and worth naming precisely: the simulator's Temp
operation applies channel gains in **linear-light RGB**, while `deltas_to_steps`
reads `b_global`, a mean of **CIELAB b\***. Those are different spaces and the
mapping between them is image-dependent — the same Temp value produces a
different b\* shift on a dark frame than on a bright one. A perfect score would
be suspicious; 87.2% with a visible weak slider is what an honest inverse looks
like. Directional agreement on the temp/tint axes is nonetheless weaker evidence
than on axes where the coupling is indirect.

Swap in real Lightroom exports and nothing downstream changes: `tools/build_kb.py`
changes its image source and the schema stays identical.

### Known limitations

1. **Subject-matter contamination.** A photo of a red car has red-dominant
   midtones regardless of grade. Mitigated by zone weighting and hue coherence,
   not eliminated — and visible in the numbers: across eight diverse plates,
   `L_p50` varies 6.3× more between photographs than between looks.
2. **Non-invertibility.** Signature → slider values is many-to-one. The rule
   table produces a first guess; the refine loop is what makes it converge.
3. **It cannot tell whether an image was deliberately graded.** At every
   threshold, ungraded photographs are flagged about as often as graded ones,
   because an untouched warm photo genuinely has the signature of a warm look.
   The claim is "this is the look your colour most resembles", never "this was
   graded". Confidence is therefore a **margin** — how far the best match beats
   the runner-up — which roughly doubles precision (47% of correct matches
   flagged vs 23% of incorrect ones).
4. **Global measurement, local edits.** Lightroom masks are invisible to a
   whole-frame statistic.
5. **Exposure coupling.** Brightening moves pixels between L\* zones, so a zone
   statistic shifts even when the tint did not. Zones that fall below 100 pixels
   report zero with zero coverage rather than inventing a hue.
6. **Small N.** Fifteen reference looks is a demo library, not a taxonomy.

---

## Deployment size

Deploying to a free tier makes install size and cold-start time matter, so the
one heavy dependency was removed rather than tolerated.

`scikit-image` was used for exactly one function, `rgb2lab`, and it pulls in
`scipy`. Together they were **143 MB of a 206 MB install** — 70% of the
deployment for one colour-space conversion. `looklab/cielab.py` now implements
that conversion in about forty lines of numpy.

| | full | slim |
|---|---|---|
| `site-packages` | 319.5 MB | **125.8 MB** (61% smaller) |
| Runtime RSS | ~143 MB | ~143 MB |
| Cold start | ~2 s | ~2 s |

Replacing a reference implementation with your own is only defensible if you
can show they agree, so `tests/test_cielab.py` asserts **bit-for-bit equality**
with scikit-image — max difference exactly `0.000e+00` across the sRGB cube,
the greyscale ramp, both piecewise knees, random images and every base plate.
That required matching scikit-image's *legacy rounded* CIE constants (0.008856
and 7.787) rather than the exact fractions; using the exact values shifts L\* by
1.6e-4, which is invisible in a photograph but would have turned a proof into
an approximation and silently invalidated the committed knowledge base.

scikit-image remains in `requirements-dev.txt` purely so that equivalence test
can run. Nothing in the running application imports it, and
`test_signature_pipeline_still_works_without_skimage` blocks the import to
prove it.

---

## Data provenance

**Base plates** — 16 photographs from Wikimedia Commons, fetched by
`python -m tools.fetch_plates`. Only public-domain, CC0, CC BY and CC BY-SA
images are accepted; anything else is skipped rather than guessed at. Licence,
author and source URL for every plate are in `looklab/data/plates/MANIFEST.json`
and rendered in `looklab/data/plates/ATTRIBUTION.md`.

Split by content category so the held-out set spans the same subjects but
entirely different photographs: `portrait_daylight_0` builds the knowledge base,
`portrait_daylight_1` is held out. A synthetic 24-patch colour chart is also
held out — it has no subject matter at all, so it isolates the grade completely.

If the plates have not been downloaded, the sample photographs bundled inside
scikit-image are used instead, so a fresh clone works offline.

**Glossary** — 15 colour-science definitions from the Wikipedia REST summary
endpoint (`python -m tools.fetch_knowledge`), stored as at most the first two
sentences of each abstract with its article URL, revision id and CC BY-SA 4.0
licence. Attribution renders beside every definition in the UI.

---

## Language model (optional)

The default narrator is `DemoChatModel`: deterministic, no API key, no network.
It composes every sentence from values the graph already computed.

To use Gemini for the prose instead, copy `.env.example` to `.env` and set
`LOOKLAB_MODEL=google` with `GEMINI_API_KEYS=key1,key2,...`. `.env` is
gitignored and **keys must never be committed**.

`looklab/gemini.py` pools the keys because a single key is not reliable enough:
measured against the live API, roughly 30% of calls to a popular model return
`503 high demand` even with a perfectly valid credential. The pool rotates,
distinguishes a bad key (401/403 — retired permanently) from a busy service
(429/503 — cooled off and retried), and enforces a wall-clock deadline so a web
request never waits on retries. Measured: 20/20 calls succeeded across 10 keys
with 14 transient 503s absorbed invisibly.

Model choice is also measured, not assumed:

| Model | Keys OK | Median latency |
|---|---|---|
| `gemini-3.1-flash-lite` | **10/10** | 2.2s |
| `gemini-3.5-flash-lite` | 5/10 | 1.2s |
| `gemini-flash-lite-latest` | 0/10 | — |
| `gemini-3.6-flash` | 0/10 | — |

The `-latest` aliases are the *most* contended, which is the opposite of the
intuition that they are safest. A `404` on a model retires it and the pool walks
down a fallback ladder rather than blaming the credentials.

The model is given a finished deterministic draft and told to change only the
wording. Any failure — no keys, all rate-limited, deadline expired, empty or
suspiciously long completion — returns the draft unchanged, so a hallucinated
slider value is structurally impossible.

---

## Testing

```
pytest -q                    209 passed
python -m tools.stress        49/49 checks passed
```

The stress harness drives a real uvicorn process over HTTP:

- **Load** — 25.6 req/s across 8 concurrent clients, p95 345 ms.
- **Lost updates** — 12 concurrent posts to *one* thread must produce 12 turns.
  Without a per-thread lock they silently collapse to 1, with no error raised.
- **Fuzz** — 28 hostile inputs: oversized bodies, null bytes, CJK/RTL/emoji,
  prompt injection, SQL and template payloads, malformed base64, 1600×1200
  JPEGs, every field at its boundary. A 4xx is a pass, a 5xx is a bug. Zero 5xx.
- **Soak** — 25 turns on one thread: history 1→49 messages while trimming held
  tokens at ≤787 against an 800 budget, and the recipe grew monotonically 6→58.

Two test-design choices worth noting. Thread isolation is asserted
**negatively** — thread B must *not* contain thread A's recipe — because a
same-process read passes trivially even with no persistence at all. And
long-term recall is asserted after discarding the store and rebuilding it from
the same file, which is the only way to distinguish real persistence from a
dictionary that happens to still be in memory.

---

## Bugs this codebase defends against

Each was reproduced against the installed versions before being fixed.

| Trap | What happens | Fix |
|---|---|---|
| `SqliteSaver.from_conn_string` | It is a `@contextmanager` that closes the connection on exit; assigning it gives you a `_GeneratorContextManager` | construct `SqliteSaver(sqlite3.connect(...))` directly |
| `SqliteStore` default isolation | first `put()` raises `cannot start a transaction within a transaction` | `isolation_level=None` |
| `async def` endpoints | `SqliteSaver` has no async methods; `await ainvoke` raises on request #1 | plain `def`, threadpooled by Starlette |
| numpy in state | `TypeError: not msgpack serializable: numpy.float64` | cast at the node boundary (`jsonable`) |
| trim-then-write-back | silent no-op — `add_messages` upserts and never deletes, so telemetry lies | trimming is ephemeral; `RemoveMessage` for real deletion |
| `trim_messages` at a tight budget | returns one message or none, no exception | floor guard + UI clamp at 200 |
| router as node *and* path function | returning a bare string raises `InvalidUpdateError` | node returns `{"intent": ...}`; path function reads it |
| stale unannotated channels | an old `identify` match still shows during a later `chat` turn | branch entry resets `matches`/`telemetry` |
| concurrent same-thread posts | 8 invokes silently collapse to 1 | per-`thread_id` lock in the server |
| `b64decode(validate=False)` | discards invalid characters, so garbage looks like "no image" and returns 200 | `validate=True` → 400 |
| default-argument DB paths | `def f(path=DEFAULT)` binds once, so tests silently wrote to the demo database | resolve the path at call time |
| system prompt in telemetry | pane read `241 → 252` tokens: trimming appearing to *add* context | count history only, on both sides |
| LAB hue as a wheel value | the two differ by 9–59°, so "hue 296" sent you to the wrong colour | measured conversion table |

---

## Layout

```
looklab/
├── graph.py        state, nodes, conditional edge, compile   (T1, T2)
├── context.py      filter + trim + telemetry                 (T3)
├── memory.py       profile extraction and store access       (T4)
├── persistence.py  SqliteSaver / SqliteStore construction    (T2, T4)
├── cielab.py       sRGB -> CIELAB in pure numpy (replaces scikit-image)
├── color.py        CIELAB signature, distance, matching
├── grading.py      numpy Lightroom-slider simulator
├── rules.py        delta -> slider steps, fault checks
├── kb.py           knowledge-base loading and matching
├── looks.py        15 reference looks with ground-truth sliders
├── plates.py       base plates and the train/test split
├── llm.py          DemoChatModel + optional Gemini narrator
├── gemini.py       multi-key pool with rotation and failover
├── seed.py         two demo threads, replayed on cold start
├── server.py       FastAPI
├── kb.json         built by tools/build_kb.py
├── knowledge.json  built by tools/fetch_knowledge.py
└── web/index.html  the whole front end, one file, no build

tools/
├── build_kb.py       KB from plates x looks
├── calibrate.py      the evaluation above
├── tune_matcher.py   cross-validated hyperparameter selection
├── fetch_plates.py   licensed photographs from Wikimedia Commons
├── fetch_knowledge.py cited colour-science glossary
└── stress.py         load, fuzz and soak against a live server

tests/
├── test_rubric.py  T1-T4, named to map onto the rubric
├── test_api.py     HTTP-level, including concurrency
├── test_cielab.py  proves cielab.py matches scikit-image bit-for-bit
└── test_domain.py  colour engine, simulator, KB, rules
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | one turn → reply, intent, recipe, telemetry, profile |
| GET | `/state/{thread_id}` | messages, recipe, images, telemetry, branch |
| GET | `/profile/{user_id}` | long-term store contents |
| GET | `/graph` | mermaid source from the compiled graph |
| GET | `/looks` | the reference library |
| GET | `/knowledge` | cited colour-science glossary (`?q=` filters) |
| GET | `/plates` | base-plate provenance and licences |
| GET | `/models` | narrator status and per-key Gemini health |
| GET | `/health` | liveness and configuration |
| GET | `/threads` | thread list, used by the switcher |
| DELETE | `/thread/{thread_id}` | delete a conversation and its images, reclaim disk |
