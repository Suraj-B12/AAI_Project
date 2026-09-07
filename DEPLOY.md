# Deploying LookLab

Goal: one HTTPS link you can send your teacher, where the whole app works.

---

## First, the thing that changes the plan

**Do not split the front end onto Vercel and the backend onto Hugging Face.**
Two reasons, both specific to this project:

1. **There is no separate front end to deploy.** The UI is a single
   `looklab/web/index.html` that FastAPI serves at `/`. It has no build step
   and no `node_modules`. Splitting it means adding a CORS boundary that does
   not currently exist, rewriting every `fetch()` to point at an absolute
   backend URL, and keeping two deployments in sync — for zero benefit, since
   serving one static file costs the backend nothing.

2. **Hugging Face Spaces is no longer free for this.** As of 2026, only
   *Static* Spaces are free. Creating a Gradio or Docker Space
   [requires a paid plan](https://huggingface.co/docs/hub/en/spaces-overview)
   (PRO for personal accounts), with a narrow exception for two Gradio Spaces
   on ZeroGPU. LookLab is a FastAPI app, so it needs the Docker SDK, so on a
   free account it cannot be created there at all.

**Deploy the whole thing as one container to one host.** One image, one port,
one URL, no CORS.

---

## Where to host it

Measured constraints for this app: **125 MB** of installed dependencies,
**~143 MB** RAM at runtime, **~2 s** cold start, and it needs a writable
directory for SQLite.

| Platform | Free? | Card needed? | Sleeps? | Verdict |
|---|---|---|---|---|
| **Render** | Yes — 750 instance-hrs/mo | **No** | After 15 min idle, ~1 min wake | **Recommended** |
| **Koyeb** | Yes — 1 web service | **No** | Scale-to-zero | Good backup; 0.1 vCPU is slow |
| Hugging Face Spaces | Docker needs **PRO** | — | 48 h idle | Not viable free |
| Vercel | Frontend only | No | Serverless | **Wrong shape** — see below |
| Fly.io | No free tier since 2024 | Yes | — | Not free |
| Railway | $5 trial credit only | Yes | — | Runs out |

**Why not Vercel for the backend**, even now that dependencies are slim:
Vercel's Python runtime is serverless, so the filesystem is ephemeral *per
invocation*. LookLab's checkpointer writes SQLite — that is what makes thread
switching work — and on Vercel that state would vanish between requests. T2
and T4 would be undemonstrable. Vercel is excellent for static sites and
Next.js; this is neither.

Sources:
[Render free tier](https://render.com/docs/free) ·
[HF Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) ·
[Koyeb pricing](https://www.koyeb.com/pricing)

---

## Recommended: Render (about 10 minutes)

You need a GitHub account and a Render account. No credit card.

### 1. Sign up
Go to <https://dashboard.render.com/register> and **Sign up with GitHub**.
Authorise Render to read your repositories.

### 2. Create the service
- Click **New +** → **Web Service**.
- Choose **Build and deploy from a Git repository** → **Next**.
- Find `Suraj-B12/AAI_Project` and click **Connect**.
  (If it is not listed: **Configure account** → grant access to the repo.)

### 3. Settings
Render reads `render.yaml` from the repo, so most fields fill themselves. Check:

| Field | Value |
|---|---|
| Name | `looklab` (this becomes your URL) |
| Language / Runtime | **Docker** |
| Branch | `main` |
| Instance Type | **Free** |
| Health Check Path | `/health` |

Leave Build Command and Start Command **empty** — the `Dockerfile` handles both.

### 4. Deploy
Click **Create Web Service**. The first build takes **5–10 minutes** (it
installs dependencies and bakes the tiktoken vocabulary into the image).
Watch the **Logs** tab. You are done when you see:

```
Uvicorn running on http://0.0.0.0:10000
==> Your service is live 🎉
```

### 5. Your link
`https://looklab.onrender.com` (or whatever name you chose). Open it — you
should see two pre-seeded threads, a filled profile pane and a recipe.

### 6. Check it before sending it
```bash
curl https://looklab.onrender.com/health
```
Expect `{"ok":true,"looks":15,...}`.

---

## Before you send the link to your teacher

**Warn them about the first load.** On the free tier the service sleeps after
15 minutes idle and takes **30–60 seconds** to wake. A cold page looks broken
if you do not expect it. Send something like:

> LookLab: https://looklab.onrender.com
> It is on a free tier and sleeps when idle — the first load takes up to a
> minute to wake up. Everything after that is fast.

**Wake it yourself a few minutes before they look**, by opening the link.

**Optional — keep it awake while it is being graded.** A free
[UptimeRobot](https://uptimerobot.com) monitor pinging `/health` every 5
minutes prevents sleep. Only do this for the day or two around grading: it
burns your 750 monthly instance-hours (750 h ÷ 24 h ≈ 31 days, so continuous
pinging uses almost exactly your whole month).

---

## What your teacher will see

The app seeds itself on every cold start (`looklab/seed.py`), so nothing is
blank. On first load:

- **Two threads**, `warm-portrait` and `cold-landscape`, with visibly
  different recipes — that is the T2 checkpointer evidence.
- **A filled profile pane** that stays identical when they switch threads —
  that is the T4 store evidence.
- **Live trim telemetry**, and a budget slider they can drag to watch the
  numbers move — T3.
- **The graph diagram**, emitted by the compiled graph itself, with the branch
  taken highlighted — T1.

Suggested one-line brief for them:

> Type "how do I get the teal and orange look?" and watch the Recipe pane
> grow. Then switch threads in the left column — the recipe changes but the
> profile does not.

---

## Optional: enable Gemini narration

The app is fully functional without any API key — the deterministic narrator
composes every sentence from computed values. To use Gemini for the prose:

1. Render dashboard → your service → **Environment**.
2. Add `LOOKLAB_MODEL` = `google` (an **Environment Variable**).
3. Add `GEMINI_API_KEYS` = `key1,key2,key3` as a **Secret**, not a plain
   variable.
4. **Save, rebuild**.

**Never put keys in `render.yaml` or any committed file** — this repository is
public. `.env` is gitignored for the same reason.

If the keys are missing, invalid or rate-limited, the app silently falls back
to the deterministic narrator. It cannot break because of a key.

---

## Backup: Koyeb

If Render misbehaves, Koyeb is the same shape and also needs no card.

1. <https://app.koyeb.com> → sign up with GitHub.
2. **Create Web Service** → **GitHub** → pick `AAI_Project`.
3. Builder: **Dockerfile**. Instance: **Free** (0.1 vCPU / 512 MB).
4. **Exposed port: 8000**, and add an environment variable `PORT` = `8000`.
5. Health check path: `/health`.
6. Deploy.

Koyeb does not sleep the same way Render does, but 0.1 vCPU makes each turn
noticeably slower. Fine for grading; worse for a live demo.

---

## Running it locally (the fallback that always works)

If hosting fails on the day, this takes two minutes on any machine:

```bash
git clone https://github.com/Suraj-B12/AAI_Project.git
cd AAI_Project
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn looklab.server:app --port 8077
```

Then open <http://127.0.0.1:8077>. No API key and no network needed — the
knowledge base and plates are committed and the narrator is deterministic.

To run the tests as well:

```bash
pip install -r requirements-dev.txt
pytest -q                 # 129 tests
python -m tools.stress    # 49 load/fuzz/soak checks
```

---

## With Docker, anywhere

```bash
docker build -t looklab .
docker run -p 8077:8077 -e PORT=8077 looklab
```

The same image runs on Render, Koyeb, Google Cloud Run and Hugging Face
Spaces (Docker SDK, paid). It listens on `$PORT`, defaults to 7860 (the port
Hugging Face requires), and runs as a non-root user because several platforms
refuse root containers.

---

## Known limitations of free hosting

**Storage is ephemeral.** Every free tier here wipes the disk on redeploy and
on the restart after a sleep. Conversations a grader has *during* a session
persist normally; they do not survive a redeploy. This does not weaken the T4
claim — that long-term memory survives a restart is proven by
`test_t4_profile_survives_a_completely_fresh_store`, which rebuilds the store
from the same file and asserts recall. The seeding on boot exists precisely so
an ephemeral disk still presents a populated app.

**One instance, one worker.** The Dockerfile runs `--workers 1` deliberately.
Multiple workers would each open the same SQLite WAL file, and the per-thread
lock that prevents lost updates is per-process. One worker handles far more
load than a grading session needs — measured at 27.7 requests/second.
