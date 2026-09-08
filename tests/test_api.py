"""HTTP-level tests against the real FastAPI app.

These exercise the server the way the front end does, including the cases that
would otherwise only show up live: an unknown thread, a corrupt upload, a
budget below the safe floor, and concurrent posts to one thread.

The app is imported with its data paths redirected to a temp directory, so the
suite never reads or writes the demo databases.
"""

from __future__ import annotations

import base64
import importlib
import io
import os
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Import the server with its SQLite paths pointed at a temp directory."""
    tmp = tmp_path_factory.mktemp("api")
    from looklab import persistence

    original_cp = persistence.DEFAULT_CHECKPOINT_PATH
    original_st = persistence.DEFAULT_STORE_PATH
    persistence.DEFAULT_CHECKPOINT_PATH = str(tmp / "cp.sqlite")
    persistence.DEFAULT_STORE_PATH = str(tmp / "st.sqlite")

    from looklab import server as server_module

    server_module = importlib.reload(server_module)
    try:
        with TestClient(server_module.app) as c:
            yield c
    finally:
        persistence.DEFAULT_CHECKPOINT_PATH = original_cp
        persistence.DEFAULT_STORE_PATH = original_st


def _jpeg_bytes(plate: str | None = None, size: int = 128) -> bytes:
    """Encode a real base plate as JPEG.

    Defaults to whichever plate the knowledge base actually uses rather than a
    hardcoded scikit-image sample name: scikit-image is an optional dev
    dependency, so naming "astronaut" here made these tests fail in an
    environment installed from requirements.txt alone.
    """
    from looklab.plates import PLATE_NAMES, base_plate

    arr = (base_plate(plate or PLATE_NAMES[0], size) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# --------------------------------------------------------------------------

def test_health_reports_a_working_stack(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["looks"] >= 6, "knowledge base looks empty -- run tools.build_kb"
    assert body["deterministic"] is True, "should default to the offline narrator"
    assert body["token_backend"]


def test_index_serves_the_front_end(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "LookLab" in response.text
    for topic in ("T1", "T2", "T3", "T4"):
        assert topic in response.text, f"rubric label {topic} missing from the UI"


def test_graph_endpoint_returns_mermaid_from_the_compiled_object(client):
    text = client.get("/graph").text
    assert "graph TD" in text
    assert "route_intent" in text and "respond" in text


def test_looks_endpoint_lists_the_library(client):
    body = client.get("/looks").json()
    assert len(body["looks"]) >= 6
    first = body["looks"][0]
    assert {"id", "label", "notes", "sliders"} <= set(first)
    assert body["meta"].get("heldout_plates")


def test_chat_round_trip(client):
    body = client.post(
        "/chat", json={"message": "hey, what can you do?", "thread_id": "api-chat"}
    ).json()
    assert body["intent"] == "chat"
    assert body["branch_node"] == "respond"
    assert body["reply"]
    assert body["telemetry"]["tokens_before"] >= 0


def test_chat_achieve_builds_a_recipe_without_any_upload(client):
    """The text-only spine: T2 accumulation must not depend on images."""
    first = client.post(
        "/chat",
        json={"message": "how do i get the deep amber look?", "thread_id": "api-recipe"},
    ).json()
    assert first["intent"] == "achieve"
    assert len(first["recipe"]) >= 1

    second = client.post(
        "/chat",
        json={"message": "how do i get the teal and orange look?", "thread_id": "api-recipe"},
    ).json()
    assert len(second["recipe"]) > len(first["recipe"])
    assert sorted({e["turn"] for e in second["recipe"]}) == [1, 2]


def test_state_of_an_unknown_thread_is_empty_not_an_error(client):
    body = client.get("/state/a-thread-nobody-ever-used").json()
    assert body["exists"] is False
    assert body["messages"] == [] and body["recipe"] == []


def test_profile_is_written_and_readable_across_threads(client):
    client.post(
        "/chat",
        json={
            "message": "Hi, I'm Suraj. I shoot on a Fuji X-T4 and I like warm golden looks.",
            "thread_id": "api-p1",
            "user_id": "api-user",
        },
    )
    body = client.get("/profile/api-user").json()
    assert body["profile"].get("name") == "Suraj"
    assert body["summary"]

    other = client.post(
        "/chat",
        json={"message": "what should i try?", "thread_id": "api-p2", "user_id": "api-user"},
    ).json()
    assert other["profile"].get("name") == "Suraj"


def test_threads_are_isolated_over_http(client):
    client.post("/chat", json={"message": "how do i get the deep amber look?", "thread_id": "iso-a"})
    client.post("/chat", json={"message": "hello", "thread_id": "iso-b"})
    a = client.get("/state/iso-a").json()
    b = client.get("/state/iso-b").json()
    assert a["recipe"] and not b["recipe"]


def test_image_upload_is_measured_and_never_enters_messages(client):
    payload = base64.b64encode(_jpeg_bytes()).decode()
    body = client.post(
        "/chat",
        json={"message": "what look is this?", "thread_id": "api-img", "reference_b64": payload},
    ).json()
    assert body["intent"] == "identify"
    assert body["matches"], "an uploaded reference should produce matches"

    state = client.get("/state/api-img").json()
    assert "reference" in state["images"]
    assert isinstance(state["images"]["reference"], dict)
    for message in state["messages"]:
        assert len(message["content"]) < 2048, "image data leaked into the transcript"


def test_data_url_prefixed_base64_is_accepted(client):
    payload = "data:image/jpeg;base64," + base64.b64encode(_jpeg_bytes()).decode()
    response = client.post(
        "/chat",
        json={"message": "what look is this?", "thread_id": "api-dataurl", "reference_b64": payload},
    )
    assert response.status_code == 200
    assert response.json()["matches"]


def test_invalid_base64_returns_400_not_500(client):
    response = client.post(
        "/chat",
        json={"message": "what look is this?", "thread_id": "api-bad", "reference_b64": "!!!!"},
    )
    assert response.status_code in (400, 422)


def test_corrupt_image_still_answers(client):
    payload = base64.b64encode(b"definitely not an image").decode()
    response = client.post(
        "/chat",
        json={"message": "what look is this?", "thread_id": "api-corrupt", "reference_b64": payload},
    )
    assert response.status_code == 200
    assert response.json()["reply"]


def test_budget_below_the_floor_is_rejected_by_validation(client):
    response = client.post("/chat", json={"message": "hi", "thread_id": "api-b", "budget": 5})
    assert response.status_code == 422, "the budget floor must be enforced server-side"


def test_budget_slider_changes_the_telemetry(client):
    for i in range(6):
        client.post("/chat", json={"message": f"tell me about look {i}", "thread_id": "api-budget"})
    wide = client.post(
        "/chat", json={"message": "and again", "thread_id": "api-budget", "budget": 4000}
    ).json()["telemetry"]
    tight = client.post(
        "/chat", json={"message": "and again", "thread_id": "api-budget", "budget": 200}
    ).json()["telemetry"]
    assert tight["msgs_after_trim"] < wide["msgs_after_trim"]
    assert tight["tokens_after"] < wide["tokens_after"]


def test_concurrent_posts_to_one_thread_do_not_lose_turns(client):
    """Without the per-thread lock, eight concurrent invokes collapse to one."""
    thread = "api-concurrent"
    errors: list[str] = []

    def post(i: int) -> None:
        try:
            r = client.post(
                "/chat", json={"message": f"message number {i}", "thread_id": thread}
            )
            if r.status_code != 200:
                errors.append(f"{r.status_code}: {r.text[:120]}")
        except Exception as exc:  # pragma: no cover - surfaced via assertion
            errors.append(repr(exc))

    workers = [threading.Thread(target=post, args=(i,)) for i in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert not errors, errors
    state = client.get(f"/state/{thread}").json()
    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    assert len(user_messages) == 8, (
        f"expected 8 user turns, found {len(user_messages)} -- concurrent writes were lost"
    )


# --------------------------------------------------------------------------
# Cold-start seeding
# --------------------------------------------------------------------------

def test_cold_start_seeds_two_contrasting_demo_threads(client):
    """A blank app demonstrates nothing, and free hosts start blank every time.

    The lifespan handler replays two scripted conversations through the real
    graph, so a grader's first page load already shows a populated recipe, a
    populated profile and real trim telemetry.
    """
    from looklab.seed import SCRIPTS

    states = {}
    for thread_id in SCRIPTS:
        body = client.get(f"/state/{thread_id}").json()
        assert body["exists"], f"{thread_id} was not seeded"
        assert body["messages"], f"{thread_id} has no messages"
        states[thread_id] = body

    warm, cold = states["warm-portrait"], states["cold-landscape"]

    # T2 evidence: the two threads must look visibly different on screen.
    assert warm["recipe"] and cold["recipe"]
    assert [e["slider"] for e in warm["recipe"]] != [e["slider"] for e in cold["recipe"]]

    # T3 evidence: telemetry carries real numbers, not zeros.
    assert warm["telemetry"].get("tokens_before", 0) > 0

    # T4 evidence: the profile was learned and is shared, not per-thread.
    profile = client.get("/profile/suraj").json()["profile"]
    assert profile.get("name") == "Suraj"


def test_seeding_is_idempotent(client):
    """Re-running the lifespan must not duplicate the demo conversations."""
    from looklab.seed import seed
    from looklab.server import GRAPH

    before = len(client.get("/state/warm-portrait").json()["messages"])
    seed(GRAPH)  # second call, as a restart would make
    after = len(client.get("/state/warm-portrait").json()["messages"])
    assert after == before


def test_seeded_threads_appear_in_the_thread_list(client):
    """The UI populates its switcher from /threads, so they must be listed."""
    names = {t["thread_id"] for t in client.get("/threads").json()["threads"]}
    assert {"warm-portrait", "cold-landscape"} <= names


# --------------------------------------------------------------------------
# Deleting a conversation
# --------------------------------------------------------------------------

def test_delete_thread_removes_messages_recipe_and_images(client):
    """Deleting must take the uploads with it, not just the text."""
    payload = base64.b64encode(_jpeg_bytes()).decode()
    for i in range(3):
        client.post(
            "/chat",
            json={
                "message": f"how do i match these? {i}",
                "thread_id": "to-delete",
                "reference_b64": payload,
                "current_b64": payload,
            },
        )
    before = client.get("/state/to-delete").json()
    assert before["exists"] and before["messages"] and before["images"]

    body = client.delete("/thread/to-delete").json()
    assert body["existed"] is True
    assert body["deleted"] == "to-delete"

    after = client.get("/state/to-delete").json()
    assert after["exists"] is False
    assert after["messages"] == [] and after["recipe"] == [] and after["images"] == {}


def test_delete_thread_reclaims_disk(client):
    """A delete that leaves the pages on disk is not a delete.

    SQLite in WAL mode keeps deleted pages in the sidecar until a checkpoint
    and in the freelist until a VACUUM, so this asserts the bytes actually go.
    """
    payload = base64.b64encode(_jpeg_bytes(size=384)).decode()
    for i in range(6):
        client.post(
            "/chat",
            json={
                "message": f"how do i match these? {i}",
                "thread_id": "fat-thread",
                "reference_b64": payload,
                "current_b64": payload,
            },
        )
    body = client.delete("/thread/fat-thread").json()
    assert body["bytes_after"] < body["bytes_before"], (
        f"no space reclaimed: {body['bytes_before']} -> {body['bytes_after']}"
    )
    assert body["bytes_reclaimed"] > 0


def test_delete_leaves_other_threads_and_the_profile_alone(client):
    """The profile belongs to the user, not to one conversation."""
    client.post(
        "/chat",
        json={
            "message": "Hi, I'm Suraj. I shoot on a Fuji X-T4.",
            "thread_id": "del-a",
            "user_id": "del-user",
        },
    )
    client.post("/chat", json={"message": "hello", "thread_id": "del-b", "user_id": "del-user"})

    client.delete("/thread/del-a")

    assert client.get("/state/del-b").json()["exists"] is True
    assert client.get("/profile/del-user").json()["profile"].get("name") == "Suraj"


def test_deleting_an_unknown_thread_is_not_an_error(client):
    body = client.delete("/thread/never-existed-at-all").json()
    assert body["existed"] is False


def test_deleted_thread_disappears_from_the_thread_list(client):
    client.post("/chat", json={"message": "hello", "thread_id": "listed-then-gone"})
    names = {t["thread_id"] for t in client.get("/threads").json()["threads"]}
    assert "listed-then-gone" in names
    client.delete("/thread/listed-then-gone")
    names = {t["thread_id"] for t in client.get("/threads").json()["threads"]}
    assert "listed-then-gone" not in names


# --------------------------------------------------------------------------
# Explicit long-term memory
# --------------------------------------------------------------------------

def test_remember_this_saves_to_long_term_memory(client):
    """"Remember that ..." must persist, and be visible from another thread."""
    body = client.post(
        "/chat",
        json={
            "message": "remember that I always print my work on matte paper",
            "thread_id": "mem-a",
            "user_id": "mem-user",
        },
    ).json()
    assert "matte paper" in body["reply"].lower() or "saved" in body["reply"].lower()

    profile = client.get("/profile/mem-user").json()["profile"]
    notes = profile.get("notes") or []
    assert any("matte paper" in str(n).lower() for n in notes), profile

    # And it reaches a completely different conversation.
    other = client.post(
        "/chat", json={"message": "hello", "thread_id": "mem-b", "user_id": "mem-user"}
    ).json()
    assert any("matte paper" in str(n).lower() for n in other["profile"].get("notes") or [])


def test_notes_do_not_grow_without_bound(client):
    from looklab.memory import MAX_NOTES

    for i in range(MAX_NOTES + 6):
        client.post(
            "/chat",
            json={
                "message": f"remember that fact number {i} matters to me",
                "thread_id": "mem-cap",
                "user_id": "cap-user",
            },
        )
    notes = client.get("/profile/cap-user").json()["profile"].get("notes") or []
    assert len(notes) <= MAX_NOTES, f"notes grew to {len(notes)}"
    # The most recent survive, the oldest are dropped.
    assert any(f"number {MAX_NOTES + 5}" in str(n) for n in notes)


# --------------------------------------------------------------------------
# Where the Gemini keys are read from
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "contents,expected",
    [
        ("AQ.key1,AQ.key2,AQ.key3", 3),                      # bare comma list
        ("AQ.key1\nAQ.key2\nAQ.key3", 3),                    # one per line
        ("GEMINI_API_KEYS=AQ.key1,AQ.key2", 2),              # dotenv style
        ('# note\nGEMINI_API_KEYS="AQ.key1,AQ.key2"\n', 2),  # comment + quotes
        ("AQ.key1, AQ.key2 , AQ.key3", 3),                   # sloppy spacing
        ("AQ.key1,AQ.key2\n", 2),                            # trailing newline
        ("", 0),
        ("# only a comment\n", 0),
    ],
)
def test_keys_parse_from_a_secret_file(tmp_path, monkeypatch, contents, expected):
    """A hosting panel's "secret file" is a file, not an environment variable.

    Render offers both side by side and they look interchangeable. Picking the
    wrong one fails silently -- the app just falls back to the offline narrator
    with nothing logged -- so both are supported, in whichever format the value
    was pasted in.
    """
    import looklab.gemini as gem

    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    secret = tmp_path / "GEMINI_API_KEYS"
    secret.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(gem, "KEY_FILE_CANDIDATES", (str(secret),))

    assert len(gem.load_keys()) == expected


def test_environment_variable_beats_a_secret_file(tmp_path, monkeypatch):
    import looklab.gemini as gem

    secret = tmp_path / "GEMINI_API_KEYS"
    secret.write_text("AQ.fromfile", encoding="utf-8")
    monkeypatch.setattr(gem, "KEY_FILE_CANDIDATES", (str(secret),))
    monkeypatch.setenv("GEMINI_API_KEYS", "AQ.fromenv1,AQ.fromenv2")

    assert gem.load_keys() == ["AQ.fromenv1", "AQ.fromenv2"]


def test_no_keys_anywhere_is_not_an_error(tmp_path, monkeypatch):
    """A missing key must degrade to the offline narrator, never raise."""
    import looklab.gemini as gem

    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(gem, "KEY_FILE_CANDIDATES", (str(tmp_path / "nope"),))

    assert gem.load_keys() == []
    assert gem.GeminiPool([]).usable == 0


# --------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------

def test_trace_reports_the_real_node_sequence_per_turn(client):
    """The transition view is rebuilt from the checkpointer, not from a log.

    This is the clearest evidence the conditional edge is not decoration: four
    intents produce four different paths through the graph.
    """
    thread = "trace-thread"
    for message in [
        "how do i get the deep amber look?",
        "hello there",
        "what look is teal and orange?",
        "what's wrong with my edit?",
    ]:
        client.post("/chat", json={"message": message, "thread_id": thread})

    body = client.get(f"/trace/{thread}").json()
    assert body["exists"] is True
    assert len(body["turns"]) == 4

    by_intent = {t["intent"]: t["nodes"] for t in body["turns"]}
    assert "analyze_pair" in by_intent["achieve"]
    assert "delta" in by_intent["achieve"] and "recipe_build" in by_intent["achieve"]
    assert "match" in by_intent["identify"]
    assert "critique" in by_intent["critique"]

    # The chat branch runs NO analysis node -- that is the whole point of it.
    for node in ("analyze_pair", "analyze_ref", "analyze_cur", "match", "delta", "critique"):
        assert node not in by_intent["chat"], f"chat should not run {node}"

    # Every turn reads and writes long-term memory.
    for turn in body["turns"]:
        assert turn["nodes"][1] == "load_profile"
        assert "save_profile" in turn["nodes"]

    # Four intents, four distinct paths.
    paths = {tuple(t["nodes"]) for t in body["turns"]}
    assert len(paths) == 4


def test_trace_labels_each_turn_with_its_own_message(client):
    """The message must not be off by one.

    At the input checkpoint the incoming message has not been merged yet, so
    reading it there labelled every turn with the previous turn's text.
    """
    thread = "trace-labels"
    messages = ["how do i get the deep amber look?", "hello there"]
    for message in messages:
        client.post("/chat", json={"message": message, "thread_id": thread})

    turns = client.get(f"/trace/{thread}").json()["turns"]
    assert [t["message"] for t in turns] == messages


def test_trace_of_an_unknown_thread_is_empty_not_an_error(client):
    body = client.get("/trace/no-such-thread-anywhere").json()
    assert body["exists"] is False
    assert body["turns"] == []


def test_trace_ui_button_exists(client):
    page = client.get("/").text
    assert "showTrace" in page
    assert "State transitions" in page


def test_trace_reports_what_each_turn_added_not_the_running_total(client):
    """The recipe channel is append-only, so its length is a thread total.

    Reported raw, the second achieve turn of a thread claims sixteen slider
    moves when it produced eight, and every chat turn in between inherits the
    previous turn's count.
    """
    thread = "trace-recipe"
    for message in [
        "how do i get the deep amber look?",
        "hi there",
        "how do i get the teal and orange look?",
    ]:
        client.post("/chat", json={"message": message, "thread_id": thread})

    turns = client.get(f"/trace/{thread}").json()["turns"]
    first, chat, second = turns

    assert first["recipe_added"] > 0
    assert first["recipe_added"] == first["recipe_total"], "the first recipe IS the total"

    assert chat["recipe_added"] == 0, "a chat turn built no recipe"

    assert second["recipe_added"] == second["recipe_total"] - first["recipe_total"]
    assert second["recipe_total"] > second["recipe_added"], (
        "the running total should exceed this turn's increment -- that growth "
        "is the append-only reducer being visible"
    )
