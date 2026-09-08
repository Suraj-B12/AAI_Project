"""The chat branch: every message class gets a real answer.

This file exists because of one transcript. A user told the app their camera
and then asked four ordinary questions:

    can my camera shoot arri log?   ->  Got it - you shoot on a sony alpha6700.
    what log can I shoot?           ->  Got it - you shoot on a sony alpha6700.
    are you hallucinating?          ->  Got it - you shoot on a sony alpha6700.
    log vs rec709                   ->  Got it - you shoot on a sony alpha6700.

The chat branch had three outcomes and a catch-all, and the catch-all
swallowed everything: measured, fourteen out of fourteen free-form questions
returned an identical non-answer. These tests pin each message class to a
useful reply, all on the offline path with no model configured.
"""

from __future__ import annotations

import pytest

from looklab.graph import build_graph, new_turn_input
from looklab.knowledge import classify_message
from looklab.memory import extract_profile_facts, forget_request
from looklab.persistence import make_checkpointer, make_store


@pytest.fixture()
def app(tmp_path):
    return build_graph(
        make_checkpointer(str(tmp_path / "cp.sqlite")),
        make_store(str(tmp_path / "st.sqlite")),
    )


def say(app, text, thread="c", user="u"):
    result = app.invoke(
        new_turn_input(text, user_id=user), {"configurable": {"thread_id": thread}}
    )
    return result["messages"][-1].content, result


CANNED = ("i will keep that in mind", "i am a colour-grading assistant. ask me what")


def assert_not_canned(reply, question):
    lowered = reply.lower()
    for phrase in CANNED:
        assert phrase not in lowered, f"{question!r} still returns the canned non-answer"


# --------------------------------------------------------------------------
# The exact transcript that prompted this
# --------------------------------------------------------------------------

def test_the_reported_transcript_no_longer_repeats_itself(app):
    say(app, "im john, remember me")
    say(app, "I shoot on Sony alpha6700, remember that too")

    questions = [
        "can my camera shoot arri log?",
        "what log can I shoot?",
        "are you hallucinating?",
        "log vs rec709",
    ]
    replies = []
    for question in questions:
        reply, _ = say(app, question)
        assert_not_canned(reply, question)
        replies.append(reply)

    # Not necessarily four distinct replies: offline, two questions of the same
    # class ("arri log" and "what log") both get the same honest decline, which
    # is correct. What must not happen is all four collapsing onto one answer,
    # which is what the transcript showed.
    assert len(set(replies)) >= 3, (
        f"only {len(set(replies))} distinct replies across four different questions"
    )


def test_the_camera_is_actually_saved_from_that_phrasing(app):
    """'remember that too' used to store the note 'too' and drop the camera."""
    say(app, "im john, remember me")
    _reply, result = say(app, "I shoot on Sony alpha6700, remember that too")
    assert "alpha6700" in (result["profile"].get("camera") or "")


# --------------------------------------------------------------------------
# Message classes
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello", "social"),
        ("thanks!", "social"),
        ("are you hallucinating?", "meta"),
        ("are you an AI?", "meta"),
        ("how many looks do you have?", "library"),
        ("list your looks", "library"),
        ("what is CIELAB?", "glossary"),
        ("what is chroma", "glossary"),
        ("my highlights are blown", "glossary"),
        ("can my camera shoot arri log?", "domain_question"),
        ("should I shoot raw or jpeg?", "domain_question"),
        ("tell me a joke", "out_of_scope"),
    ],
)
def test_message_classification(text, expected):
    assert classify_message(text) == expected


def test_glossary_question_is_answered_with_a_citation(app):
    reply, _ = say(app, "what is CIELAB?")
    assert_not_canned(reply, "what is CIELAB?")
    assert "CIELAB" in reply
    assert "wikipedia.org" in reply.lower(), "the answer should cite its source"
    assert "CC BY-SA" in reply, "the licence must be shown"


def test_library_question_lists_the_looks(app):
    reply, _ = say(app, "how many looks do you have?")
    assert "15" in reply
    assert "Warm Golden" in reply and "Teal and Orange" in reply


def test_meta_question_explains_how_it_works(app):
    """'Are you hallucinating?' deserves a real answer, not a profile recital."""
    reply, _ = say(app, "are you hallucinating?")
    assert_not_canned(reply, "are you hallucinating?")
    lowered = reply.lower()
    assert "measured" in lowered
    # It should own its limitations rather than only claiming strengths.
    assert "get wrong" in lowered or "approximate" in lowered


def test_out_of_scope_is_declined_not_answered(app):
    reply, _ = say(app, "tell me a joke")
    assert "outside what I do" in reply
    assert "identify the look" in reply, "a decline should say what it CAN do"


def test_social_messages_get_a_short_reply(app):
    reply, _ = say(app, "thanks!")
    assert len(reply) < 200, "small talk should not trigger a capabilities dump"


def test_domain_question_is_declined_honestly_with_no_model(app):
    """Offline it must say it cannot answer, not invent a camera spec."""
    reply, _ = say(app, "can my camera shoot arri log?")
    assert_not_canned(reply, "can my camera shoot arri log?")
    assert "guessing" in reply.lower() or "cannot" in reply.lower()
    assert "arri log" not in reply.lower(), "it must not attempt a factual answer offline"


def test_capabilities_needs_a_phrase_not_the_word_help(app):
    """'help me understand log' is a question about log, not a request for the menu."""
    reply, _ = say(app, "help me understand log")
    assert "Three things I can do" not in reply


def test_a_message_that_both_tells_and_asks_gets_both(app):
    reply, result = say(app, "i use lightroom, what can you do?")
    assert result["profile"].get("editor") == "lightroom", "the fact should still be saved"
    assert "Noted" in reply, "the save should be acknowledged"
    assert "Three things I can do" in reply, "the question should still be answered"


# --------------------------------------------------------------------------
# Cues that steal ordinary questions
# --------------------------------------------------------------------------

def test_a_routing_cue_does_not_swallow_an_ordinary_question(app):
    """'is anything WRONG with log footage' hits a critique cue with no image."""
    reply, result = say(app, "is anything wrong with log footage")
    assert result["intent"] == "critique"
    assert "Upload the edit" not in reply, (
        "a general question was answered as though it were about the user's image"
    )


def test_asking_about_my_edit_still_asks_for_an_upload(app):
    reply, _ = say(app, "what's wrong with my edit?")
    assert "Upload the edit" in reply


def test_naming_a_look_still_produces_sliders(app):
    _reply, result = say(app, "how do I get the warm golden look?")
    assert result["intent"] == "achieve"
    assert result["recipe"], "the text-only spine must still work"


# --------------------------------------------------------------------------
# Extraction traps found by audit
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "i shoot log",                  # a format, not a camera
        "i shoot raw",
        "i am confused",                # a state, not a name
        "i'm stuck",
        "don't forget what a LUT does",  # a question, not a fact
        "remember what a LUT is",
        "remember that",                # nothing to remember
    ],
)
def test_extraction_rejects_junk(text):
    assert extract_profile_facts(text) == {}


@pytest.mark.parametrize(
    "text,field,value",
    [
        ("I shoot on Sony alpha6700, remember that too", "camera", "sony alpha6700"),
        ("I shoot on a Fuji X-T4, remember", "camera", "fuji x-t4"),
        ("I like warm tones, remember this too", "preferred_look", "warm"),
    ],
)
def test_trailing_remember_captures_what_came_before_it(text, field, value):
    """'..., remember that too' used to store the note 'too'.

    The structured field is what matters. A free-form note is NOT expected
    here: when the sentence is fully captured by a field, keeping it as a note
    as well just makes the recall listing repeat itself.
    """
    facts = extract_profile_facts(text)
    assert facts.get(field) == value
    assert facts.get("_note", "").lower() not in ("too", "that", "this")


def test_a_note_that_adds_nothing_is_not_stored_twice():
    """'im john, remember me' is a name, not a name plus a note saying so."""
    assert extract_profile_facts("im john, remember me") == {"name": "John"}


def test_a_note_that_adds_something_is_kept():
    facts = extract_profile_facts("remember that I print on matte paper")
    assert facts.get("_note") == "I print on matte paper"


# --------------------------------------------------------------------------
# Forgetting
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("delete my profile", ("all", None)),
        ("forget everything", ("all", None)),
        ("clear my memory", ("all", None)),
        ("forget my camera", ("field", "camera")),
        ("forget my name", ("field", "name")),
    ],
)
def test_forget_requests_are_recognised(text, expected):
    assert forget_request(text) == expected


def test_forget_one_field_leaves_the_rest(app):
    say(app, "Hi, I'm Suraj. I shoot on a Fuji X-T4 and I like warm golden looks.")
    reply, result = say(app, "forget my camera")
    assert "forgotten your camera" in reply
    assert "camera" not in result["profile"]
    assert result["profile"].get("name") == "Suraj", "the rest must survive"
    assert "fuji" not in reply.lower(), (
        "the confirmation listed the value it had just deleted"
    )


def test_delete_my_profile_clears_everything_including_the_store(app):
    say(app, "Hi, I'm Suraj. I shoot on a Fuji X-T4.")
    say(app, "remember that I print on matte paper")
    reply, result = say(app, "delete my profile")
    assert "Cleared" in reply
    assert result["profile"] == {}

    # And it is gone from long-term memory, not just this conversation.
    _reply, cold = say(app, "what do you know about me?", thread="a-new-thread")
    assert cold["profile"] == {}


def test_delete_my_profile_is_not_answered_by_listing_it(app):
    """'delete my profile' contains 'my profile', which is also a recall cue."""
    say(app, "Hi, I'm Suraj.")
    reply, _ = say(app, "delete my profile")
    assert reply.startswith("Cleared"), reply[:120]
    assert "Here is everything I have saved" not in reply, (
        "the request to erase the profile was answered by listing it"
    )


def test_forget_a_note_by_phrase(app):
    say(app, "remember that I print on matte paper")
    _reply, result = say(app, "forget that I print on matte paper")
    assert not (result["profile"].get("notes") or [])
