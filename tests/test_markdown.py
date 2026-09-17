"""The front end's inline markdown renderer, tested from its own source.

Why this file exists
--------------------
The retrieval tier's whole claim is "check me": an answer arrives with links to
the passages it came from. Shipped, those links rendered as literal
``[Color temperature](https://...)`` bracket soup, because ``md()`` handled
bold, code and underscore-italics but not links at all. A citation a reader
cannot click is decoration.

Adding a link rule to a renderer that interpolates into ``innerHTML`` is exactly
where an XSS hole gets opened, and adding asterisk-italics to a COLOUR SCIENCE
app is exactly where ``L*a*b*`` gets mangled into ``L<em>a</em>b*``. Both risks
are real and both are pinned below.

What this can and cannot check
------------------------------
There is no JavaScript test runner in this project, so these tests read the
regexes out of ``web/index.html`` and run them through Python's ``re``. That
tracks the shipped source -- deleting or weakening a rule fails here -- but it
is a port, not an execution of the page. The behaviour was also verified in a
real browser against the live renderer; these tests are what keeps it true.
"""

from __future__ import annotations

import os
import re

import pytest

PAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "looklab",
    "web",
    "index.html",
)


def page_source() -> str:
    with open(PAGE, encoding="utf-8") as handle:
        return handle.read()


def _js_regex_to_python(pattern: str) -> str:
    """Translate the subset of JS regex syntax this renderer uses."""
    # In JS the forward slash must be escaped inside a literal; Python neither
    # needs nor minds it, but re would read "\/" as an unknown escape.
    return pattern.replace(r"\/", "/")


def extract_rules() -> list[tuple[str, str]]:
    """The ordered (pattern, replacement) pairs md() actually applies.

    Parsed from the page rather than retyped, so a rule that is removed or
    weakened in the real file is removed or weakened here too.
    """
    source = page_source()

    body = re.search(r"function md\(s\)\s*\{(.*?)\n  \}", source, re.DOTALL)
    assert body, "md() not found in index.html"

    link_literal = re.search(r"var MD_LINK = /(.*)/[gimsuy]*;", source)
    assert link_literal, "MD_LINK not found in index.html"

    rules: list[tuple[str, str]] = []
    call = re.compile(
        r"\.replace\(\s*(MD_LINK|/(?P<pat>.*?)/[gimsuy]*)\s*,\s*"
        r"(?P<quote>['\"])(?P<rep>.*?)(?P=quote)\s*\)"
    )
    for match in call.finditer(body.group(1)):
        pattern = link_literal.group(1) if match.group(1) == "MD_LINK" else match.group("pat")
        replacement = re.sub(r"\$(\d)", r"\\\1", match.group("rep"))
        rules.append((_js_regex_to_python(pattern), replacement))

    assert len(rules) >= 4, f"expected md() to apply several rules, parsed {len(rules)}"
    return rules


def esc(text: str) -> str:
    """The page's esc(): ampersand, then angle brackets. Quotes are NOT escaped."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md(text: str) -> str:
    """Apply the real rules, in the real order, to already-escaped text."""
    out = esc(text)
    for pattern, replacement in extract_rules():
        out = re.sub(pattern, replacement, out)
    return out


# --------------------------------------------------------------------------
# The reason the link rule exists
# --------------------------------------------------------------------------

def test_a_citation_becomes_a_clickable_link():
    out = md("[1] [Color temperature](https://en.wikipedia.org/wiki/Color_temperature)")
    assert '<a href="https://en.wikipedia.org/wiki/Color_temperature"' in out
    assert ">Color temperature</a>" in out
    assert "[Color temperature]" not in out, "the raw markdown is still showing"


def test_links_open_safely():
    """target=_blank without rel=noopener hands the opener to the linked page."""
    out = md("[x](https://example.com/a)")
    assert 'rel="noopener noreferrer"' in out
    assert 'target="_blank"' in out


def test_the_glossary_citation_form_also_works():
    out = md("[Source: CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space)")
    assert 'href="https://en.wikipedia.org/wiki/CIELAB_color_space"' in out


# --------------------------------------------------------------------------
# A link rule feeding innerHTML is where XSS gets in
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hostile",
    [
        "[click me](javascript:alert(1))",
        "[click me](JaVaScRiPt:alert(1))",
        "[click me](data:text/html,<script>alert(1)</script>)",
        "[click me](vbscript:msgbox(1))",
        "[click me](/relative/path)",
    ],
)
def test_only_http_urls_become_links(hostile):
    """The scheme is whitelisted in the pattern, not filtered afterwards."""
    out = md(hostile)
    assert "<a " not in out, f"{hostile!r} was turned into a link"
    assert "href" not in out


def test_a_quote_in_a_url_cannot_break_out_of_the_attribute():
    """esc() deliberately does not escape quotes, so the URL class must.

    Without excluding the quote characters, a URL containing one would close
    the href early and everything after it would be parsed as attributes.
    """
    out = md('[x](https://a.com/"onmouseover=alert(1))')
    assert "onmouseover" not in out or "<a " not in out
    assert 'href="https://a.com/"onmouseover' not in out


def test_raw_markup_is_still_inert():
    out = md('<img src=x onerror=alert(1)>')
    assert "<img" not in out
    assert "&lt;img" in out


def test_escaping_happens_before_every_rule():
    """Order is the whole safety argument: rules may only wrap inert text."""
    body = re.search(r"function md\(s\)\s*\{(.*?)\n  \}", page_source(), re.DOTALL)
    assert body
    inner = body.group(1)
    assert inner.index("esc(s)") < inner.index(".replace("), (
        "a replace rule runs before esc(), so it could wrap live markup"
    )


# --------------------------------------------------------------------------
# Asterisk italics in an app where the asterisk is notation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "notation",
    [
        "The CIELAB color space, also referred to as L*a*b*, expresses colour.",
        "Its a* and b* axes are the green-magenta and blue-yellow axes.",
        "L* for perceptual lightness and a* and b* for the four unique colors.",
        "a* is the green-magenta axis and b* is blue-yellow.",
    ],
)
def test_cielab_notation_survives_the_italic_rule(notation):
    """"L*a*b*" must not become "L<em>a</em>b*".

    The rule requires a boundary before the opening marker, and in every one of
    these the asterisk follows a word character, so it cannot fire. This is why
    a naive /\\*([^*]+)\\*/ would be wrong here specifically.
    """
    assert "<em>" not in md(notation)
    assert md(notation) == notation


@pytest.mark.parametrize(
    "text,inner",
    [
        ("defn.\n\n*Why it matters here:* The colour space.", "Why it matters here:"),
        ("body\n\n*Answered by a language model, not measured.*", "Answered by a language model, not measured."),
        ("cite\n*Wikipedia, CC BY-SA 4.0.*", "Wikipedia, CC BY-SA 4.0."),
    ],
)
def test_real_emphasis_still_renders(text, inner):
    """Every one of these is a footer the app actually emits."""
    assert f"<em>{inner}</em>" in md(text)


def test_identifiers_with_underscores_are_left_alone():
    """The trace view names nodes like recipe_build and save_profile."""
    text = "the node recipe_build runs before save_profile."
    assert md(text) == text


def test_bold_and_code_still_work():
    assert "<strong>Teal and Orange</strong>" in md("that is **Teal and Orange** exactly.")
    assert "<code>analyze_pair</code>" in md("call `analyze_pair` first.")
