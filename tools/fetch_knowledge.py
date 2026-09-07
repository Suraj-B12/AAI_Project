"""Build ``looklab/knowledge.json`` -- a cited glossary of colour-science terms.

The app explains itself in terms a photographer may not know: CIELAB, chroma,
hue coherence, split toning, colour temperature. This fetches a short, cited
definition for each so the UI can link out instead of asserting things with no
source.

On sources and copyright, handled deliberately:

* Content comes from the Wikipedia REST summary endpoint, which returns a
  short abstract rather than the article body. Only the first two sentences of
  that abstract are stored, so this is a citation with a snippet, not a copy of
  the article.
* Wikipedia text is CC BY-SA 4.0. Every entry stores its licence, the article
  URL and the revision it came from, and the UI renders the attribution.
* Nothing is scraped from sites whose terms forbid it. The REST API is a
  public, documented interface and a descriptive User-Agent is sent.

Run:  python -m tools.fetch_knowledge
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = (
    "LookLab/1.0 (student colour-grading project; "
    "https://github.com/Suraj-B12/AAI_Project; surajbrajesh@gmail.com)"
)

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "looklab", "knowledge.json"
)

# term -> (wikipedia article, why LookLab mentions it)
TERMS: dict[str, tuple[str, str]] = {
    "CIELAB": (
        "CIELAB_color_space",
        "The colour space every LookLab measurement is taken in. Its a* and b* axes "
        "are the green-magenta and blue-yellow axes, which is why they map so directly "
        "onto Lightroom's Tint and Temp sliders.",
    ),
    "colour grading": (
        "Color_grading",
        "The craft this whole application is about: altering the colour of an image "
        "for a stylistic result rather than a corrective one.",
    ),
    "white balance": (
        "Color_balance",
        "What the Temp and Tint sliders adjust. LookLab measures the residual cast as "
        "global a* and b*.",
    ),
    "colour temperature": (
        "Color_temperature",
        "The warm-cool axis. LookLab uses b* as its proxy, because b* is the "
        "blue-yellow axis of CIELAB.",
    ),
    "chroma": (
        "Colorfulness",
        "How colourful a pixel is, independent of how light it is. LookLab reports "
        "mean chroma and uses it to choose between Vibrance and Saturation.",
    ),
    "hue": (
        "Hue",
        "The angle of a colour around the colour circle. LookLab measures a "
        "chroma-weighted circular mean hue per tonal zone, so grey pixels do not vote.",
    ),
    "circular mean": (
        "Circular_mean",
        "Why LookLab cannot average hues arithmetically: the mean of 359 degrees and "
        "1 degree is 0, not 180.",
    ),
    "split toning": (
        "Cross_processing",
        "Tinting shadows and highlights different colours. LookLab measures the "
        "angular gap between the two as its 'split' feature.",
    ),
    "gamma correction": (
        "Gamma_correction",
        "Why measurements are taken in linear light rather than on the stored sRGB "
        "values: arithmetic on gamma-encoded numbers is not linear-light arithmetic.",
    ),
    "sRGB": (
        "SRGB",
        "The colour space uploaded images are assumed to be in. skimage's rgb2lab "
        "handles the linearisation.",
    ),
    "tone curve": (
        "Curve_(tonality)",
        "What the Contrast slider applies. LookLab measures the result as the L* "
        "spread between the 5th and 95th percentiles.",
    ),
    "clipping": (
        "Clipping_(photography)",
        "Detail lost at pure black or pure white. LookLab reports the fraction of "
        "pixels at L* < 1 and L* > 99 and flags it in critique.",
    ),
    "dynamic range": (
        "Dynamic_range",
        "The span between the darkest and brightest recorded detail.",
    ),
    "colour appearance model": (
        "Color_appearance_model",
        "Why perceptual lightness L* is not the same thing as the V in HSV, and why "
        "zone-splitting on V puts the wrong pixels in the wrong zones.",
    ),
    "curse of dimensionality": (
        "Curse_of_dimensionality",
        "Why LookLab matches on eleven weighted features rather than all twenty-five: "
        "in high dimensions with few samples, distances concentrate and everything "
        "looks equally far away.",
    ),
}


def fetch(article: str) -> dict | None:
    url = WIKI_SUMMARY + urllib.parse.quote(article, safe="")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"    fetch failed: {type(exc).__name__}: {exc}")
        return None


def first_sentences(text: str, count: int = 2) -> str:
    """Keep a snippet, not the article. Abbreviations are handled crudely but safely."""
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return " ".join(parts[:count]).strip()


def main() -> int:
    entries = []
    print(f"Fetching {len(TERMS)} colour-science definitions from Wikipedia\n")
    for term, (article, why) in TERMS.items():
        data = fetch(article)
        if not data or not data.get("extract"):
            print(f"  {term:26s} SKIPPED")
            continue
        entries.append(
            {
                "term": term,
                "summary": first_sentences(data["extract"]),
                "why_it_matters": why,
                "source_title": data.get("title", article),
                "source_url": (data.get("content_urls", {}).get("desktop", {}) or {}).get(
                    "page", f"https://en.wikipedia.org/wiki/{article}"
                ),
                "revision": data.get("revision"),
                "licence": "CC BY-SA 4.0",
                "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            }
        )
        print(f"  {term:26s} OK  ({len(entries[-1]['summary'])} chars)")
        time.sleep(0.25)

    payload = {
        "meta": {
            "source": "English Wikipedia, REST summary endpoint",
            "licence": "CC BY-SA 4.0",
            "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "n_terms": len(entries),
            "note": (
                "Snippets only -- at most the first two sentences of each article's "
                "abstract, stored with the article URL, revision id and licence. "
                "Attribution is rendered in the UI beside every definition."
            ),
        },
        "terms": entries,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(entries)} terms to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
