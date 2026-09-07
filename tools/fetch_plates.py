"""Fetch freely-licensed real photographs from Wikimedia Commons as base plates.

The design document's Part 6 wants real photographs. The scikit-image samples
that ship with the package are real photographs and work, but there are only a
handful and they were not chosen for this task. This script widens the pool
with properly-licensed images from Wikimedia Commons.

Licensing is handled explicitly, not assumed:

* Only images whose ``extmetadata`` licence is public domain, CC0, CC BY or
  CC BY-SA are accepted. Anything else is skipped.
* Every accepted image's licence, author and source URL is written to
  ``looklab/data/plates/MANIFEST.json`` and rendered into ``ATTRIBUTION.md``,
  so the attribution CC BY requires actually exists.
* The Wikimedia API policy requires a descriptive User-Agent. One is sent.

Candidates are then filtered on fitness for the task, not on looks: a usable
base plate needs pixels in every L* zone, or its zone statistics are
meaningless. That check is the same ``MIN_ZONE_PIXELS`` rule the colour engine
applies at measurement time.

Run:  python -m tools.fetch_plates            (fetch, filter, write manifest)
      python -m tools.fetch_plates --dry-run  (report candidates, download none)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looklab.color import ZONES, signature_from_array  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "LookLab/1.0 (student colour-grading project; "
    "https://github.com/Suraj-B12/AAI_Project; surajbrajesh@gmail.com)"
)

PLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "looklab", "data", "plates"
)

# Licences we will actually use. Anything else is skipped rather than guessed at.
ACCEPTED_LICENCE = re.compile(
    r"(public domain|^pd|cc0|cc[- ]by(?![- ]nc)|attribution)", re.IGNORECASE
)
REJECTED_LICENCE = re.compile(r"(non[- ]?commercial|[- ]nc[- ]|no[- ]deriv|nd\b|fair use)", re.I)

# Search terms chosen for tonal and chromatic variety, not for prettiness --
# each targets a different region of the signature space.
SEARCHES: list[tuple[str, str]] = [
    ("portrait_daylight", "portrait photograph daylight face"),
    ("landscape_sky", "landscape photograph sky clouds"),
    ("interior_lowkey", "interior room window light photograph"),
    ("food_saturated", "food photograph colourful dish"),
    ("street_neutral", "street photography city building"),
    ("nature_green", "forest trees green photograph"),
    ("golden_hour", "sunset golden hour photograph"),
    ("snow_highkey", "snow winter landscape photograph"),
]

MIN_ZONE_FRACTION = 0.02  # each L* zone must hold at least 2% of pixels
MAX_SIDE = 512


def _get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    return json.loads(_get(url).decode("utf-8"))


def _plain(html: str | None) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_candidates(term: str, limit: int = 14) -> list[dict]:
    """Search Commons for images and return those with an acceptable licence."""
    try:
        data = _api(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {term}",
                "gsrnamespace": "6",  # File:
                "gsrlimit": str(limit),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mime",
                "iiurlwidth": str(MAX_SIDE),
            }
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"    search failed for {term!r}: {type(exc).__name__}: {exc}")
        return []

    out: list[dict] = []
    for page in (data.get("query", {}) or {}).get("pages", []) or []:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        meta = info.get("extmetadata", {}) or {}
        licence = _plain(meta.get("LicenseShortName", {}).get("value"))
        if not licence or REJECTED_LICENCE.search(licence) or not ACCEPTED_LICENCE.search(licence):
            continue
        out.append(
            {
                "title": page.get("title", ""),
                "url": info.get("thumburl") or info.get("url"),
                "descriptionurl": info.get("descriptionurl", ""),
                "licence": licence,
                "licence_url": _plain(meta.get("LicenseUrl", {}).get("value")),
                "author": _plain(meta.get("Artist", {}).get("value"))[:160] or "unknown",
                "credit": _plain(meta.get("Credit", {}).get("value"))[:160],
            }
        )
    return out


def evaluate(raw: bytes) -> tuple[np.ndarray | None, dict]:
    """Decode, downscale and check the image is usable as a base plate."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        return None, {"reason": f"decode failed: {type(exc).__name__}"}
    if min(img.size) < 200:
        return None, {"reason": f"too small {img.size}"}
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0

    sig = signature_from_array(arr)
    coverage = sig.get("zone_coverage", {})
    thin = [z for z in ZONES if float(coverage.get(z, 0.0)) < MIN_ZONE_FRACTION]
    stats = {
        "coverage": {k: round(float(v), 4) for k, v in coverage.items()},
        "contrast": round(sig["contrast"], 1),
        "chroma": round(sig["chroma"], 1),
        "L_p50": round(sig["L_p50"], 1),
        "a_global": round(sig["a_global"], 2),
        "b_global": round(sig["b_global"], 2),
    }
    if thin:
        return None, {"reason": f"thin L* zones: {thin}", **stats}
    return arr, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report candidates, download none")
    parser.add_argument("--per-term", type=int, default=1, help="plates to keep per search term")
    args = parser.parse_args()

    os.makedirs(PLATE_DIR, exist_ok=True)
    manifest: list[dict] = []

    for slug, term in SEARCHES:
        print(f"\n[{slug}] searching Commons for {term!r}")
        candidates = search_candidates(term)
        print(f"  {len(candidates)} candidates with an acceptable licence")
        kept = 0
        for cand in candidates:
            if kept >= args.per_term:
                break
            if not cand["url"]:
                continue
            try:
                raw = _get(cand["url"])
            except Exception as exc:
                print(f"    skip (download {type(exc).__name__}) {cand['title'][:50]}")
                continue
            arr, stats = evaluate(raw)
            if arr is None:
                print(f"    skip ({stats['reason']}) {cand['title'][:50]}")
                continue

            name = f"{slug}_{kept}"
            entry = {
                "name": name,
                "title": cand["title"],
                "source": cand["descriptionurl"],
                "licence": cand["licence"],
                "licence_url": cand["licence_url"],
                "author": cand["author"],
                "credit": cand["credit"],
                "stats": stats,
            }
            if not args.dry_run:
                from PIL import Image

                out_path = os.path.join(PLATE_DIR, f"{name}.jpg")
                Image.fromarray((arr * 255).astype("uint8")).save(
                    out_path, format="JPEG", quality=94
                )
                entry["file"] = f"{name}.jpg"
            manifest.append(entry)
            kept += 1
            print(
                f"    KEEP {name}: {cand['licence'][:28]:28s} "
                f"cov={stats['coverage']} con={stats['contrast']} chroma={stats['chroma']}"
            )
            time.sleep(0.4)  # be polite to the API

    if args.dry_run:
        print(f"\ndry run: {len(manifest)} plates would be kept")
        return 0

    with open(os.path.join(PLATE_DIR, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": "Wikimedia Commons", "plates": manifest}, fh, indent=2)

    lines = [
        "# Base plate attribution",
        "",
        "Every image below was downloaded from Wikimedia Commons under the licence",
        "stated in its row, and is used here only as a neutral base plate for colour",
        "measurement. Downscaled to at most "
        f"{MAX_SIDE}px on the long edge; no other alteration.",
        "",
        "| Plate | Source | Author | Licence |",
        "|---|---|---|---|",
    ]
    for entry in manifest:
        licence = (
            f"[{entry['licence']}]({entry['licence_url']})"
            if entry["licence_url"]
            else entry["licence"]
        )
        lines.append(
            f"| `{entry['name']}` | [{entry['title']}]({entry['source']}) | "
            f"{entry['author']} | {licence} |"
        )
    with open(os.path.join(PLATE_DIR, "ATTRIBUTION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nwrote {len(manifest)} plates to {PLATE_DIR}")
    print("  MANIFEST.json and ATTRIBUTION.md written alongside them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
