"""Read the verdicts marked by a human in REVIEW.md and compute manifest-noise stats.

Companion to ``audit_manifest.py``. After the human marks each card with
``M`` / ``W`` / ``A``, this script extracts the verdicts, joins them against
the bucket info from ``candidates.json``, and prints:

- Manifest-error rate among the *confident-disagree* bucket (high signal —
  these are the ones we expect to be manifest errors).
- Overall manifest-error estimate (extrapolated from random-disagree bucket).
- Sanity check on the random-agree bucket (should be ~0% manifest errors).
- Adjusted v3_v2 set_code accuracy under the assumption manifest errors are
  bug-for-bug consistent across the test set.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Match a section header like:
#   ### 080d9565 · `confident-disagree` · confidence=0.987
_HEADER_RE = re.compile(
    r"^###\s+(?P<sid>[0-9a-f]+)\s+·\s+`(?P<bucket>[a-z\-]+)`\s+·\s+confidence=(?P<conf>[\d.]+)"
)
# Match `**Verdict**: `M`` or `**Verdict**: \`_\`` — extract the single character.
_VERDICT_RE = re.compile(r"\*\*Verdict\*\*:\s*`([MWA_])`")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to candidates.json (defaults to sibling of REVIEW.md)",
    )
    args = parser.parse_args(argv)

    cand_path = args.candidates or args.review.parent / "candidates.json"
    candidates = json.loads(cand_path.read_text())["candidates"]
    bucket_by_sid = {c["scryfall_id"][:8]: c["bucket"] for c in candidates}

    # Parse REVIEW.md sequentially, pairing each header with the next Verdict.
    text = args.review.read_text(encoding="utf-8")
    sections = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    verdicts: list[tuple[str, str, str]] = []  # (sid_short, bucket, letter)
    for section in sections:
        h = _HEADER_RE.search(section)
        v = _VERDICT_RE.search(section)
        if not h or not v:
            continue
        sid = h.group("sid")
        bucket = h.group("bucket")
        letter = v.group(1)
        verdicts.append((sid, bucket, letter))

    if not verdicts:
        print("no parseable sections found in", args.review)
        return 1

    # Aggregate by bucket.
    by_bucket: dict[str, Counter[str]] = {}
    for _sid, bucket, letter in verdicts:
        by_bucket.setdefault(bucket, Counter())[letter] += 1

    print(f"# Manifest audit results — {len(verdicts)} cards reviewed\n")
    for bucket in ("confident-disagree", "random-disagree", "random-agree"):
        c = by_bucket.get(bucket, Counter())
        total = c["M"] + c["W"] + c["A"] + c["_"]
        if total == 0:
            print(f"## {bucket}: no entries")
            continue
        marked = c["M"] + c["W"] + c["A"]
        print(f"## {bucket}: {marked}/{total} marked")
        print(f"  M (manifest wrong) : {c['M']:>3}  ({c['M'] / max(marked, 1) * 100:.1f}% of marked)")
        print(f"  W (model wrong)    : {c['W']:>3}  ({c['W'] / max(marked, 1) * 100:.1f}% of marked)")
        print(f"  A (ambiguous)      : {c['A']:>3}  ({c['A'] / max(marked, 1) * 100:.1f}% of marked)")
        if c["_"]:
            print(f"  _ (unmarked)       : {c['_']:>3}")
        print()

    # Extrapolate: among ALL test-set disagreements, what fraction is manifest noise?
    # If we sampled both buckets representatively, weighted average gives the estimate.
    confident = by_bucket.get("confident-disagree", Counter())
    random_dis = by_bucket.get("random-disagree", Counter())
    cm = confident["M"] + random_dis["M"]
    cw = confident["W"] + random_dis["W"]
    ca = confident["A"] + random_dis["A"]
    cmarked = cm + cw + ca
    if cmarked > 0:
        print("## Combined disagreement signal")
        print(f"  total marked disagreements: {cmarked}")
        print(f"  manifest-error rate among disagreements (combined): {cm / cmarked * 100:.1f}%")
        if random_dis:
            rand_marked = random_dis["M"] + random_dis["W"] + random_dis["A"]
            if rand_marked:
                rand_rate = random_dis["M"] / rand_marked
                print(f"  manifest-error rate among RANDOM disagreements (less biased): {rand_rate * 100:.1f}%")
                print()
                print("## Adjusted set_code accuracy (estimate)")
                print(f"  v3_v2 reported set_code_accuracy: 23.34%")
                # raw eval was: 246/1054 correct, 808 wrong.
                # If `rand_rate` of those 808 are actually manifest errors (model right),
                # adjusted hits = 246 + 808 * rand_rate, denom unchanged.
                wrong = 808
                adjusted = (246 + wrong * rand_rate) / 1054
                print(f"  estimated 'true' set_code_accuracy: {adjusted * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
