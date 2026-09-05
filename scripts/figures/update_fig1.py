#!/usr/bin/env python3
"""Replace Fig. 1 labels without changing its vector layout."""
from __future__ import annotations

import sys
import re
import zipfile
from pathlib import Path


REPLACEMENTS = {
    b"<a:t>Safe</a:t>": (b"<a:t>Correct</a:t>", 1),
    b"<a:t>Escape</a:t>": (b"<a:t>Coalition</a:t>", 1),
    b"<a:t>Traced to Account A</a:t>": (b"<a:t>Matches Account A</a:t>", 1),
    b"<a:t>Attacker escapes</a:t>": (b"<a:t>No member match</a:t>", 1),
    b"<a:t>Watermark detected</a:t>": (b"<a:t></a:t>", 2),
    b"<a:t>No match</a:t>": (b"<a:t></a:t>", 1),
}

# Two shield outlines and their check marks in the decoded-payload column.
CHECK_ICON_IDS = (92, 93, 96, 97)
CHECK_ICON_PATTERN = re.compile(
    rb'<p:sp><p:nvSpPr><p:cNvPr id="(?:92|93|96|97)".*?</p:sp>',
    re.DOTALL,
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: update_fig1_labels.py INPUT.pptx OUTPUT.pptx")
    source, destination = map(Path, sys.argv[1:])
    destination.parent.mkdir(parents=True, exist_ok=True)

    found = {old: 0 for old in REPLACEMENTS}
    removed_icons = 0
    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "ppt/slides/slide1.xml":
                for old, (new, _) in REPLACEMENTS.items():
                    count = data.count(old)
                    found[old] = count
                    data = data.replace(old, new)
                data, removed_icons = CHECK_ICON_PATTERN.subn(b"", data)
            dst.writestr(item, data)

    missing = [
        old.decode() for old, count in found.items()
        if count != REPLACEMENTS[old][1]
    ]
    if missing:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"unexpected Fig. 1 label counts: {missing}")
    if removed_icons != len(CHECK_ICON_IDS):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"unexpected Fig. 1 check-icon count: {removed_icons}"
        )
    print(destination)


if __name__ == "__main__":
    main()
