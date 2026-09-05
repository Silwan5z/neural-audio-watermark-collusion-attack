#!/usr/bin/env python3
"""Extract the original first-draft Fig. 1 artwork without redrawing it."""
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/private/users/lym/drafts/Tracing_Neural_Audio_Watermarks_under_Multi_Copy_Collusion.pdf")
OUT = ROOT / "paper_v4_revision_20260831" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

source = fitz.open(SOURCE)
clip = fitz.Rect(49, 60, 563, 276)
target = fitz.open()
page = target.new_page(width=clip.width, height=clip.height)
page.show_pdf_page(page.rect, source, 1, clip=clip, keep_proportion=False)
target.save(OUT / "fig_scenario.pdf", garbage=4, deflate=True)

page = target[0]
pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
pix.save(OUT / "fig_scenario.png")
(OUT / "fig_scenario.svg").write_text(page.get_svg_image(), encoding="utf-8")
