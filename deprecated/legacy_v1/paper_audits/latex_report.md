# LaTeX Report

Status: **PASS**

## Compilation

- Engine: Tectonic (XeTeX-compatible pipeline)
- Command: `/private/users/lym/bin/tectonic -X compile main.tex --keep-logs --keep-intermediates`
- Working directory: `final_paper/`
- Output: `final_paper/main.pdf`, synchronized `final_paper/paper.pdf`, and `/private/users/lym/paper/v3/v3.pdf`
- Pages: 5, US Letter (612 x 792 pt)
- Fatal errors: none
- Undefined citations/references: none
- Overfull boxes: none
- Remaining messages are non-blocking font-load, underfull bibliography, and included-PDF version notices; no content is clipped.

## ICASSP layout

- The ICASSP `spconf` style renders the paper title and major headings in the required uppercase form.
- Technical content, discussion, and conclusion end on page 4.
- Page 5 contains references only, as permitted by the ICASSP 2027 paper kit.
- Figure 1 is double-column at the bottom of page 1.
- The manuscript contains no appendix.

## Content and visual integrity

- Seven numbered technical sections serve distinct functions. Discussion and Conclusion are separate; References renders as Section 8.
- Four figures and four tables are present. Each is introduced before its visual position and interpreted afterward.
- Fig. 1 has no check-mark icons, `Watermark detected`, or small `No match` label.
- Fig. 2 uses the full column width and its coordinate region fills nearly the full exported width. System names sit inside the panels; the external left-side title is removed, and the caption directly defines both axes and the dashed Ideal line.
- Fig. 3 uses the full column width and its two vertically separated coordinate regions fill nearly the full exported width. Its axes state `Weight on copy B` and `B-bit support`; the strip labels A, B, and Other.
- Fig. 4 uses nearly the full exported width for the confidence axis. System names sit inside their data rows, and Valid--Uniform--MRC order is retained.
- Plot text is 8.4--9.2 pt at source size, with thicker axis and data strokes. All three analytical figures are embedded as vector PDFs with embedded fonts, so labels remain sharp under PDF zoom.
- Fig. 3 retains AudioSeal and VoiceMark as one bit-based/latent-based contrast. WMCodec is not added because no matching source-correct one-bit path experiment exists, and a third panel would reduce the size of the two evidential plots without establishing a new claim.
- Table 1 uses parallel `Bit-based` and `Latent-based` decoder labels. Table 2 and its supporting prose use `Ideal recombination` and state its interpretive purpose directly.
- Table 4 includes K=5 and K=8; the confidence analysis remains K=8 only.
- The ICASSP template supplies the 9-pt body baseline, paragraph spacing, indentation, and flush-bottom behavior; the manuscript does not override line or paragraph spacing and contains no manual `\vspace`. Float gaps are consistently set to 5--6 pt. Table 2 is a double-column bottom float on page 2. Final-size raster inspection of all five pages found no overlap, truncation, unreadable label, or abnormal technical-page blank block.
