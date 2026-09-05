# Writing-Principles Audit for v3

Status: **PASS**

## Direct user requirements

| Requirement | Verification | Status |
|---|---|---|
| Fig. 1 cleanup | The editable PPTX and exported PDF contain no check-mark icons, no `Watermark detected`, and no small `No match`. The lower branch is labeled `Coalition`. | PASS |
| Escape versus TF | A single mixture *escapes tracing* when its decoded payload matches no coalition member. *Tracing failure (TF)* is the percentage of trials that escape. TF is explicitly identified as an outcome rate, not an attack method. | PASS |
| Table 1 grouping | The compact table uses parallel, reviewer-facing labels: three `Bit-based` decoders recover payload bits directly, while two `Latent-based` decoders recover latent codes that map to payload bits. | PASS |
| Section 3 focus | Section 3 retains the five-system TF/quality result, the K=8 bit-recombination explanation, and one compact one-bit diagnostic that explains why shared input bits can change in VoiceMark. The unrelated region scan remains removed. | PASS |
| Figure order and labels | Fig. 2 removes the external left-side title and defines both quantities directly in its caption. Fig. 3 uses two vertically separated one-bit mixture-path panels at full column width; `Weight on copy B` and `B-bit support` are printed directly on the figure. In confidence Fig. 4, both marks and legend run Valid, Uniform, MRC from top to bottom. Decimal labels use a leading zero. | PASS |
| Table 4 scope | Table 4 keeps K=5 and K=8 as a compact repeatability check. The confidence analysis remains K=8 only. | PASS |
| References | The manuscript cites 20 references. PM is linked directly to StegaStamp and Distortion Agnostic Deep Watermarking, both CVPR 2020 papers. | PASS |
| No appendix | The manuscript contains no appendix or supplemental section. | PASS |

## One center and cognitive order

The paper makes one central claim: native message recovery does not guarantee recipient tracing after valid copies are averaged. The evidence follows one forward chain:

1. Section 2 defines the attack input, tracing escape, and TF.
2. Section 3 establishes the scale of TF while controlling speech quality, shows how coalition-supported bits recombine, and uses one bit-based/latent-based comparison to expose changes at shared input bits.
3. Section 4 tests the stronger and distinct outcome of exact matches to ten method-selected nonmembers.
4. Section 5 compares the minimum bit confidence of Valid, Uniform, and successful MRC outputs.
5. Section 6 discusses the evaluation and design implications without adding a new result.
6. Section 7 closes on the evidence-backed change required in evaluation and introduces no new claim.

The method description is shorter than the main evidence. The one-bit experiment is retained only in the form needed to explain the second decoder response pattern; the region scan remains excluded.

## Fixed terminology

| Term | Fixed meaning |
|---|---|
| recipient | Person assigned a personalized copy |
| coalition member | Recipient whose copy contributes to the average |
| payload | Complete embedded and decoded bit string |
| tracing escape | A mixture whose decoded payload matches no coalition member |
| tracing failure (TF) | Percentage of trials that escape tracing |
| targeted payload tampering | Exact decoded match to a selected nonmember |
| minimum bit confidence | Lowest support among the bits of the decoded payload |

The manuscript does not use `escape` as a metric or TF as a procedure. It does not alternate targeted payload tampering with redirection, steering, or targeting.

## Figures and tables

- Fig. 1 has one task: contrast correct tracing from one copy with tracing escape after coalition averaging.
- Fig. 2 gives nearly the full exported width to the coordinate region. System names sit inside the panels, the external left-side title is removed, and the caption states the horizontal and vertical quantities and names the dashed line as Ideal.
- Fig. 3 gives nearly the full exported width to two enlarged, vertically separated coordinate regions. Its short axis labels state the mixture weight and decoder support directly; the decoded-payload strip distinguishes A, B, and Other.
- Fig. 4 gives nearly the full exported width to the confidence axis. System names sit inside their data rows rather than consuming a separate label margin; the legend retains Valid--Uniform--MRC order.
- Analytical-figure text is 8.4--9.2 pt, axes and marks use publication-weight strokes, and embedded vector fonts keep English labels sharp at final PDF size and under zoom.
- Fig. 3 intentionally uses AudioSeal and VoiceMark as a single bit-based/latent-based contrast. WMCodec is not added without a matching source-correct path experiment; doing so would shrink the existing evidence without supporting an additional conclusion.
- Tables use horizontal IEEE rules without vertical lines or leaderboard decoration.
- Every figure and table is defined before its visual position and interpreted immediately afterward.
- The ICASSP template's default flush-bottom typesetting aligns both column bottoms; body line spacing and paragraph spacing are not overridden. Float gaps are set consistently, without manual `\vspace`. Final-size raster inspection found no overlap, clipping, unreadable label, or abnormal blank block. Table 2 is at the bottom of page 2 and remains ahead of Tables 3--4 in visual order.

## Evidence boundaries

- Bit-support patterns are evidence about decoder responses, not proof of an internal representation.
- Targeted tampering is limited to ten method-selected nonmembers; the paper does not claim arbitrary-recipient impersonation.
- VoiceMark and WMCodec's low target counts are reported as results rather than explained away.
- Fig. 4 is conditional and descriptive because MRC already favors stronger payload support.
- Confidence-based rejection remains a design direction with no universal threshold or validated-defense claim.

## ICASSP and scan checks

- The title and major headings render in uppercase because the ICASSP template requires that style; the source titles are concise and descriptive.
- Technical content ends on page 4. Page 5 contains references only.
- A 30-second scan of the title, abstract, headings, Fig. 1, captions, and conclusion yields the same message: coalition averaging preserves audio and watermark evidence but breaks recipient tracing.
- The prose contains none of the targeted mechanical connectors or promotional self-evaluations, and it reports representative contrasts rather than reading tables cell by cell.
