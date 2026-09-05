# Final Artifact Manifest

Workflow: `rewrite_existing`
Tier: `pro`
Status: **PASS**

## Required manuscript artifacts

| Category | Artifact | Purpose |
|---|---|---|
| required | `final_paper/main.tex` | Complete revised LaTeX manuscript |
| required | `final_paper/main.pdf` | Compiled v3 manuscript |
| required | `final_paper/paper.pdf` | Canonical PaperSpine PDF copy |
| required | `final_paper/references.bib` | Bibliography database |
| required | `final_paper/spconf.sty` | ICASSP conference style |
| required | `final_paper/IEEEbib.bst` | IEEE bibliography style |
| required | `latex_report.md` | Compilation and visual-integrity report |
| required | `numeric_audit.md` | Source-to-claim numerical verification |

## Required workflow artifacts

| Category | Artifact group | Purpose |
|---|---|---|
| required | `paper_spine_config.{json,md}` | Frozen workflow and output contract |
| required | `source_map.md`, `reference_materials/source_index.md` | Input provenance |
| required | `research_dossier.md`, `exemplar_learning_dossier.md`, `style_profile.md` | Research and style grounding |
| required | `sota_gap_map.md`, `motivation_options_after_research.md`, `confirmed_motivation.md` | Motivation and gap selection |
| required | `citation_support_bank.md`, `citation_verification_en.md` | Citation support and verification |
| required | `original_logic_map.md`, `section_blueprints.md`, `rewrite_matrix.md` | Rewrite planning |
| required | `writing_rationale_matrix.md`, `logic_transfer_audit.md` | Sentence/section rationale and logic preservation |
| required | `evidence_bank.md`, `results_validation.md`, `confirmed_contribution.md` | Evidence and claim boundary |

## Pro-extra audit artifacts

| Category | Artifact | Purpose |
|---|---|---|
| pro-extra | `structured_review.md` | Structured manuscript review |
| pro-extra | `reviewer_audit.md` | Reviewer-value and objection audit |
| pro-extra | `reviewer_audit_check.md` | Machine-checked reviewer audit |
| pro-extra | `humanize_matrix.md`, `humanize_report.md` | Anti-template and prose naturalness audit |
| pro-extra | `citation_quality_audit.md`, `citation_bank_check.md` | Citation quality and bank validation |
| pro-extra | `contribution_check.md`, `results_validation_check.md` | Contribution and evidence checks |
| pro-extra | `revision_1_literature.md` | Literature-stage audit |
| pro-extra | `revision_audit.md` | v2-to-v3 structural rewrite audit |
| pro-extra | `writing_principles_audit.md` | User-specified clarity, terminology, figure-order, and evidence-boundary audit |

## Figures used by the manuscript

| Category | Artifact | SHA-256 |
|---|---|---|
| required | `final_paper/figures/fig1.pdf` | `6b022478f30cc4a5a3a90ca9b83d87478e7e166f852c625e7f8edefec2c743af` |
| required | `final_paper/figures/fig2_composition_v3.pdf` | `1d87c5415212bead10fe5dd6c50c54cc446a0cb4831c159452c0f47315be7345` |
| required | `final_paper/figures/fig3_paths.pdf` | `8362ef60c9a33b2b9c87d3d50902359dd5d55ca90d1f775eda437db5c33cce3f` |
| required | `final_paper/figures/fig4_confidence.pdf` | `aa368b905023ef90920918df6f6808bbc2503e67b0c42152c17be6dc608f497d` |

The editable Fig. 1 file is `final_paper/figures/fig1.pptx`. PNG/SVG counterparts and the scripts `build_composition_figure_v3.py`, `build_paths_figure.py`, `build_confidence_figure.py`, and `update_fig1_labels.py` are included for inspection and reproducibility.

## Tables and verified analysis

| Category | Artifact | Purpose |
|---|---|---|
| required | `final_paper/tables/table2_risk_quality_v15.tex` | Full risk and quality table |
| required | `final_paper/tables/table3_support_v15.tex` | K=8 bit-support table |
| required | `final_paper/tables/table4_target_redirection.tex` | K=5 and K=8 PM/MRC target-match table |
| pro-extra | `final_paper/analysis/v3_risk_quality_verified.csv` | Verified main aggregates |
| pro-extra | `final_paper/analysis/v3_targeted_hits_verified.csv` | Verified PM/MRC aggregates |
| pro-extra | `final_paper/analysis/k8_composition_source_correct_v3.csv` | Final source-correct composition data |
| pro-extra | `final_paper/analysis/min_bit_confidence_k8_*.csv` | Minimum bit confidence data |
| pro-extra | `final_paper/analysis/legacy_v2_not_used/` | Preserved but quarantined stale v2 summaries |

## Optional outputs

- optional-word: not requested (`word_output=none`).
- optional-translation: not requested (`translation_package=none`).
- optional-submission: not requested.
- optional-review-response: not requested.

## Final delivery checksums

| Category | Artifact | SHA-256 |
|---|---|---|
| required | `/private/users/lym/paper/v3/v3.pdf` | `1cb29b5c40f73ea225db0d929022349f8ac1394453f7de80f194e35fd83f68de` |
| required | `final_paper/main.pdf` | `1cb29b5c40f73ea225db0d929022349f8ac1394453f7de80f194e35fd83f68de` |
| required | `final_paper/main.tex` | `3c6f65e224cf4e9ad611d71e48335eb298d01b8d5b6b14a10ea384a114aaeb31` |

The matching PDF hashes confirm that the root-level delivery is byte-identical to the compiled manuscript.
