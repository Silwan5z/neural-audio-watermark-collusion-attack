# Manuscript

- `main.tex` is the current source.
- `paper.pdf` is the compiled manuscript.
- `figures/` contains the four figures used by `main.tex`; Fig. 1 also includes an editable PPTX.
- `tables/` contains the three external LaTeX tables.
- `audits/` records the numerical, LaTeX, artifact, and writing checks completed for this version.

Build from this directory with:

```bash
tectonic -X compile main.tex
```

The manuscript has no appendix. Figure source data live once under `../data/`, and figure builders live under `../scripts/figures/`.
