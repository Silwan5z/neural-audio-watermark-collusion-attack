# Release check

Checked on 2026-09-05 against the files in this repository.

- Manuscript compiles with `/private/users/lym/bin/tectonic -X compile main.tex`.
- Compiled output: 5 US-Letter pages.
- The manuscript has no appendix.
- All four figures and all three external tables resolve from relative paths.
- All 20 uniform-average system/K cells contain 300 source-correct trials.
- All 20 PM/MRC result files contain 300 trials and ten selected targets per trial.
- K=8 records contain 300 distinct source paths and 100 speakers per system.
- The one-bit summary reproduces AudioSeal 299 direct paths and VoiceMark 113 direct / 187 other-payload paths.
- Fig. 4 condition counts match the text, including sparse successful-MRC observations for VoiceMark and WMCodec.
- `python scripts/verify_paper_data.py` passes without model inference.
- No tracked file exceeds GitHub's 100 MB per-file limit.

The historical v1 result tree and scripts are isolated under `deprecated/legacy_v1/` and are not inputs to the current paper.
