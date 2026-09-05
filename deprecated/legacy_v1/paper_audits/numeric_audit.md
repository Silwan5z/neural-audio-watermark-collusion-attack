# Numeric Audit for v3

Status: **PASS**

## Experimental unit and valid-input rule

- The main risk table was recomputed from `results/evaluation_300clips_20260902/attack_<model>_K{2,3,5}.csv` and the 1,500 final JSON records under `data/k8_population_source_correct_300clips_20260902/raw/`.
- Every model--coalition-size cell contains 300 trials, 100 speakers, 300 distinct source paths, and exactly 100 examples for clip indices 1, 2, and 3.
- `source_exact_count` equals the coalition size in every main-table record. No single-copy decoding failure enters the collusion denominator.
- The exact verified aggregates are stored in `final_paper/analysis/v3_risk_quality_verified.csv`.

## Main claims

- At K=2, the tracing-failure range is 86.333%--99.333%; the manuscript reports 86.3%--99.3%.
- At K=2, mean PESQ spans 4.2036--4.6040 and mean STOI spans 0.9827--0.9992; the manuscript reports 4.20--4.60 and 0.983--0.999.
- At K=8, the five verified tracing-failure rates are AudioSeal 100.0%, WavMark 99.3%, TimbreWM 92.0%, VoiceMark 98.7%, and WMCodec 99.7%.
- The lowest main-experiment means are VoiceMark K=8 PESQ 4.0490 and STOI 0.97094, matching the rounded prose values 4.05 and 0.971.

## Coalition composition

- Figure 2 and Table 3 were recomputed from the final source-correct K=8 JSON records, not from the stale v2 composition export.
- Each system contributes 300 trials from 100 speakers. Figure 2 uses 5,000 speaker-cluster bootstrap resamples.
- Verified Table 3 rows are: AudioSeal 84.9/96.7/20.7, WavMark 95.0/100.0/54.3, TimbreWM 100.0/100.0/100.0, VoiceMark 61.9/74.3/0.7, and WMCodec 62.5/59.7/1.0 percent.
- The figure input is `final_paper/analysis/k8_composition_source_correct_v3.csv`.

## One-bit paths (verified but not used in v3)

- `results/one_bit_k5_300clips_20260902/onebit_k5_{audioseal,voicemark}_full.csv` contains 300 trials per system and 300/300 exact base and flipped endpoints.
- The complete path summaries contain 299 direct and 1 via-another-payload AudioSeal paths, and 113 direct and 187 via-another-payload VoiceMark paths. The VoiceMark percentage is 187/300 = 62.333%, reported as 62.3%.

## Targeted payload tampering

- The source is `results/mrc_pm_native_shared4_fulltop10_300clips_20260904/shards/`.
- Every model--K--method cell contains 3,000 target rows: ten method-selected native-registry targets for each of 300 source-correct trials. Target indicators were summed within a trial before the 300 trial counts were averaged.
- AudioSeal, WavMark, VoiceMark, and WMCodec use identical source paths and coalition payloads for corresponding trial IDs at both K=5 and K=8. TimbreWM uses its own source-correct 10-bit coalitions.
- The K=8 PM/MRC means are 3.57/4.36 for AudioSeal, 3.55/5.44 for WavMark, 6.07/9.23 for TimbreWM, 0.20/0.06 for VoiceMark, and 0.18/0.13 for WMCodec.
- A 10,000-replicate speaker-cluster bootstrap gives K=8 MRC 95% intervals of 4.10--4.62 (AudioSeal), 5.23--5.67 (WavMark), 9.12--9.33 (TimbreWM), 0.04--0.09 (VoiceMark), and 0.09--0.18 (WMCodec). The manuscript reports the three lower bounds and two upper bounds used to establish the system split.
- Across all K=5/K=8 method--system cells, mean PESQ spans 3.9338--4.5825 and mean STOI spans 0.96820--0.99870. The prose explicitly labels these as means.
- All 30,000 stored PM solutions and all 30,000 stored MRC solutions satisfy their method-specific feasibility checks and report successful solver status, including the documented MRC retry path.
- The exact verified aggregates are stored in `final_paper/analysis/v3_targeted_hits_verified.csv`.

## Minimum bit confidence

- Source CSVs are under `results/identity_bit_confidence_k8_20260905/`; the manuscript-facing copies are named `final_paper/analysis/min_bit_confidence_k8_*.csv`.
- Valid-copy and uniform-average conditions each contain one observation for all 300 trials per system. Exact-hit MRC contributes one observation for 294 AudioSeal, 300 WavMark, 300 TimbreWM, 19 VoiceMark, and 34 WMCodec trials.
- The verified valid/uniform/exact-hit-MRC medians are 0.714/0.528/0.609 for AudioSeal, 0.977/0.560/0.729 for WavMark, 0.684/0.505/0.559 for TimbreWM, 0.999/0.710/0.920 for VoiceMark, and 0.995/0.611/0.852 for WMCodec.
- The manuscript interprets the last two exact-hit strata as sparse and makes within-system comparisons only.

## Stale-data quarantine

- Six inherited v2 summaries disagreed with the final source-correct records and are not used by v3. They were preserved under `final_paper/analysis/legacy_v2_not_used/`: `risk_and_quality_summary.csv`, `k8_system_summary.csv`, `k8_source_correct_trials.csv`, `k8_composition_by_count.csv`, `revision_v22_k8_composition.csv`, and `frequency_temporal_summary.csv`.
- No number in `main.tex`, Tables 2--4, or Figures 2--4 is sourced from those quarantined files.
