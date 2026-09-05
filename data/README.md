# Paper data

Only data used by the current manuscript are tracked here.

| Directory | Contents | Expected size |
|---|---|---:|
| `main/` | Uniform-average trial records for K=2, 3, and 5 | 15 files × 300 trials |
| `k8/raw/` | Source-correct K=8 trial records | 5 systems × 300 trials |
| `coalitions/` | Shared 16-bit coalitions and cached PM/MRC selections | 300 trials at K=5 and K=8 |
| `targeted/` | Merged PM/MRC target attempts | 20 files × 3,000 attempts |
| `one_bit/` | Valid endpoint pairs and a 600-trial path summary | 300 trials per system |
| `confidence/` | Per-trial minimum-bit-confidence records used by Fig. 4 | 5 files |
| `summary/` | Compact table and figure inputs | small CSV files |

The four 16-bit systems use the same K=5 and K=8 coalitions. TimbreWM uses separately validated coalitions because its payload has 10 bits. Absolute source paths are retained as provenance; audio and marked-copy caches are not distributed.

Run `python scripts/verify_release.py` from the repository root to check counts, schemas, key manuscript aggregates, and `MANIFEST.csv` checksums.
