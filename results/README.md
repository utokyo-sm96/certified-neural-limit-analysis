# Run records

| path | content |
|---|---|
| `runs/heldout-v2/` | one-shot held-out evaluation of the final method (synthetic test plates, real scans): certified histories, FE references on $t_g$ |
| `runs/review2/` | controlled timing of the method, the adaptive FE solver and the cheap classical bounds; verification times; cold start; leave-one-bridge-out |
| `runs/training-logs/`, `runs/learning-curve/`, `runs/fvs-ft30k/` | training logs |
| `runs/` (other folders) | development runs (speed-accuracy sweep, global stage, held-out evaluation of earlier versions) |
| `tuning/` | budgeted tuning loops with frozen evaluators: `results.tsv` lists every run and whether it was kept |
| `probes/` | probe scripts and their outputs behind design decisions (thickness model, mesh floors, solver tests) |

Records that contain measured tensile loads are not included; the measured data are available on request.
