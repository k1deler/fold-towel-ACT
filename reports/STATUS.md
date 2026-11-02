# Experiment status

Prepared from 51 user-supplied recorded episodes (about 2.61GB, 943 seconds). The full archive and per-file SHA256 checks completed on the authorized training server. Conversion produced 28,271 frames at 30Hz, two RGB wrist cameras at 480×640. No recorder errors or missing required streams were found by the current audit; visual success quality has not been individually labeled.

CPU contract tests passed on the training server, including one real gradient update of a small generated test model and exact equality with the deployed ACTRef network. This test model is never saved as a deployable folding model.

Current observed phase: `training`. On 2026-11-02 the user explicitly authorized shared compute on GPU2. The pipeline prefers idle GPUs but also permits the authorized shared card when at least 32GiB of memory is free. No existing job was stopped. Real recorded-data smoke training finished, and exact train/deploy network agreement on recorded input passed with max_abs_diff=0.0. Formal training is active on GPU2: seed42, batch8, chunk100, 5000 epochs. Snapshot: epoch72 completed, 365 optimizer updates. See `training_started_snapshot.json` for the measured status and approximate remaining duration.

The active training process is not converted to dynamic multi-GPU training. If another GPU is idle when final held-out selection runs, the pipeline can use it for evaluation. Shared compute speed and the remaining-time estimate can change.

Workspace: `~/output/personal/k1deler/fold_towel_20261101`. Status: `reports/pipeline_status.json`. The status file and logs on the server are authoritative; this Markdown is a snapshot, not a live monitor.

No new full-task weight has been exported yet. No real robot command has been sent. Code, configurations, attributed upstream sources and validation reports have been pushed to `k1deler/fold-towel` through the repository-specific SSH deploy key. The remote main ref was checked against the local commit after upload. Raw recordings, model files and credentials were not committed.
