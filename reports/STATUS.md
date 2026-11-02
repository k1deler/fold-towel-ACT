# Experiment status

Prepared from 51 user-supplied recorded episodes (about 2.61GB, 943 seconds). The full archive and per-file SHA256 checks completed on the authorized training server. Conversion produced 28,271 frames at 30Hz, two RGB wrist cameras at 480×640. No recorder errors or missing required streams were found by the current audit; visual success quality has not been individually labeled.

CPU contract tests passed on the training server, including one real gradient update of a small generated test model and exact equality with the deployed ACTRef network. This test model is never saved as a deployable folding model.

Current observed phase: `waiting_for_idle_gpu`. All four H20 GPUs were busy. The detached pipeline will start its own real-data smoke training and then the configured independent run only when a GPU is idle. It waits at most six hours, then records an incomplete result rather than preempting existing work.

Workspace: `~/output/personal/k1deler/fold_towel_20261101`. Status: `reports/pipeline_status.json`. The status file and logs on the server are authoritative; this Markdown is a snapshot, not a live monitor.

No new full-task weight has been exported yet. No real robot command has been sent. Code, configurations, attributed upstream sources and validation reports have been pushed to `k1deler/fold-towel` through the repository-specific SSH deploy key. The remote main ref was checked against the local commit after upload. Raw recordings, model files and credentials were not committed.
