# Experiment status

Prepared from 51 user-supplied recorded episodes (about 2.61GB, 943 seconds). The full archive and per-file SHA256 checks completed on the authorized training server. Conversion produced 28,271 frames at 30Hz, two RGB wrist cameras at 480×640. No recorder errors or missing required streams were found by the current audit; visual success quality has not been individually labeled.

CPU contract tests passed on the training server, including one real gradient update of a small generated test model and exact equality with the deployed ACTRef network. This test model is never saved as a deployable folding model.

Current observed phase: `complete_offline`. Training finished all 5000 epochs and 25,000 optimizer updates, about 3h18m. Held-out z=0 selection compared 21 checkpoints and selected epoch3499, mean physical left/right joint MAE 0.0733335 rad. The selected checkpoint was replayed with an actual recorded input through the deployment adapter. Exported model size: 336,101,927 bytes; SHA256 is recorded in `artifact_manifest.json`.

The run used the explicitly authorized shared GPU2 without stopping existing jobs. Training and evaluation exited normally and released their CUDA memory; there is no active training process to wait on. Earlier progress reports are historical snapshots, not current status.

Workspace: `~/output/personal/k1deler/fold_towel_20261101`. Status: `reports/pipeline_status.json`. The status file and logs on the server are authoritative; this Markdown is a snapshot, not a live monitor.

New fold-towel task weights have been exported and downloaded locally with a matching SHA256. Model card, all candidate scores, curves and final pipeline status are included. Deployment installation now includes a held-out actual observation replay in addition to generated input smoke checks. No real robot command has been sent; closed-loop folding remains untested. The laboratory machine must independently run `install_bundle.sh` before hardware startup.

Code, configurations, attributed upstream sources and validation reports are kept in `k1deler/fold-towel`. Raw recordings, model files and credentials are excluded from Git; the separate deployment ZIP contains only the selected trained model and one held-out validation observation as deployment assets, plus project code and reports.
