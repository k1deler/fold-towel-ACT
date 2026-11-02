# Project conventions

- This is the user's fold_towel ACT experiment. Preserve attributed upstream ACT/DETR/YAM/I2RT code and licenses.
- Raw recordings, checkpoints, credentials and virtual environments must never be committed.
- Do not run live robot interfaces remotely as part of development or training checks.
- Use recorded/generated inputs and simulated endpoints for tests. Check train/deploy agreement before changing runtime contracts.
- Task: fold_towel; observations: two RGB wrists plus follower state; labels: recorded leader state; delta for all 14 axes; 30Hz, chunk100.
- Split by complete episodes. Fit statistics and start_pose on training episodes only. Do not call validation MAE a folding success rate.
- Preserve existing successful deployment/weights. New candidates use independent output directories and explicit validation.
- Training may use only an idle GPU; do not stop or change other jobs. The runner has bounded time limits and writes status/logs.
