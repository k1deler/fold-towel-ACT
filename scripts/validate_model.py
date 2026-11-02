"""CPU/CUDA deployment check on generated inputs, no real camera or arm."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'inference'),str(ROOT/'training')]
from audit import sha

def fingerprints():
    files=[ROOT/'configs/model.yaml',ROOT/'configs/runtime.yaml',ROOT/'artifact_manifest.json']
    files += list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))
    files += [ROOT/'artifacts/validation_observation.npz']
    files += [p for base in ('inference/yam_infer','inference/third_party') for p in (ROOT/base).rglob('*')
              if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files.append(ROOT/'inference/models/fold_towel_act_delta_seed42.pt')
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    import torch
    marker=ROOT/'reports/model_verified.json'
    environment={'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.executable,
                 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.check:
        report=json.loads(marker.read_text())
        if report['files']!=fingerprints() or report['environment']!=environment:raise ValueError('Stale model validation')
        print('MODEL_GATE_OK');return
    marker.unlink(missing_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required before real robot run')
    from yam_infer.policy.policy import Policy
    from yam_infer.config import load_model_config,check_model_matches,load_run_config
    config=load_model_config(ROOT/'configs/model.yaml')
    if config.task!='fold_towel' or config.action_mode!='delta' or config.format!='act_ref':raise ValueError('Wrong task')
    policy=Policy(config.ckpt,'cuda');check_model_matches(config,policy.format,policy.tasks,policy.action_mode)
    identity=json.loads((ROOT/'reports/registered_model.json').read_text())
    if sha(config.ckpt)!=identity['model_sha256']:raise ValueError('Unregistered model change')
    runtime=load_run_config(ROOT/'configs/runtime.yaml')
    if not 1<=runtime['execution']['horizon']<=policy.chunk_size:raise ValueError('Wrong execution horizon')
    q=policy.start_pose['fold_towel'];frames={c:np.zeros((480,640,3),np.uint8) for c in policy.cameras}
    times=[]
    for _ in range(3):
        t=time.monotonic();actions,_=policy.predict(frames,q,'fold_towel');times.append(time.monotonic()-t)
        if actions.shape!=(100,14) or not np.isfinite(actions).all():raise ValueError('Wrong model output')
    with np.load(ROOT/'artifacts/validation_observation.npz',allow_pickle=False) as observation:
        if list(observation['cameras'])!=list(policy.cameras):raise ValueError('Reference camera order mismatch')
        replay=policy.predict_prepared(observation['images'],observation['qpos'],'fold_towel')
        difference=float(np.max(np.abs(replay-observation['absolute_action'])))
    if not np.isfinite(difference) or difference>.002:
        raise ValueError(f'Real recorded observation replay mismatch: max_abs_error={difference}')
    marker.write_text(json.dumps(dict(files=fingerprints(),environment=environment,scope='generated input plus held-out recorded observation GPU replay',
          recorded_observation_max_error=difference,
          inference_seconds=times,physical_commands_sent=0,task_success_verified=False),indent=2)+'\n')
    print('FOLD_DEPLOY_MODEL_VERIFIED; no robot commands')

if __name__=='__main__':main()
