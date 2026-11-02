"""Exact same-checkpoint training vs deployed network check; no live hardware."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training/upstream'),str(ROOT/'inference')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    import torch
    from yam_train.policy import ACTPolicy
    from yam_train.data.dataset import FrameStore
    from yam_infer.policy.policy import Policy
    torch.set_num_threads(4)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if c['task']!='fold_towel' or c['action_mode']!='delta' or c['chunk_size']!=100:raise ValueError('Wrong task contract')
    model=ACTPolicy(c['config']['policy'],c['camera_names']);model.load_state_dict(c['model'],strict=True)
    model.cuda().eval().requires_grad_(False)
    policy=Policy(args.checkpoint,'cuda',jpeg_roundtrip=False)
    store=FrameStore(args.data);eps=store.episodes('fold_towel');row=eps[0]['start']
    frames={cam:store.image(cam,row) for cam in c['camera_names']};q=store.qpos[row].astype(np.float64)
    prepared=policy.prepare(frames)
    image=torch.from_numpy(prepared).cuda().permute(0,3,1,2).float()[None]/255
    qn=torch.from_numpy(((q-np.asarray(c['stats']['qpos_mean']))/np.asarray(c['stats']['qpos_std'])).astype(np.float32))[None].cuda()
    with torch.inference_mode():raw=model.predict(qn,image)[0].cpu().numpy()
    expected=raw*np.asarray(c['stats']['action_std'])+np.asarray(c['stats']['action_mean'])+q
    actual=policy.predict_prepared(prepared,q,'fold_towel')
    delta=float(np.max(np.abs(expected-actual)))
    if delta>1e-6 or not np.isfinite(actual).all():raise ValueError('Training/deployment semantics mismatch')
    args.report.write_text(json.dumps(dict(status='passed',max_abs_diff=delta,real_recorded_input=True,
        image_size=c['image_size'],chunk_size=100,task='fold_towel',physical_robot_commands=0,
        smoke_checkpoint_only=bool(c.get('test_only_smoke'))),indent=2)+'\n')
    print('TRAIN_DEPLOY_CONTRACT_OK',delta,flush=True)

if __name__=='__main__':main()
