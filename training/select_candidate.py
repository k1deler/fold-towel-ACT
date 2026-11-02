"""Physical held-out z=0 selection and checkpoint compatibility/export checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training/upstream'),str(ROOT/'inference')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import torch
    from yam_train.data.dataset import FrameStore,EpisodicDataset
    from yam_train.policy import ACTPolicy
    from yam_train.train import fixed_samples,extra_eval
    from yam_infer.policy.policy import Policy
    out=args.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    store=FrameStore(args.data);paths=sorted((args.run/'ckpt').glob('epoch_*.pt'))+list((args.run/'ckpt').glob('last.pt'))
    rows=[];best=None;best_score=None
    for path in paths:
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c.get('test_only_smoke'):raise ValueError('Cannot deploy smoke weights')
        if set(c['train_episodes'])&set(c['val_episodes']):raise ValueError('Episode split leakage')
        val=[e for e in store.episodes('fold_towel') if e['name'] in c['val_episodes']]
        stats={k:np.asarray(v,np.float32) for k,v in c['stats'].items()}
        ds=EpisodicDataset(store,val,c['camera_names'],stats,100,0,'delta')
        model=ACTPolicy(c['config']['policy'],c['camera_names']);model.load_state_dict(c['model'],strict=True)
        model.cuda().eval().requires_grad_(False)
        met=extra_eval(model,ds,fixed_samples(val,512),torch.device('cuda'),8)
        if not np.isfinite(list(met.values())).all():raise ValueError('Nonfinite validation')
        # Fold towel can use both arms: neither arm is ignored in selection.
        score=(met['right_arm_rad']+met['left_arm_rad'])/2
        rows.append(dict(checkpoint=path.name,epoch=c['epoch'],joint_mae=score,metrics=met))
        (out/'selection.json').write_text(json.dumps(rows,indent=2)+'\n')
        if best_score is None or score<best_score:best_score,best=score,path
        del model,c;torch.cuda.empty_cache()
    if best is None:raise ValueError('No trained candidates')
    chosen=torch.load(best,map_location='cpu',weights_only=False)
    chosen['selection']={'method':'fixed 512 held-out windows; mean physical left/right joint MAE; z=0',
                         'checkpoint':best.name,'joint_mae':best_score,'robot_tested':False}
    exported=out/'fold_towel_act_delta_seed42.pt';torch.save(chosen,exported)
    policy=Policy(exported,'cuda');eps=store.episodes('fold_towel');ep=next(e for e in eps if e['name'] in chosen['val_episodes'])
    row=ep['start'];frames={c:store.image(c,row) for c in chosen['camera_names']}
    value,_=policy.predict(frames,store.qpos[row],'fold_towel')
    if value.shape!=(100,14) or not np.isfinite(value).all():raise ValueError('Deployment replay failed')
    (out/'model_card.json').write_text(json.dumps(dict(task='fold_towel',format=chosen['format'],
          source=chosen['source_provenance'],initialization=chosen['initialization'],selected_from=best.name,
          epoch=chosen['epoch'],joint_mae=best_score,real_recorded_input_replay_passed=True,
          deployment_adapter='yam_infer ACTRef Policy',robot_tested=False,
          caution='Offline selection is not folding success; full recorded/closed-loop testing remains required.'),indent=2)+'\n')
    print('SELECTED_MODEL',exported,flush=True)

if __name__=='__main__':main()
