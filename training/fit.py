"""A recorded, bounded ACT run using attributed yam_train network/data contracts.

Task representation and optimization match the reference implementation. This
entrypoint adds finite checks, disjoint episode bookkeeping and atomic checkpoints.
"""
import argparse,json,time
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training/upstream'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5000);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    import torch,yaml
    from torch.utils.data import DataLoader
    from yam_train.config import load_config
    from yam_train.data.dataset import FrameStore,EpisodicDataset,split_episodes,norm_stats
    from yam_train.policy import ACTPolicy
    from yam_train.train import extra_eval,fixed_samples,plot_curves
    if not torch.cuda.is_available():raise RuntimeError('Full-resolution actual training needs an idle CUDA GPU')
    if args.epochs<1:raise ValueError('epoch count')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/fold_towel.yaml');cfg['data']['root']=str(args.data.resolve())
    cfg['train']['num_epochs']=args.epochs;cfg['train']['batch_size']=2 if args.smoke else 8
    cfg['train']['num_workers']=0 if args.smoke else 4
    torch.set_num_threads(4);torch.manual_seed(cfg['train']['seed']);np.random.seed(cfg['train']['seed'])
    torch.cuda.set_device(0);device=torch.device('cuda')
    store=FrameStore(args.data);eps=store.episodes('fold_towel')
    ti,vi=split_episodes(len(eps),cfg['data']['train_ratio'],cfg['data']['split_seed'])
    tr=[eps[i] for i in ti];val=[eps[i] for i in vi]
    if not tr or not val or set(e['name'] for e in tr)&set(e['name'] for e in val):raise ValueError('episode leakage')
    stats=norm_stats(store,tr,100,0,'delta')
    kw=dict(camera_names=['right_wrist','left_wrist'],norm_stats=stats,chunk_size=100,action_shift=0,action_mode='delta')
    train=EpisodicDataset(store,tr,**kw);valid=EpisodicDataset(store,val,**kw)
    dl=dict(batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['train']['num_workers'],pin_memory=True)
    if dl['num_workers']:dl.update(prefetch_factor=2,persistent_workers=True)
    loader=DataLoader(train,**dl);vl=DataLoader(valid,**dl)
    policy=ACTPolicy(cfg['policy'],kw['camera_names']).cuda()
    optimizer=policy.configure_optimizers()
    info=dict(format='yam_train/act_ref-v1',config=cfg,stats={k:v.tolist() for k,v in stats.items()},
              task='fold_towel',camera_names=kw['camera_names'],image_size=store.meta['image_size'],fps=30,
              chunk_size=100,action_mode='delta',action_shift=0,
              start_pose=np.mean([store.qpos[e['start']] for e in tr],axis=0).tolist(),
              train_episodes=[e['name'] for e in tr],val_episodes=[e['name'] for e in val],
              initialization='new seed42 ACT with standard ImageNet ResNet18, no prior task checkpoint',
              source_provenance=json.loads((ROOT/'training/source.json').read_text()),
              normalization_fitted_on='training episodes only',test_only_smoke=args.smoke)
    (out/'config.yaml').write_text(yaml.safe_dump({**cfg,'train_episodes':info['train_episodes'],'val_episodes':info['val_episodes']},sort_keys=False))
    (out/'normalization.json').write_text(json.dumps(info['stats'],indent=2)+'\n')
    (out/'ckpt').mkdir()
    def save(name,epoch):
        target=out/'ckpt'/name;temporary=target.with_suffix('.tmp')
        payload={**info,'model':{k:v.detach().cpu() for k,v in policy.state_dict().items()},'epoch':epoch}
        torch.save(payload,temporary);temporary.replace(target)
    def batch_loss(batch):
        image,q,a,pad=batch
        return policy(q.to(device),image.to(device).float()/255,a.to(device),pad.to(device))
    started=time.monotonic();updates=0;last=-1
    fixed=fixed_samples(val,8 if args.smoke else 256)
    with (out/'metrics.jsonl').open('x') as metrics:
        def record(row):metrics.write(json.dumps(row,allow_nan=False)+'\n');metrics.flush()
        try:
            for epoch in range(args.epochs):
                policy.eval();vs=[]
                with torch.inference_mode():
                    for i,b in enumerate(vl):
                        loss=batch_loss(b);vs.append({k:float(v) for k,v in loss.items()})
                        if args.smoke:break
                policy.train();ts=[]
                for i,b in enumerate(loader):
                    optimizer.zero_grad(set_to_none=True);loss=batch_loss(b)
                    if not all(torch.isfinite(v).all() for v in loss.values()):raise FloatingPointError('Nonfinite loss')
                    loss['loss'].backward()
                    if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in policy.parameters()):
                        raise FloatingPointError('Nonfinite gradient')
                    optimizer.step();updates+=1;ts.append({k:float(v.detach()) for k,v in loss.items()})
                    if args.smoke:break
                row=dict(kind='epoch',epoch=epoch,updates=updates,time=time.monotonic()-started,
                         train={k:float(np.mean([x[k] for x in ts])) for k in ts[0]},
                         val={k:float(np.mean([x[k] for x in vs])) for k in vs[0]})
                record(row);last=epoch
                if epoch%10==0 or epoch==args.epochs-1:
                    print('EPOCH',epoch,'updates',updates,'train',row['train']['loss'],'val',row['val']['loss'],
                          'seconds',row['time'],flush=True)
                if epoch%250==0 or epoch==args.epochs-1:
                    result=extra_eval(policy,valid,fixed,device,cfg['train']['batch_size'])
                    record(dict(kind='extra',epoch=epoch,val=result,train=None));print('Z0_VALIDATION',json.dumps(result),flush=True)
                if (epoch+1)%250==0:
                    save(f'epoch_{epoch:05d}.pt',epoch);save('last.pt',epoch);plot_curves(out)
            save('last.pt',last);plot_curves(out)
        except BaseException:
            if last>=0:save('interrupted.pt',last)
            raise
    print('TRAINING_FINISHED',json.dumps(dict(epochs=args.epochs,updates=updates,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
