"""Copy an actually trained fold_towel model into this deployment, offline only."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'inference'),str(ROOT/'training')]
from audit import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);args=p.parse_args()
    import torch
    source=args.checkpoint.resolve()
    c=torch.load(source,map_location='cpu',weights_only=False)
    if c.get('test_only_smoke') or c.get('reproduction_test_only'):raise ValueError('Cannot deploy smoke/test weights')
    if (c.get('format'),c.get('task'),c.get('action_mode'),c.get('chunk_size'))!=(
            'yam_train/act_ref-v1','fold_towel','delta',100):raise ValueError('Wrong training/deployment task contract')
    if len(c['start_pose'])!=14 or c['image_size']!=[480,640]:raise ValueError('Wrong model metadata')
    target=ROOT/'inference/models/fold_towel_act_delta_seed42.pt'
    target.parent.mkdir(exist_ok=True)
    if target.exists():raise FileExistsError('Refusing overwrite of an existing deployment model')
    shutil.copy2(source,target)
    if sha(source)!=sha(target):raise RuntimeError('Model copy hash mismatch')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/registered_model.json').write_text(json.dumps(dict(task='fold_towel',model_sha256=sha(target),
             source_file=source.name,robot_tested=False),indent=2)+'\n')
    print('FOLD_MODEL_REGISTERED; no hardware connected')

if __name__=='__main__':main()
