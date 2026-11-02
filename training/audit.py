"""Validate the actual fold-towel recordings and converted training contract."""
import hashlib,json
from pathlib import Path
import numpy as np

TASK='fold_towel'
CAMS=('right_wrist','left_wrist')
TOPICS=[f'nmx/hal/i2rt/{role}/{side}/{kind}' for role in ('leader','follower')
        for side in ('left','right') for kind in ('joint','hand_state')]
TOPICS += [f'nmx/hal/camera/camera_{i}/rgbd' for i in (0,1)]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def raw_audit(root):
    root=Path(root);accepted=[];rejected=[]
    for ep in sorted(root.glob('episode_*')):
        if not ep.is_dir():continue
        try:
            m=json.loads((ep/'metadata.json').read_text());s=json.loads((ep/'episode_stats.json').read_text())
            if ep.name.endswith('_inprogress') or s.get('writer_error') or s.get('dropped_batches',0):
                raise ValueError('incomplete/error/drop in recorder statistics')
            if m.get('collection_mode')!='teleoperation':raise ValueError('not a human demonstration')
            extra=m.get('extra',{})
            if extra.get('state_layout')!='left_joint0..5,left_gripper,right_joint0..5,right_gripper':
                raise ValueError('unknown state order')
            if extra.get('camera_color_space',{}).get('jpeg')!='rgb':raise ValueError('unknown recorded color contract')
            if any(s.get('topic_counts',{}).get(t,0)<2 for t in TOPICS):raise ValueError('required stream missing')
            if not list((ep/'data').glob('*.mcap')):raise ValueError('no MCAP shard')
            if s.get('duration_s',0)<4:raise ValueError('too short for full fold')
            accepted.append(ep.name)
        except (OSError,ValueError,KeyError) as e:rejected.append({'episode':ep.name,'reason':str(e)})
    if len(accepted)<5:raise ValueError('insufficient complete demonstrated episodes')
    return {'task':TASK,'accepted':accepted,'rejected':rejected,
            'metadata_name_note':'Recorder task name may be a generic label; user identifies these recordings as fold_towel.'}

def converted_audit(root):
    import cv2
    root=Path(root);m=json.loads((root/'meta.json').read_text())
    if m.get('format')!='yam_train/jpeg-v1' or m.get('fps')!=30 or m.get('image_size')!=[480,640]:
        raise ValueError('wrong converted dimensions/frequency/format')
    if m.get('qpos')!='follower' or m.get('action')!='leader':raise ValueError('wrong target source')
    eps=sorted([e for e in m['episodes'] if e['task']==TASK],key=lambda e:e['name'])
    if len(eps)<5 or len(set(e['name'] for e in eps))!=len(eps):raise ValueError('invalid episode identities')
    q=np.load(root/'qpos.npy',mmap_mode='r');a=np.load(root/'action.npy',mmap_mode='r')
    if q.shape!=a.shape or q.ndim!=2 or q.shape[1]!=14 or not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError('invalid physical vectors')
    if any(np.any(v[:,[6,13]]<0) or np.any(v[:,[6,13]]>1) for v in (q,a)):raise ValueError('invalid recorded gripper')
    for cam in CAMS:
        idx=np.load(root/(cam+'.idx.npy'));size=(root/(cam+'.jpg.bin')).stat().st_size
        if idx.shape!=(len(q),2) or np.any(idx<0) or np.any(idx[:,1]<=0) or np.any(idx.sum(1)>size):
            raise ValueError('invalid camera index')
        with (root/(cam+'.jpg.bin')).open('rb') as f:
            for ep in eps:
                off,n=idx[ep['start']];f.seek(int(off));im=cv2.imdecode(np.frombuffer(f.read(int(n)),np.uint8),1)
                if im is None or im.shape!=(480,640,3):raise ValueError('decoded image mismatch')
    for ep in eps:
        if ep['length']<100:raise ValueError('episode shorter than ACT horizon')
        if ep['max_cam_gap_ms']>100:raise ValueError('excess camera gap in '+ep['name'])
    names=['meta.json','qpos.npy','action.npy']+[c+s for c in CAMS for s in ('.idx.npy','.jpg.bin')]
    return {'task':TASK,'episodes':len(eps),'frames':len(q),'dataset_sha256':{n:sha(root/n) for n in names},
            'image_size':[480,640],'fps':30,'label':'leader(t+k)-follower(t), ALL 14 dims including grippers'}
