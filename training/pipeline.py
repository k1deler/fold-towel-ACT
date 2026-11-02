"""Data verification -> conversion -> idle-GPU wait -> smoke -> train -> select.

Only starts this project's own subprocesses; never terminates another user's job.
"""
import argparse,json,os,signal,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'training/upstream')]
from audit import sha,raw_audit,converted_audit

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15)
    return [dict(index=i,used_mib=m,total_mib=t,utilization=u) for i,m,t,u in
            ([int(x.strip()) for x in line.split(',')] for line in raw.splitlines())]

def eligible_gpus(shared_gpu=None):
    rows=gpu_inventory()
    idle=[r['index'] for r in rows if r['used_mib']<1000 and r['utilization']<5]
    shared=[r['index'] for r in rows if shared_gpu is not None and r['index']==shared_gpu
            and r['total_mib']-r['used_mib']>=32768 and r['index'] not in idle]
    return idle+shared

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--max-wait-hours',type=float,default=6)
    p.add_argument('--allow-shared-gpu',type=int,default=None,
                   help='Explicitly authorized shared GPU; requires >=32GiB free. Idle GPUs are preferred.')
    args=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Training pipeline requires the authorized Linux server')
    import fcntl
    workspace=args.workspace.resolve();reports=workspace/'reports';reports.mkdir(exist_ok=True)
    lock=(workspace/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'task':'fold_towel','phase':'starting','pid':os.getpid(),'code_root':str(ROOT),
           'hardware_connected':False,'training_started':False,
           'authorized_shared_gpu':args.allow_shared_gpu}
    def status(phase,**values):
        state.update(phase=phase,updated_wall_ns=time.time_ns(),**values);atomic(reports/'pipeline_status.json',state)
        print('STATUS',phase,flush=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             PYTHONPATH=os.pathsep.join([str(ROOT/'training/upstream'),str(workspace/'dependencies'),str(ROOT/'inference')]),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCH_HOME=str(workspace/'cache/torch'))
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise InterruptedError('Operator stopped this pipeline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run(command,log_name,timeout):
        nonlocal child
        with (reports/log_name).open('a',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(state['phase'],child_pid=child.pid,child_log=log_name)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
                raise RuntimeError('Own pipeline stage exceeded its time budget: '+log_name)
            if code:raise RuntimeError(f'Own stage failed code {code}; inspect reports/{log_name}')
    try:
        status('verifying_uploaded_bytes')
        manifest=json.loads((workspace/'raw_manifest.json').read_text())
        archive=workspace/'raw_upload.tar'
        if sha(archive)!=manifest['archive_sha256']:raise ValueError('Uploaded archive SHA mismatch')
        raw_parent=workspace/'raw';raw_parent.mkdir(exist_ok=True)
        raw=raw_parent/'fold_towel'
        if not raw.exists():
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    target=(raw_parent/member.name).resolve()
                    if not target.is_relative_to(raw_parent) or not member.isfile():raise ValueError('Unsafe dataset archive member')
                tar.extractall(raw_parent)
        for name,info in manifest['files'].items():
            path=raw/name
            if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:raise ValueError('Raw source differs: '+name)
        ra=raw_audit(raw);atomic(reports/'raw_audit.json',ra)
        accepted_root=workspace/'accepted';(accepted_root/'fold_towel').mkdir(parents=True,exist_ok=True)
        for name in ra['accepted']:
            target=accepted_root/'fold_towel'/name
            if not target.exists():target.symlink_to(raw/name,target_is_directory=True)
        data=workspace/'converted'
        status('converting_recorded_data',accepted_episodes=len(ra['accepted']),rejected=ra['rejected'])
        if not (data/'meta.json').exists():
            if data.exists():raise ValueError('Incomplete conversion exists; preserved for inspection')
            run([sys.executable,'-u','-m','yam_train.data.convert','--raw',str(accepted_root),
                 '--out',str(data),'--workers','4'], 'conversion.log',3600)
        audit=converted_audit(data);atomic(reports/'converted_audit.json',audit)
        status('waiting_for_eligible_gpu',episodes=audit['episodes'],frames=audit['frames'],
               note='Idle GPU preferred; only the explicitly authorized shared GPU may also be used. No other jobs stopped.')
        deadline=time.monotonic()+args.max_wait_hours*3600;gpu_lock=None;gpu=None
        while time.monotonic()<deadline:
            for candidate in eligible_gpus(args.allow_shared_gpu):
                locks=Path.home()/'.cache/k1deler_training';locks.mkdir(parents=True,exist_ok=True)
                handle=(locks/f'gpu_{candidate}.lock').open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                time.sleep(3)
                if candidate not in eligible_gpus(args.allow_shared_gpu):handle.close();continue
                gpu_lock=handle;gpu=candidate;break
            if gpu is not None:break
            time.sleep(30)
        if gpu is None:raise TimeoutError('No eligible GPU became available within configured waiting budget; no training started')
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        state['allocation_at_launch']=gpu_inventory()
        status('real_data_smoke_training',gpu=gpu,training_started=True)
        smoke=workspace/'runs/smoke'
        if not (smoke/'ckpt/last.pt').is_file():
            run([sys.executable,'-u',str(ROOT/'training/fit.py'),'--data',str(data),'--out',str(smoke),
                 '--epochs','2','--smoke'],'smoke.log',1200)
        status('smoke_deployment_contract_check')
        run([sys.executable,'-u',str(ROOT/'training/check_contract.py'),'--checkpoint',str(smoke/'ckpt/last.pt'),
             '--data',str(data),'--report',str(reports/'smoke_contract.json')],'smoke_contract.log',300)
        run_dir=workspace/'runs/act_delta_seed42'
        status('training',gpu=gpu,epochs=5000,seed=42)
        run([sys.executable,'-u',str(ROOT/'training/fit.py'),'--data',str(data),'--out',str(run_dir),
             '--epochs','5000'],'train.log',43200)
        status('held_out_selection_and_export')
        # Releasing the training process frees this card. If a different card is
        # idle at evaluation time, use it under the same cooperative GPU lock.
        alternatives=[r['index'] for r in gpu_inventory() if r['index']!=gpu
                      and r['used_mib']<1000 and r['utilization']<5]
        for other in alternatives:
            handle=(Path.home()/'.cache/k1deler_training'/f'gpu_{other}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();continue
            gpu_lock.close();gpu_lock=handle;env['CUDA_VISIBLE_DEVICES']=str(other)
            status('held_out_selection_and_export',evaluation_gpu=other)
            break
        run([sys.executable,'-u',str(ROOT/'training/select_candidate.py'),'--run',str(run_dir),'--data',str(data),
             '--out',str(workspace/'artifacts')],'selection.log',7200)
        status('complete_offline',model_path=str(workspace/'artifacts/fold_towel_act_delta_seed42.pt'),
               model_sha256=sha(workspace/'artifacts/fold_towel_act_delta_seed42.pt'),robot_tested=False)
    except BaseException as e:
        status('failed_or_incomplete',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        if 'gpu_lock' in locals() and gpu_lock:gpu_lock.close()
        lock.close()

if __name__=='__main__':main()
