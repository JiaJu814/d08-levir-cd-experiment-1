"""One fixed stage per package. A restart never resets the deadline."""
import json
import os
import subprocess
import sys
import time
import traceback
import psutil
from common import ROOT,RESULTS,read,write,now,code_fingerprints
from watchdog import alive,terminate_registered

def event(event_kind,**kwargs):
    with (RESULTS/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time=now(),event=event_kind,**kwargs),ensure_ascii=False)+'\n')

def run_phase(kind,deadline,max_retries=0):
    for attempt in range(max_retries+1):
        if now()>=deadline or (RESULTS/'STOP').exists():return False
        logfile=RESULTS/'logs'/f'{kind}_attempt{attempt}.log'
        command=[sys.executable,'-u',str(ROOT/'scripts/worker.py'),kind,'--deadline',str(deadline)]
        with logfile.open('a',encoding='utf-8') as log:
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(RESULTS/'worker_identity.json',dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),kind=kind))
            event('phase_started',kind=kind,attempt=attempt,deadline=deadline,command=command,log=str(logfile))
            while p.poll() is None:
                if now()>=deadline:
                    # Worker checks its deadline and saves. At most 20 seconds grace, within training cutoff.
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:terminate_registered()
                    break
                if (RESULTS/'STOP').exists():
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:terminate_registered()
                    break
                time.sleep(1)
            rc=p.wait();event('phase_ended',kind=kind,attempt=attempt,returncode=rc)
        if rc==0:return True
        text=logfile.read_text(encoding='utf-8',errors='replace')
        # No semantic changes, no OOM batch-size search. Retry transient IO only.
        retryable=any(x in text for x in ['PermissionError','WinError 32','temporarily unavailable'])
        if not retryable:return False
        event('transient_retry_only',kind=kind,attempt=attempt)
    return False

def baseline_gate(cfg):
    reasons=[]; detail={}
    for kind in ['sl10','sl100']:
        rp=RESULTS/kind/'result.json'; hp=RESULTS/kind/'history.json'
        if not rp.exists() or not hp.exists():reasons.append(kind+': missing result');continue
        r=read(rp);h=read(hp);n=cfg['validation_stability_last_n']
        f=[x['val']['f1'] for x in h[-n:]]
        stable=len(f)==n and all(x is not None for x in f) and max(f)-min(f)<=cfg['validation_stability_range']
        near_best=bool(f) and f[-1] is not None and r['best_val_f1'] is not None and r['best_val_f1']-f[-1]<=cfg['validation_stability_range']
        detail[kind]=dict(completed=r['completed'],last_validation_f1=f,stable=stable,near_best=near_best)
        if not r['completed']:reasons.append(kind+': fixed update budget unfinished')
        if not stable or not near_best:reasons.append(kind+': stability screen unresolved')
    if all((RESULTS/k/'result.json').exists() for k in ['sl10','sl100']):
        x=read(RESULTS/'sl10/result.json');y=read(RESULTS/'sl100/result.json')
        if x['best_val_f1'] is not None and y['best_val_f1'] is not None:
            gain=y['best_val_f1']-x['best_val_f1'];detail['full_vs10_gain_pp']=gain*100
            if gain< -0.01:reasons.append('100% worse than10% by over1pp: investigate before expansion')
    passed=not reasons
    write(RESULTS/'baseline_gate.json',dict(passed=passed,reasons=reasons,detail=detail,
        caution='Heuristic expansion gate, not proof of convergence or bug-free implementation.'))
    return passed

def acquire():
    lock=RESULTS/'run.lock'
    if lock.exists():
        x=read(lock)
        if alive(x['pid'],x['created']):raise RuntimeError('Already running; inspect results/task.json and logs')
        lock.unlink()
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),f)
    return lock

def main():
    RESULTS.mkdir(exist_ok=True);(RESULTS/'logs').mkdir(exist_ok=True)
    cfg=read(ROOT/'config.json'); pre=read(RESULTS/'preflight.json')
    assert pre['passed'] and pre['code_sha256']==code_fingerprints(),'Preflight failed or code changed; rerun deployment gate'
    lock=acquire(); watchdog=None
    try:
        path=RESULTS/'task.json'
        if path.exists():
            state=read(path)
            if state.get('status') in ['complete','blocked','stopped']:
                print('Existing stage already ended; no automatic new budget. See results/REPORT.md');return
            assert state['code_sha256']==code_fingerprints(),'Existing task fingerprint mismatch'
        else:
            t=now();state=dict(status='running',start_epoch=t,deadline_epoch=t+cfg['budget_seconds'],
                code_sha256=code_fingerprints(),formal_goal_achieved=False,seed=cfg['seed'])
            write(path,state);write(RESULTS/'code_sha256.json',state['code_sha256']);write(RESULTS/'config_snapshot.json',cfg)
        cutoff=state['deadline_epoch']-cfg['report_reserve_seconds']-30
        if now()>=cutoff:raise RuntimeError('Original training budget expired; do not reset deadline')
        # Keep Windows awake while this supervisor is alive, without changing global power plan.
        if sys.platform=='win32':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        with (RESULTS/'logs/watchdog.log').open('a',encoding='utf-8') as log:
            watchdog=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/watchdog.py'),'--parent',str(os.getpid()),
                    '--created',str(psutil.Process().create_time())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=='win32' else 0)
        event('stage_started_or_resumed',deadline=state['deadline_epoch'])
        if not run_phase('audit',min(cutoff,now()+1200),cfg['max_retries']):raise RuntimeError('Data/metric audit failed')
        if not run_phase('micro',min(cutoff,now()+cfg['micro_cap_seconds'])):raise RuntimeError('Micro learning screen unresolved')
        for kind in ['sl10','sl100']:
            rp=RESULTS/kind/'result.json'
            if rp.exists() and read(rp).get('completed'):continue
            if not run_phase(kind,min(cutoff,now()+cfg['baseline_cap_seconds_each']),cfg['max_retries']):
                raise RuntimeError(kind+' failed; evidence retained')
        passed=baseline_gate(cfg)
        if passed:
            a=read(RESULTS/'sl10/result.json'); estimate=2.5*a['seconds']+120
            if cutoff-now()>=estimate:
                event('semi_frozen_candidate_started',estimated_seconds=estimate)
                ok=run_phase('semi10',cutoff,cfg['max_retries'])
                if not ok:event('semi_failed',note='no alternative candidate allowed')
            else:event('semi_skipped',reason='insufficient budget',estimated_seconds=estimate,remaining=cutoff-now())
        else:event('semi_skipped',reason='baseline gate unresolved')
        state.update(status='complete',reason='diagnostic workflow ended; inspect gate and completeness',end_epoch=now())
        write(path,state)
    except Exception:
        event('stage_blocked',error=traceback.format_exc())
        path=RESULTS/'task.json'
        state=read(path) if path.exists() else dict(start_epoch=None,deadline_epoch=None)
        state.update(status='blocked',reason=traceback.format_exc(),end_epoch=now());write(path,state)
    finally:
        (RESULTS/'STOP').touch()
        terminate_registered()
        from report import build
        build()
        if sys.platform=='win32':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if lock.exists():lock.unlink()
        if watchdog:
            try:watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:watchdog.terminate()

if __name__=='__main__':main()
