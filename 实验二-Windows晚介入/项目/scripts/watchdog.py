"""Independent deadline guard; only controls registered task process identities."""
import argparse
import os
import subprocess
import sys
import time
import psutil
from common import ROOT, RESULTS, read, write

def alive(pid,created):
    try:return psutil.Process(pid).is_running() and abs(psutil.Process(pid).create_time()-created)<0.1
    except psutil.Error:return False

def terminate_registered():
    path=RESULTS/'worker_identity.json'
    if path.exists():
        x=read(path)
        if alive(x['pid'],x['created']):
            p=psutil.Process(x['pid']);children=p.children(recursive=True)
            for q in children+[p]:
                try:q.terminate()
                except psutil.Error:pass
            _,running=psutil.wait_procs(children+[p],timeout=5)
            for q in running:
                try:q.kill()
                except psutil.Error:pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=int,required=True);parser.add_argument('--created',type=float,required=True)
    a=parser.parse_args();state=read(RESULTS/'task.json'); cfg=read(ROOT/'config.json')
    wall0=time.time();mono0=time.monotonic(); stop_at=state['deadline_epoch']-cfg['report_reserve_seconds']
    while True:
        state=read(RESULTS/'task.json')
        if state.get('status') in ['complete','blocked','stopped']:break
        wall=time.time();elapsed=time.monotonic()-mono0
        hard=wall>=state['deadline_epoch'] or elapsed>=max(0,state['deadline_epoch']-wall0)
        drift=abs((wall-wall0)-elapsed)>120
        parent_dead=not alive(a.parent,a.created)
        if wall>=stop_at or hard or drift or parent_dead:
            (RESULTS/'STOP').touch()
        if hard or drift or parent_dead:
            if not hard:time.sleep(5)
            terminate_registered()
            reason='deadline' if hard else ('clock_changed' if drift else 'supervisor_exit')
            state.update(status='stopped',reason=reason,end_epoch=time.time());write(RESULTS/'task.json',state)
            subprocess.run([sys.executable,str(ROOT/'scripts/report.py')],cwd=ROOT,timeout=120)
            break
        time.sleep(1)

if __name__=='__main__':main()
