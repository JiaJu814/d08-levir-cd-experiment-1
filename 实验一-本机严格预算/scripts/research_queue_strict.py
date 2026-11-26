"""冻结协议后执行完整研究矩阵，最多并行两次MPS训练，最终评价并生成报告。"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION=ROOT/'versions/strict_budget_v3'
ENTRY=VERSION/'main.py'

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temporary,path)
def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def jobs_for(full):
    jobs=[(10,g,s) for s in (17,42,2026) for g in 'AB']
    jobs += [(10,g,s) for s in (17,42,2026) for g in 'CD']
    jobs += [(100,'A',s) for s in (17,42,2026)]
    if full:jobs += [(p,g,s) for p in (5,20) for s in (17,42,2026) for g in 'ABCD']
    return [{'percent':p,'group':g,'seed':s,'id':f'{p}pct_{g}_seed{s}','status':'queued'} for p,g,s in jobs]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(VERSION/'configs/base.json'));ap.add_argument('--full',action='store_true');ap.add_argument('--execute',action='store_true');ap.add_argument('--max-parallel',type=int,choices=[1,2],default=2);ap.add_argument('--evaluate',action='store_true');ap.add_argument('--illumination',action='store_true');args=ap.parse_args()
    config=read(args.config);run_root=Path(config['run_root']);plan=jobs_for(args.full)
    if not args.execute:
        print(json.dumps({'jobs':plan,'total':len(plan),'max_parallel':args.max_parallel,'test_evaluation':args.evaluate},ensure_ascii=False,indent=2));return
    folder=ROOT/'work/research_queue_strict';folder.mkdir(parents=True,exist_ok=True)
    pidfile=folder/'pid'
    if pidfile.exists() and alive(int(pidfile.read_text())):raise RuntimeError('研究队列已在运行，不能重复启动。')
    pidfile.write_text(str(os.getpid()))
    frozen=folder/'frozen_config.json';freeze=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    if frozen.exists() and read(frozen)!=config:raise ValueError('队列配置已冻结，另建版本才能更换。')
    write(frozen,config)
    active={};handles={}
    milestone_file=folder/'ten_percent_milestone.json'
    def evaluate_jobs(selected):
        for job in selected:
            out=run_root/job['id']
            if not (out/'test.json').exists():
                subprocess.run([sys.executable,str(ENTRY),'evaluate','--checkpoint',str(out/'best.pt'),'--split','test','--device','mps'],cwd=ROOT,check=True)
            if args.illumination and job['percent']==10:
                for factor in (.8,1.2):
                    if not (out/('test_brightness'+str(factor)+'.json')).exists():
                        subprocess.run([sys.executable,str(ENTRY),'evaluate','--checkpoint',str(out/'best.pt'),'--split','test','--device','mps','--brightness',str(factor)],cwd=ROOT,check=True)
    def summarize():
        subprocess.run([sys.executable,str(ENTRY),'summarize','--run-root',str(run_root)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/research_report.py'),'--run-root',str(run_root)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_results.py'),'--run-root',str(run_root)],cwd=ROOT,check=True)
    def status(stage,error=None):
        for job in plan:
            history=run_root/job['id']/'history.jsonl'
            if history.exists():
                try:
                    row=json.loads(history.read_text().splitlines()[-1]);job['completed_epochs']=row['epoch']
                except (ValueError,IndexError):pass
        state={'stage':stage,'version':'strict_budget_v3','updated_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'config_sha256':freeze,'run_root':str(run_root),'total_jobs':len(plan),'completed_jobs':sum(j['status']=='completed' for j in plan),'active':[j['id'] for j in plan if j['status']=='running'],'jobs':plan,'test_set_used_for_tuning':False}
        if error:state['error']=str(error)
        write(folder/'status.json',state);write(ROOT/'docs/status.json',state)
    try:
        for job in plan:
            out=run_root/job['id']
            if (out/'training_complete.json').exists():job['status']='completed'
        status('training_matrix_running')
        while any(j['status']!='completed' for j in plan):
            # 任何训练失败均停止启动新任务，保留其他进程与全部断点。
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    job=next(j for j in plan if j['id']==name);handles.pop(name).close();active.pop(name)
                    if code or not (run_root/name/'training_complete.json').exists():
                        job['status']='failed';raise RuntimeError(name+'失败，请检查'+str(folder/(name+'.log')))
                    job['status']='completed';job['best_val_f1']=read(run_root/name/'training_complete.json')['best_val_f1'];print('completed',name,job['best_val_f1'],flush=True)
            ten=[j for j in plan if j['percent']==10 and j['group'] in 'AB']
            if args.evaluate and all(j['status']=='completed' for j in ten) and not milestone_file.exists():
                status('evaluating_ten_percent_milestone')
                evaluate_jobs(ten);summarize();write(milestone_file,read(run_root/'goal_assessment.json'))
                print('10% primary A/B milestone complete; boundary and budget experiments continue',flush=True)
            for job in plan:
                if len(active)>=args.max_parallel:break
                if job['status']!='queued':continue
                out=run_root/job['id']
                cmd=[sys.executable,str(ENTRY),'train','--config',str(frozen),'--group',job['group'],'--percent',str(job['percent']),'--seed',str(job['seed']),'--device','mps']
                if out.exists():
                    if (out/'last.pt').exists():cmd.append('--resume')
                    else:raise RuntimeError('已有不完整且无断点目录，禁止覆盖: '+str(out))
                log=(folder/(job['id']+'.log')).open('a');proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[job['id']]=proc;handles[job['id']]=log;job.update(status='running',pid=proc.pid);print('started',job['id'],proc.pid,flush=True)
            status('training_matrix_running')
            time.sleep(10)
        if args.evaluate:
            status('evaluating_frozen_models')
            evaluate_jobs(plan)
            status('summarizing_results');summarize()
            result=read(run_root/'goal_assessment.json');status('complete_target_met' if result['target_met'] else 'complete_target_not_met')
            print('Final target assessment',json.dumps(result,ensure_ascii=False),flush=True)
        else:status('training_complete_awaiting_explicit_test_evaluation')
    except BaseException as exc:
        for name,proc in active.items():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
            job=next(j for j in plan if j['id']==name)
            if job['status']=='running':job['status']='interrupted'
        status('queue_needs_attention',repr(exc));raise
    finally:
        for handle in handles.values():handle.close()

if __name__=='__main__':main()
