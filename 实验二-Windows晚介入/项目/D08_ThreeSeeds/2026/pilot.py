"""Run the approved single-seed D08 comparison in its own results directory."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RESULTS = ROOT / "results"
OLD = ROOT / "inputs"
STATE = RESULTS / "task.json"
BUDGET = 8100
RESERVE = 600
GRACE = 20
MANIFESTS = {"labeled10": (44, "train", True), "unlabeled10": (401, "train", False),
             "val": (64, "val", True)}
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "baseline"))
from common import read, write, sha, code_fingerprints


def config():
    c = read(PROJECT / "config.json")
    assert c["seed"] == 17 and c["supervised_steps"] == 10000
    c.update(seed=2026, semi_warmup_steps=9000, supervised_steps=15000, pseudo_high=.95, pseudo_low=.01, budget_seconds=BUDGET,
             report_reserve_seconds=RESERVE)
    assert all(c[k] == v for k,v in {'protocol': 'D08_windows_stage1_v1', 'seed': 17, 'budget_seconds': 4800, 'report_reserve_seconds': 600, 'device': 'cuda', 'crop': 256, 'batch_size': 4, 'lr': 0.001, 'weight_decay': 0.0001, 'grad_clip': 5.0, 'supervised_steps': 15000, 'baseline_cap_seconds_each': 4500, 'validate_every': 500, 'checkpoint_every': 100, 'micro_steps': 400, 'micro_cap_seconds': 900, 'micro_required_f1': 0.9, 'micro_required_loss_ratio': 0.5, 'semi_warmup_steps': 9000, 'semi_ramp_steps': 2000, 'semi_weight': 1.0, 'ema_decay': 0.99, 'pseudo_high': 0.95, 'pseudo_low': 0.01, 'pseudo_class_balance': True, 'threshold': 0.5, 'boundary_weight': 0.0, 'max_retries': 2, 'validation_stability_last_n': 3, 'validation_stability_range': 0.01, 'formal_target_gain_pp': 3.0, 'formal_paired_seeds_required': 3}.items() if k not in {"seed", "budget_seconds", "report_reserve_seconds"})
    return c


def source():
    code = (PROJECT / "scripts/worker.py").read_text(encoding="utf-8-sig")
    changes = [
        ("pw=min(20.,(total-pos)/pos)", "pw=1.0"),
        ("start=0;best=-1.;history=[];seconds_prior=0.",
         "start=0;best=-1.;history=[];seconds_prior=0.;pseudo_totals=[0,0,0,0,0]"),
        ("seconds_prior=saved['seconds']", "seconds_prior=saved['seconds'];pseudo_totals=saved['pseudo_totals']"),
        ("step=step,best=best,history=history,seconds=", "step=step,best=best,history=history,pseudo_totals=pseudo_totals,seconds="),
        ("        loss=sl+c['semi_weight']*ramp*ul",
         "            collect_pseudo(q,c,pseudo_totals)\n        loss=sl+c['semi_weight']*ramp*ul"),
        ("ramp=ramp,**stats,val=result)", "ramp=ramp,**stats,pseudo_totals=list(pseudo_totals),val=result)"),
        ("unique_labeled_pairs=len(rs),test_accessed=False,formal_result=False,",
         "unique_labeled_pairs=len(rs),pseudo_totals=list(pseudo_totals),test_accessed=False,formal_result=False,"),
    ]
    changes.extend([
        ("cosine fixed 10000 updates", "cosine fixed 15000 updates"),
        ("    with torch.no_grad():\n        for i,r in enumerate(rs):",
         "    histogram=np.zeros((2,10001),dtype=np.int64)\n    with torch.no_grad():\n        for i,r in enumerate(rs):"),
        ("            p=predict(model,a,b,'cuda',256,4)>=0.5",
         "            probability=predict(model,a,b,'cuda',256,4)\n"
         "            p=probability>=0.5\n"
         "            bins=np.floor(probability.astype(np.float64)*10000).astype(np.int32)\n"
         "            histogram[0]+=np.bincount(bins[truth],minlength=10001)\n"
         "            histogram[1]+=np.bincount(bins[~truth],minlength=10001)"),
        ("    write(out/'latest_validation.json',result)",
         "    result.update(precision_recall(histogram))\n    write(out/'latest_validation.json',result)"),
    ])
    for old,new in changes:
        if code.count(old) != 1:
            raise RuntimeError("Historical worker layout changed: "+old)
        code=code.replace(old,new,1)
    return code


def precision_recall(histogram):
    """Global pixel-level PR metrics from 0.0001-wide score bins."""
    import numpy as np
    tp, fp = np.cumsum(histogram[:, ::-1], axis=1)
    if not tp[-1]:
        return {"ap": None, "precision_at_recall": {}}
    recall = tp / tp[-1]
    precision = np.divide(tp, tp + fp, out=np.ones_like(recall), where=(tp + fp) > 0)
    return {
        "ap": float(np.sum(precision * np.diff(recall, prepend=0))),
        "precision_at_recall": {
            str(target): float(precision[np.searchsorted(recall, target)])
            for target in (.4, .5, .55, .6, .7)
        },
    }


def collect_pseudo(q, c, totals):
    import torch
    pos, neg = torch.stack(((q >= c["pseudo_high"]).sum(), (q <= c["pseudo_low"]).sum())).tolist()
    for i,n in enumerate((1, int(pos > 0), q.numel(), pos, neg)):
        totals[i] += n


def validate_inputs():
    import torch
    import psutil
    assert sys.version_info[:2] == (3,11)
    assert torch.__version__ == "2.9.1+cu128" and torch.cuda.is_available()
    assert shutil.disk_usage(ROOT).free >= 1024**3
    names = {}
    for key,(count,split,has_label) in MANIFESTS.items():
        rows = read(OLD / "manifests" / (key+".json"))
        names[key] = [row["name"] for row in rows]
        assert len(rows) == len(set(names[key])) == count
        for row in rows:
            expected = {"a":"A", "b":"B"}
            if has_label:
                expected["label"]="label"
            assert set(row) == set(expected) | {"name"}
            name = row["name"]
            assert Path(name).name == name and name.endswith(".png")
            for field,folder in expected.items():
                p = (PROJECT / "data/LEVIR-CD" / split / folder / name).resolve()
                assert Path(row[field]).resolve() == p and p.is_file()
    assert names["labeled10"] == read(OLD/"labeled_names.json")
    assert not (set(names["labeled10"]) & set(names["unlabeled10"]))
    assert set(names["labeled10"]+names["unlabeled10"]) == {
        p.name for p in (PROJECT/"data/LEVIR-CD/train/A").glob("*.png")}
    compiled=source()
    compile(compiled, "pilot_worker.py", "exec")
    old=code_fingerprints()
    verified=read(PROJECT/"D08_BaselineSupplement/results/preflight.json")
    assert verified["passed"]
    assert all(old[k] == v for k,v in verified["core_hashes"].items())
    return dict(code=old, runner=sha(Path(__file__)), worker=hashlib.sha256(compiled.encode()).hexdigest(),
                manifests={k:sha(OLD/"manifests"/(k+".json")) for k in MANIFESTS}, config=config())


def prepare():
    import torch
    from cd.model import ChangeDetector
    from cd.losses import pseudo_loss
    stamps=validate_inputs()
    import numpy as np
    perfect=np.array([[0,0,2],[2,0,0]])
    assert precision_recall(perfect)["ap"] == 1.0
    reversed_scores=np.array([[2,0,0],[0,0,2]])
    assert precision_recall(reversed_scores)["ap"] == .5
    ids=[]
    for _ in range(2):
        from cd.utils import seed_everything
        seed_everything(2026)
        model=ChangeDetector().cuda().eval()
        ids.append(hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                          for p in model.state_dict().values())).hexdigest())
    assert ids[0]==ids[1]
    with torch.no_grad():
        x=torch.zeros((1,3,64,64),device="cuda")
        y=model(x,x)
        assert y.shape==(1,1,64,64) and torch.isfinite(y).all()
        q=torch.tensor([[[[.95,.01,.05,.5]]]],device="cuda")
        loss,stats=pseudo_loss(torch.zeros_like(q),q,high=.95,low=.01,class_balance=True)
        totals=[0,0,0,0,0]
        collect_pseudo(q,config(),totals)
        assert torch.isfinite(loss) and stats["coverage"]==.5 and totals==[1,1,4,1,1]
    write(RESULTS/"preflight.json",dict(passed=True, stamps=stamps, device=torch.cuda.get_device_name(0),
                                      identical_initialization=ids[0], synthetic_cuda_forward=True))
    print("Preflight passed. No training launched.")


def worker(branch, deadline):
    import torch
    torch.set_num_threads(4)
    state=read(STATE)
    assert state["status"]=="running" and state["phase"]==branch
    pre=read(RESULTS/"preflight.json")
    assert pre["stamps"]==validate_inputs()==state["stamps"]
    for name,wanted in pre["stamps"]["manifests"].items():
        assert sha(RESULTS/"manifests"/(name+".json")) == wanted
    path=RESULTS/"executed_worker.py"
    assert sha(path)==pre["stamps"]["worker"]
    spec=importlib.util.spec_from_file_location("pilot_worker",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS=RESULTS
    module.collect_pseudo=collect_pseudo
    module.precision_recall=precision_recall
    module.code_fingerprints=lambda:pre["stamps"]["code"]
    module.train(pre["stamps"]["config"],branch,deadline)


def report(state):
    summary={"task":state,"branches":{}}
    for branch in ("sl10","semi10"):
        folder=RESULTS/branch
        path=folder/"result.json"
        if path.exists():
            result=read(path)
            history_path=folder/"history.json"
            history=read(history_path) if history_path.exists() else []
            best=max((r for r in history if r["val"]["f1"] is not None),
                     key=lambda r:r["val"]["f1"],default=None)
            result["best_validation"]=best
            b,positive_batches,pixels,pos,neg=result["pseudo_totals"]
            if b:
                result["pseudo_label_summary"]=dict(batches=b, positive_batches=positive_batches,
                    positive_batch_fraction=positive_batches/b,pixel_count=pixels,
                    positive_coverage=pos/pixels,negative_coverage=neg/pixels)
            summary["branches"][branch]=result
    branches=summary["branches"]
    if len(branches)==2 and all(v["completed"] for v in branches.values()):
        summary["best_validation_f1_gain_pp"]=100*(branches["semi10"]["best_val_f1"]-branches["sl10"]["best_val_f1"])
    write(RESULTS/"summary.json",summary)


def stop_tree(process):
    import psutil
    try:
        parent=psutil.Process(process.pid)
        children=parent.children(recursive=True)
        for child in reversed(children):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def run():
    import psutil
    if STATE.exists():
        print(json.dumps(read(STATE),ensure_ascii=True))
        return
    stamps=validate_inputs()
    assert read(RESULTS/"preflight.json")["stamps"]==stamps
    start=time.time()
    cutoff=start+BUDGET-RESERVE
    state=dict(status="running",phase="setup",start_epoch=start,deadline_epoch=start+BUDGET,
               training_cutoff_epoch=cutoff,stamps=stamps,pid=os.getpid())
    state["pid_created"]=psutil.Process(state["pid"]).create_time()
    claim = RESULTS / f"claim-{os.getpid()}.json"
    try:
        claim.write_text(json.dumps(state), encoding="utf-8")
        os.link(claim, STATE)
    except FileExistsError:
        raise SystemExit("An experiment identity already exists")
    finally:
        claim.unlink(missing_ok=True)
    process=None
    monotonic_cutoff=time.monotonic()+BUDGET-RESERVE
    try:
        for name in MANIFESTS:
            dest=RESULTS/"manifests"/(name+".json")
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(OLD/"manifests"/(name+".json"),dest)
        (RESULTS/"executed_worker.py").write_bytes(source().encode("utf-8"))
        for branch in ("sl10","semi10"):
            if time.time()>=cutoff-GRACE or time.monotonic()>=monotonic_cutoff-GRACE:
                break
            state["phase"]=branch
            write(STATE,state)
            with (RESULTS/(branch+".log")).open("ab") as log:
                process=subprocess.Popen([sys.executable,"-B",str(__file__),"worker",branch,
                                          str(cutoff-GRACE)],stdout=log,stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    code=process.wait(timeout=max(0,min(cutoff-time.time(),monotonic_cutoff-time.monotonic())))
                except subprocess.TimeoutExpired:
                    stop_tree(process)
                    process.wait(timeout=5)
                    break
                if code:
                    raise RuntimeError(f"{branch} exited {code}; see its log")
                if not read(RESULTS/branch/"result.json")["completed"]:
                    break
        complete=all((RESULTS/b/"result.json").exists() and read(RESULTS/b/"result.json")["completed"]
                     for b in ("sl10","semi10"))
        state["status"]="complete" if complete else "incomplete"
    except Exception:
        state.update(status="blocked",error=traceback.format_exc())
        raise
    finally:
        if process is not None and process.poll() is None:
            stop_tree(process)
        state["end_epoch"]=time.time()
        try:
            report(state)
        finally:
            write(STATE,state)


def status():
    state=read(STATE) if STATE.exists() else {"status":"not_started"}
    output={k:v for k,v in state.items() if k!="stamps"}
    if state["status"]=="running":
        import psutil
        try:
            p=psutil.Process(state["pid"])
            output["supervisor_alive"]=abs(p.create_time()-state["pid_created"])<.1
        except psutil.NoSuchProcess:
            output["supervisor_alive"]=False
    for b in ("sl10","semi10"):
        p=RESULTS/b/"runtime.json"
        if p.exists():output[b]=read(p)
    output["report_ready"]=(RESULTS/"summary.json").is_file()
    print(json.dumps(output,ensure_ascii=True))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","run","status","worker"])
    p.add_argument("branch",nargs="?",choices=["sl10","semi10"])
    p.add_argument("deadline",nargs="?",type=float)
    a=p.parse_args()
    if a.mode=="worker":
        worker(a.branch,a.deadline)
    else:
        {"prepare":prepare,"run":run,"status":status}[a.mode]()


