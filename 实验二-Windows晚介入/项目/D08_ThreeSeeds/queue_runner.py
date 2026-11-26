"""Run two fixed-budget D08 pairs sequentially and assemble the validation result."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
STATE = ROOT / "queue.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def metrics(branch):
    assert branch["completed"] and branch["steps"] == 15000
    best = branch["best_validation"]
    val = best["val"]
    assert val["threshold"] == .5 and not val["test_accessed"]
    return {"step":best["step"], **{k:val[k] for k in ("f1","ap","precision","recall")}}


def pair(seed, a, b, ah, bh):
    a,b = metrics(a),metrics(b)
    keys = ("tp","fp","fn","tn")
    head_a = [(v["step"],*(v["val"][k] for k in keys)) for v in ah if v["step"]<=9000]
    head_b = [(v["step"],*(v["val"][k] for k in keys)) for v in bh if v["step"]<=9000]
    assert len(head_a)==19 and head_a[-1][0]==9000
    return dict(seed=seed,A=a,B=b,warmup_matches=(head_a==head_b),
                gain_f1_pp=100*(b["f1"]-a["f1"]),gain_ap_pp=100*(b["ap"]-a["ap"]))


def summarize():
    a17=PROJECT/"D08_PairedPilot15000/results"
    b17=PROJECT/"D08_LatePseudo15000/results"
    rows=[pair(17,read(a17/"summary.json")["branches"]["sl10"],
              read(b17/"summary.json")["branches"]["semi10"],
              read(a17/"sl10/history.json"),read(b17/"semi10/history.json"))]
    for seed in (42,2026):
        r=ROOT/str(seed)/"results"
        branches=read(r/"summary.json")["branches"]
        assert branches["semi10"]["pseudo_totals"][0]==6000
        rows.append(pair(seed,branches["sl10"],branches["semi10"],
                         read(r/"sl10/history.json"),read(r/"semi10/history.json")))
    avg=sum(row["gain_f1_pp"] for row in rows)/3
    write(ROOT/"summary.json",dict(pairs=rows,mean_gain_f1_pp=avg,
          validation_target_met=avg>=3,all_warmups_match=all(row["warmup_matches"] for row in rows),
          test_evaluated=False))


def status():
    if not STATE.exists():
        print("not_started")
        return
    print(STATE.read_text(encoding="utf-8"))
    for seed in (42,2026):
        directory=ROOT/str(seed)
        subprocess.run([sys.executable,"-B",str(directory/"pilot.py"),"status"],check=True)


def run():
    now=time.time()
    state=dict(status="running",pid=os.getpid(),started=now,seed=None)
    claim=ROOT/f"claim-{os.getpid()}.json"
    try:
        claim.write_text(json.dumps(state),encoding="utf-8")
        os.link(claim,STATE)
    finally:
        claim.unlink(missing_ok=True)
    try:
        for seed in (42,2026):
            state["seed"]=seed
            write(STATE,state)
            path=ROOT/str(seed)/"pilot.py"
            subprocess.run([sys.executable,"-B","-u",str(path),"run"],check=True)
            verified=read(path.parent/"results/task.json")
            if verified["status"]!="complete":
                raise RuntimeError(f"Seed {seed} ended as {verified['status']}")
        summarize()
        state["status"]="complete"
    except Exception:
        state.update(status="blocked", error=traceback.format_exc())
        raise
    finally:
        state["ended"]=time.time()
        write(STATE,state)


if __name__=="__main__":
    {"status":status,"run":run}[sys.argv[1]]()
