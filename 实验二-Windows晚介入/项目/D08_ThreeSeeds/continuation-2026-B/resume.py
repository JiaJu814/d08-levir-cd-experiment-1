"""Continue only seed 2026-B, preserving the original timed-out run."""
from pathlib import Path

import os
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
QUEUE = HERE.parent
ORIGINAL = QUEUE / "2026"
sys.path.insert(0, str(ORIGINAL))
import pilot
sys.path.insert(0, str(QUEUE))
import queue_runner

source_results = ORIGINAL / "results"
pilot.RESULTS = HERE / "results"
pilot.STATE = pilot.RESULTS / "task.json"


def preflight():
    import torch
    state = pilot.read(source_results / "task.json")
    assert state["status"] == "incomplete"
    assert pilot.read(QUEUE / "queue.json")["status"] == "blocked"
    previous = pilot.read(source_results / "semi10/result.json")
    assert previous["steps"] == 12044 and not previous["completed"]
    pre = pilot.read(source_results / "preflight.json")
    assert pre["stamps"] == state["stamps"] == pilot.validate_inputs()
    assert pilot.sha(source_results / "executed_worker.py") == pre["stamps"]["worker"]
    for name, digest in pre["stamps"]["manifests"].items():
        assert pilot.sha(source_results / "manifests" / (name+".json")) == digest
    path = source_results / "semi10/last.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["step"] == 12044 and saved["pseudo_totals"][0] == 3044
    assert saved["config"] == pre["stamps"]["config"]
    assert saved["code_sha256"] == pre["stamps"]["code"]
    assert saved["history"] == pilot.read(source_results / "semi10/history.json")
    for key in ("teacher", "model", "optimizer", "rng_python", "rng_numpy", "rng_torch",
                "rng_cuda", "rng_dataset", "rng_indices", "rng_udataset", "rng_uindices"):
        assert saved[key] is not None
    return pre, pilot.sha(path)


def aggregate():
    project = QUEUE.parent
    original_a = project / "D08_PairedPilot15000/results"
    late_b = project / "D08_LatePseudo15000/results"
    r42 = QUEUE / "42/results"
    r26 = source_results
    resumed = pilot.RESULTS
    rows = []
    for seed,a,b in [(17,original_a,late_b),(42,r42,r42),(2026,r26,resumed)]:
        aa = pilot.read(a/"summary.json")["branches"]["sl10"]
        bb = pilot.read(b/"summary.json")["branches"]["semi10"]
        assert bb["pseudo_totals"][0] == 6000
        rows.append(queue_runner.pair(seed,aa,bb,pilot.read(a/"sl10/history.json"),
                                     pilot.read(b/"semi10/history.json")))
    avg = sum(pair["gain_f1_pp"] for pair in rows)/3
    pilot.write(HERE/"three_seed_summary.json",dict(pairs=rows,mean_gain_f1_pp=avg,
        validation_target_met=avg>=3,all_warmups_match=all(p["warmup_matches"] for p in rows),
        continuation_step=12044,test_evaluated=False))


def run():
    import psutil
    if pilot.RESULTS.exists():
        raise SystemExit("Continuation already initialized")
    pre, digest = preflight()
    pilot.RESULTS.mkdir()
    for name in ("manifests","semi10"):
        shutil.copytree(source_results/name,pilot.RESULTS/name)
    for name in ("preflight.json","executed_worker.py"):
        shutil.copy2(source_results/name,pilot.RESULTS/name)
    assert pilot.sha(pilot.RESULTS/"semi10/last.pt")==digest
    began = time.time()
    state = dict(status="running",phase="semi10",pid=os.getpid(),
                 pid_created=psutil.Process().create_time(),start_epoch=began,
                 deadline_epoch=began+2100,training_cutoff_epoch=began+1800,
                 continuation_from=12044,source_checkpoint_sha256=digest,
                 original_task=str(source_results/"task.json"),stamps=pre["stamps"])
    pilot.write(pilot.STATE,state)
    try:
        pilot.worker("semi10",state["training_cutoff_epoch"])
        result = pilot.read(pilot.RESULTS/"semi10/result.json")
        state["status"]="complete" if result["completed"] and result["steps"]==15000 else "incomplete"
        state["end_epoch"]=time.time()
        pilot.report(state)
        if state["status"]=="complete":
            aggregate()
    except Exception:
        state.update(status="blocked",error=traceback.format_exc())
        raise
    finally:
        state.setdefault("end_epoch",time.time())
        pilot.write(pilot.STATE,state)


if __name__=="__main__":
    if sys.argv[1]=="verify":
        _, digest=preflight()
        print("Checkpoint verified at step 12044:",digest)
    else:
        run()


