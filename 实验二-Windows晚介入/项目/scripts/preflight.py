"""Deployment gate; no LEVIR training and no experiment timer starts here."""
import json
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from common import ROOT, RESULTS, read, write, sha, safe_members, metric, code_fingerprints

def main():
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT/'baseline'))
    from cd.model import ChangeDetector
    from cd.losses import supervised_loss
    from cd.metrics import Metrics
    assert sys.platform == 'win32', 'Target preflight requires native Windows; no silent platform fallback'
    assert sys.version_info[:2] == (3,11), 'Use Python 3.11'
    assert torch.__version__.split('+')[0] == '2.9.1', 'Pinned torch version mismatch'
    assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver; do not use CPU fallback'
    assert torch.version.cuda == '12.8', 'Expected cu128 wheel'
    assert shutil.disk_usage(ROOT).free >= 12*1024**3, 'Need at least 12 GiB free after deployment'
    torch.set_num_threads(4)
    model = ChangeDetector().cuda().train()
    a = torch.rand(2,3,64,64,device='cuda')
    b = torch.rand_like(a)
    y = torch.zeros(2,1,64,64,device='cuda'); y[:,:,16:48,16:48] = 1
    logits = model(a,b)
    assert logits.shape == y.shape and torch.isfinite(logits).all()
    loss = supervised_loss(logits,y)
    loss.backward(); torch.cuda.synchronize()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    del model,a,b,y,logits,loss
    torch.cuda.empty_cache()
    m=Metrics(); m.update(np.array([[1,1],[0,0]]),np.array([[1,0],[1,0]]))
    observed=m.result(); expected=metric(1,1,1,1)
    for k,v in expected.items(): assert observed[k] == v, (k,observed[k],v)
    inventory=read(ROOT/'assets.json')
    extracted={}
    for split in ['train','val']:
        p=ROOT/'data_archives'/f'{split}.zip'
        assert sha(p) == inventory['splits'][split]['sha256'], 'Archive hash mismatch: '+split
        dest=ROOT/'data/LEVIR-CD'
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None, 'CRC mismatch: '+split
            for info in safe_members(z):
                parts=list(Path(info.filename).parts)
                if not parts or '__MACOSX' in parts or any(x.startswith('.') for x in parts): continue
                if parts[0]==split: parts=parts[1:]
                if not parts or parts[0] not in ['A','B','label']: continue
                out=dest/split/Path(*parts)
                if info.is_dir(): out.mkdir(parents=True,exist_ok=True); continue
                out.parent.mkdir(parents=True,exist_ok=True)
                temp=out.with_name(out.name+'.tmp')
                with z.open(info) as src, temp.open('wb') as target: shutil.copyfileobj(src,target)
                temp.replace(out)
        extracted[split]=str(dest/split)
    sealed=ROOT/'sealed_test/test.zip'
    assert sha(sealed) == inventory['splits']['test']['sha256'], 'Sealed test archive hash mismatch'
    assert not (ROOT/'data/LEVIR-CD/test').exists(), 'Stage1 must not expose extracted test data'
    deps=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (RESULTS/'environment').mkdir(parents=True,exist_ok=True)
    (RESULTS/'environment/pip-freeze.txt').write_text(deps,encoding='utf-8')
    smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=20)
    (RESULTS/'environment/nvidia-smi.txt').write_text(smi.stdout+smi.stderr,encoding='utf-8')
    write(RESULTS/'preflight.json', dict(passed=True, platform=platform.platform(), python=sys.version,
          torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
          capability=list(torch.cuda.get_device_capability(0)), arch_list=torch.cuda.get_arch_list(),
          real_cuda_forward_backward_passed=True, metric_fixture_passed=True,
          archive_integrity_passed=True, test_sealed=True, extracted=extracted,
          code_sha256=code_fingerprints(), experiment_timer_started=False))
    print('PREFLIGHT PASSED. Experiment timer has NOT started.',flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        RESULTS.mkdir(parents=True,exist_ok=True)
        write(RESULTS/'preflight.json',dict(passed=False,error=traceback.format_exc(),experiment_timer_started=False))
        raise
