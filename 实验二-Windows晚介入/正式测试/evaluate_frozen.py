from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from io import BytesIO
import hashlib
import json
import math
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(r'D:\DSH\D08_Win11_Stage1\D08_Win11_Stage1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'baseline'))
from cd.evaluate import predict
from cd.metrics import Metrics
from cd.model import ChangeDetector
from cd.utils import digest, seed_everything

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p,data):
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(p)

def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    seed_everything(17)
    common = ROOT/'D08_ThreeSeeds'
    resume = common/'continuation-2026-B'
    assert not (OUT/'protocol.json').exists(), 'Evaluation previously initialized'
    prior = read(resume/'three_seed_summary.json')
    assert prior['all_warmups_match'] and not prior['test_evaluated']
    paths = {
        17: (ROOT/'D08_PairedPilot15000/results/sl10', ROOT/'D08_LatePseudo15000/results/semi10'),
        42: (common/'42/results/sl10', common/'42/results/semi10'),
        2026: (common/'2026/results/sl10', resume/'results/semi10'),
    }
    points = []
    models = []
    for pair in prior['pairs']:
        seed = pair['seed']
        for arm, folder in zip(('A','B'),paths[seed],strict=True):
            history = read(folder/'history.json')
            result = read(folder/'result.json')
            assert result['completed'] and result['steps']==15000 and not result['test_accessed']
            assert history[-1]['step']==15000 and all(row['val']['threshold']==.5 and not row['val']['test_accessed'] for row in history)
            best = max(history,key=lambda row: row['val']['f1'])
            path = folder/'best.pt'
            sha = digest(path)
            with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype('float64'))]):
                checkpoint = torch.load(path,map_location='cpu',weights_only=True)
            expected = pair[arm]
            assert checkpoint['step']==best['step']==expected['step']
            assert checkpoint['val']==best['val'] and checkpoint['val']['f1']==expected['f1']
            config=checkpoint['config']
            assert config['seed']==seed and config['supervised_steps']==15000 and config['threshold']==.5
            if arm=='B':
                assert config['semi_warmup_steps']==9000 and config['semi_ramp_steps']==2000
                assert result['pseudo_totals'][0]==6000
            model=ChangeDetector().cuda()
            model.load_state_dict(checkpoint['model'],strict=True)
            model.eval()
            points.append(dict(seed=seed,arm=arm,step=best['step'],validation_f1=expected['f1'],path=str(path),sha256=sha))
            models.append(model)
    archive = ROOT/'sealed_test/test.zip'
    asset = read(ROOT/'assets.json')['splits']['test']
    assert digest(archive)==asset['sha256']
    assert len(points)==6
    protocol=dict(data=str(archive),data_sha256=asset['sha256'],threshold=.5,tile=256,batch_size=4,selection='best validation F1',current_protocol_prior_test_evaluation=False,earlier_research_used_same_test_set=True,checkpoints=points,target_mean_gain_pp=3,evaluator_sha256=digest(Path(__file__)))
    write(OUT/'protocol.json',protocol)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        names={part:set() for part in ('A','B','label')}
        png=[x for x in z.namelist() if x.endswith('.png')]
        assert len(png)==len(set(png))==384
        for path in png:
            part,name=PurePosixPath(path).parts
            names[part].add(name)
        assert names['A']==names['B']==names['label'] and len(names['A'])==128
        totals=[Metrics() for m in models]
        for i,name in enumerate(sorted(names['A']),1):
            def load(part,mode):
                with Image.open(BytesIO(z.read(f'{part}/{name}'))) as img:
                    assert img.size==(1024,1024)
                    return img.convert(mode)
            a,b=load('A','RGB'),load('B','RGB')
            truth=np.asarray(load('label','L'))>0
            for model,total in zip(models,totals,strict=True):
                score=predict(model,a,b,'cuda',tile=256,batch_size=4)
                assert np.isfinite(score).all()
                total.update(score>=.5,truth)
            if i%16==0:print(f'{i}/128 images evaluated for all six checkpoints',flush=True)
    for point,total in zip(points,totals,strict=True):
        v=total.result()
        assert sum(v[k] for k in ('tp','fp','fn','tn'))==128*1024**2
        assert math.isclose(v['f1'],2*v['tp']/(2*v['tp']+v['fp']+v['fn']))
        point['test']=v
    pairs=[]
    for a,b in zip(points[::2],points[1::2],strict=True):
        assert a['seed']==b['seed']
        pairs.append(dict(seed=a['seed'],A=a['test'],B=b['test'],gain_f1_pp=100*(b['test']['f1']-a['test']['f1'])))
    mean=sum(x['gain_f1_pp'] for x in pairs)/3
    write(OUT/'summary.json',dict(pairs=pairs,checkpoints=points,mean_gain_f1_pp=mean,target_met=mean>=3,images=128,threshold=.5))
    print('COMPLETE',mean,flush=True)

if __name__=='__main__':
    main()

