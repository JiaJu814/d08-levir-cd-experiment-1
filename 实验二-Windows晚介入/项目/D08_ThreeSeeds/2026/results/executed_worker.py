"""Fixed diagnostic/training worker. Never opens test data or unlabeled masks."""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from common import ROOT, RESULTS, read, write, metric, code_fingerprints

sys.path.insert(0,str(ROOT/'baseline'))

def stopped(deadline):
    return time.time() >= deadline or (RESULTS/'STOP').exists()

def records(split,names,labels=True):
    assert split in ['train','val'], 'Test data forbidden'
    base=ROOT/'data/LEVIR-CD'/split
    return [dict(name=n,a=str(base/'A'/n),b=str(base/'B'/n),
                 **({'label':str(base/'label'/n)} if labels else {})) for n in names]

def audit():
    import numpy as np
    from PIL import Image
    expected={'train':445,'val':64}; checked={}; labeled=read(ROOT/'manifests/seed17_labeled_names.json')
    assert len(labeled)==len(set(labeled))==44
    for split,n in expected.items():
        base=ROOT/'data/LEVIR-CD'/split
        names={g:sorted(p.name for p in (base/g).glob('*.png')) for g in ['A','B','label']}
        assert names['A']==names['B']==names['label'] and len(names['A'])==n,split
        positive=total=0; value_set=set(); sizes=set(); value_counts={}; nonbinary=[]
        for name in names['A']:
            for group in ['A','B','label']:
                with Image.open(base/group/name) as im:
                    im.load(); sizes.add(im.size)
                    if group=='label':
                        mask=np.asarray(im)
                        assert mask.ndim==2, 'Expected single-channel binary mask: '+name
                        values,counts=np.unique(mask,return_counts=True)
                        vals=set(values.tolist()); value_set.update(vals)
                        assert vals <= {0,156,254,255}, (split,name,vals)
                        for value,count in zip(values,counts):value_counts[str(int(value))]=value_counts.get(str(int(value)),0)+int(count)
                        if not vals <= {0,255}:nonbinary.append(dict(name=name,values=sorted(vals)))
                        positive+=int((mask>0).sum()); total+=mask.size
            assert sizes=={(1024,1024)}, (split,name,sizes)
        checked[split]=dict(pairs=n,mask_values=sorted(value_set),positive_pixels=positive,total_pixels=total,
                            dimensions=[1024,1024],full_decode_passed=True,paired_filenames_passed=True,
                            value_counts=value_counts,non_0_255_files=nonbinary,
                            encoding='Original archives include rare156/254. Preserve files; historical loader converts L then >0.')
    assert set(labeled)<=set(names_for('train'))
    train_ids={Path(n).stem for n in names_for('train')}; val_ids={Path(n).stem for n in names_for('val')}
    # Names may have split prefixes; official archive hashes and historical split are authoritative.
    write(RESULTS/'audit.json',dict(passed=True,counts=checked,labeled_10pct=labeled,
          unlabeled_pairs=401,rounding='floor',seed=17,test_accessed=False,
          cross_split_names_overlap=sorted(train_ids & val_ids),
          note='100% labels read only for approved supervised reference/audit; semi loader exposes no U masks.'))
    write(RESULTS/'manifests/labeled10.json',records('train',labeled))
    write(RESULTS/'manifests/unlabeled10.json',records('train',[n for n in names_for('train') if n not in set(labeled)],False))
    write(RESULTS/'manifests/labeled100.json',records('train',names_for('train')))
    write(RESULTS/'manifests/val.json',records('val',names_for('val')))

def names_for(split):
    assert split in ['train','val']
    return sorted(p.name for p in (ROOT/'data/LEVIR-CD'/split/'A').glob('*.png'))

def check_metrics():
    import numpy as np
    from cd.metrics import Metrics
    cases=[(np.array([[1,1],[0,0]]),np.array([[1,0],[1,0]])),
           (np.ones((2,2)),np.ones((2,2))), (np.zeros((2,2)),np.ones((2,2))),
           (np.ones((2,2)),np.zeros((2,2))), (np.zeros((2,2)),np.zeros((2,2)))]
    results=[]
    for pred,true in cases:
        p=pred.astype(bool); t=true.astype(bool)
        counts=[int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum()),int((~p&~t).sum())]
        ref=metric(*counts); m=Metrics(); m.update(p,t); actual=m.result()
        for k,v in ref.items(): assert actual[k]==v,(k,v,actual[k])
        results.append(ref)
    write(RESULTS/'metric_checks.json',dict(passed=True,cases=results,scope='synthetic fixtures, not experimental metrics'))

def atomic_torch(value,path):
    import torch
    temp=path.with_name(path.name+'.tmp'); torch.save(value,temp); os.replace(temp,path)

def validate(model,rs,out,deadline):
    import numpy as np
    import torch
    from PIL import Image
    from cd.evaluate import predict
    model.eval(); totals=np.zeros(4,dtype=np.int64); rows=[]; begin=time.time()
    histogram=np.zeros((2,10001),dtype=np.int64)
    with torch.no_grad():
        for i,r in enumerate(rs):
            if stopped(deadline): return None
            with Image.open(r['a']) as im:a=im.convert('RGB')
            with Image.open(r['b']) as im:b=im.convert('RGB')
            with Image.open(r['label']) as im:truth=np.asarray(im.convert('L'))>0
            probability=predict(model,a,b,'cuda',256,4)
            p=probability>=0.5
            bins=np.floor(probability.astype(np.float64)*10000).astype(np.int32)
            histogram[0]+=np.bincount(bins[truth],minlength=10001)
            histogram[1]+=np.bincount(bins[~truth],minlength=10001)
            assert p.shape==truth.shape
            c=np.array([(p&truth).sum(),(p&~truth).sum(),(~p&truth).sum(),(~p&~truth).sum()],dtype=np.int64)
            totals+=c; rows.append(dict(name=r['name'],**metric(*c)))
            if i<3:
                visual=out/'visuals'; visual.mkdir(exist_ok=True)
                error=np.zeros((*truth.shape,3),dtype=np.uint8)
                error[p&truth]=(255,255,255); error[p&~truth]=(255,0,0); error[~p&truth]=(0,100,255)
                panels=[a,b,Image.fromarray(truth.astype('uint8')*255).convert('RGB'),
                        Image.fromarray(p.astype('uint8')*255).convert('RGB'),Image.fromarray(error)]
                canvas=Image.new('RGB',(1280,256))
                for j,image in enumerate(panels):canvas.paste(image.resize((256,256)),(256*j,0))
                canvas.save(visual/(r['name']+'_panels.png'))
    result=dict(**metric(*totals),images=len(rows),threshold=0.5,seconds=time.time()-begin,per_image=rows,test_accessed=False)
    result.update(precision_recall(histogram))
    write(out/'latest_validation.json',result)
    return {k:v for k,v in result.items() if k!='per_image'}

def micro(c,deadline):
    import numpy as np
    import torch
    from PIL import Image
    from cd.data import rgb_tensor
    from cd.model import ChangeDetector
    from cd.losses import supervised_loss
    from cd.utils import seed_everything
    out=RESULTS/'micro';out.mkdir(exist_ok=True);seed_everything(c['seed'])
    samples=[];used=[]
    for r in read(RESULTS/'manifests/labeled10.json'):
        with Image.open(r['label']) as im:mask=np.asarray(im.convert('L'))
        points=np.argwhere(mask>0)
        if not len(points):continue
        y,x=map(int,points[len(points)//2]); x=max(0,min(768,x-128)); y=max(0,min(768,y-128))
        images=[]
        for key in ['a','b','label']:
            with Image.open(r[key]) as im: images.append(im.convert('L' if key=='label' else 'RGB').crop((x,y,x+256,y+256)))
        samples.append((rgb_tensor(images[0]),rgb_tensor(images[1]),torch.from_numpy((np.asarray(images[2])>0).astype(np.float32).copy())[None]))
        used.append(dict(name=r['name'],x=x,y=y,scope='diagnostic fixed positive crop; not validation'))
        if len(samples)==2:break
    assert len(samples)==2,'Insufficient positive labeled crops'
    a,b,y=[torch.stack([s[i] for s in samples]).cuda() for i in range(3)]
    m=ChangeDetector().cuda();opt=torch.optim.Adam(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    history=[]; initial=None; result=None
    for step in range(1,c['micro_steps']+1):
        if stopped(deadline):break
        m.train();opt.zero_grad(set_to_none=True);loss=supervised_loss(m(a,b),y)
        assert torch.isfinite(loss);loss.backward();opt.step()
        if step==1:initial=loss.item()
        if step==1 or step%25==0:
            m.eval()
            with torch.no_grad():
                logits=m(a,b);p=logits.sigmoid()>=0.5;t=y.bool()
                result=metric(int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum()),int((~p&~t).sum()))
                evaluation_loss=supervised_loss(logits,y).item()
            row=dict(step=step,loss=loss.item(),eval_loss=evaluation_loss,**result);history.append(row);write(out/'history.json',history)
            if result['f1'] is not None and result['f1']>=c['micro_required_f1'] and evaluation_loss<=initial*c['micro_required_loss_ratio']:break
    passed=bool(result and result['f1'] is not None and result['f1']>=c['micro_required_f1'] and history[-1]['eval_loss']<=initial*c['micro_required_loss_ratio'])
    write(out/'result.json',dict(passed=passed,steps=history[-1]['step'] if history else 0,crops=used,history=history,
          interpretation='PASS is pipeline memorization sanity only; FAIL is inconclusive and blocks expansion, not proof of implementation bug.'))
    atomic_torch({'model':m.state_dict(),'config':c},out/'last.pt')
    return passed

def train(c,kind,deadline):
    import numpy as np
    import torch
    from cd.data import PairDataset
    from cd.model import ChangeDetector
    from cd.losses import supervised_loss,pseudo_loss,update_teacher
    from cd.utils import seed_everything
    out=RESULTS/kind;out.mkdir(exist_ok=True)
    fingerprint=code_fingerprints();seed_everything(c['seed'])
    semi=kind=='semi10'; percent=100 if kind=='sl100' else 10
    rs=read(RESULTS/f'manifests/labeled{percent}.json'); val=read(RESULTS/'manifests/val.json')
    from PIL import Image
    pos=total=0
    for r in rs:
        with Image.open(r['label']) as im:mask=np.asarray(im.convert('L'))>0
        pos+=int(mask.sum());total+=mask.size
    assert pos>0
    pw=1.0
    ds=PairDataset(rs,c['crop'],seed=c['seed']+101,cache_pairs=8)
    irng=random.Random(c['seed']+301);uirng=random.Random(c['seed']+501)
    uds=PairDataset(read(RESULTS/'manifests/unlabeled10.json'),c['crop'],unlabeled=True,seed=c['seed']+201,cache_pairs=4) if semi else None
    m=ChangeDetector().cuda();teacher=copy.deepcopy(m).eval() if semi else None
    if teacher:
        for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    start=0;best=-1.;history=[];seconds_prior=0.;pseudo_totals=[0,0,0,0,0]
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert saved['code_sha256']==fingerprint and saved['config']==c,'Resume code/config mismatch'
        m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        if teacher:teacher.load_state_dict(saved['teacher'])
        start=saved['step'];best=saved['best'];history=saved['history'];seconds_prior=saved['seconds'];pseudo_totals=saved['pseudo_totals']
        random.setstate(saved['rng_python']); np.random.set_state(saved['rng_numpy']); torch.set_rng_state(saved['rng_torch'])
        torch.cuda.set_rng_state_all(saved['rng_cuda']);ds.rng.setstate(saved['rng_dataset']);irng.setstate(saved['rng_indices'])
        if uds: uds.rng.setstate(saved['rng_udataset']); uirng.setstate(saved['rng_uindices'])
    runconfig=dict(config=c,kind=kind,positive_weight=pw,labeled_pairs=len(rs),unlabeled_pairs=len(uds) if uds else 0,
        sampling='uniform replacement, random 256 crops',optimizer='Adam',schedule='cosine fixed 15000 updates, min lr 1% initial',
        initial_seed=c['seed'],code_sha256=fingerprint,hidden_u_masks_exposed=False,
        augmentation='historical shared geometry; per-date color jitter',
        unlabeled_rng='fork_rng cuda, seeded by update; restored before next supervised forward',
        normalization='historical BatchNorm: U student forward updates its buffers; teacher EMA includes buffers',
        evaluation='student, all64 validation pairs, nonoverlap256 tiles, threshold0.5, global confusion',
        project_supplements='weighted BCE+Dice, pseudo hard confidence BCE; not exact MeanTeacher MSE reproduction')
    write(out/'config.json',runconfig)
    begin=time.time();last_step=start; interval_losses=[]; last_val=None
    def save(step):
        atomic_torch(dict(model=m.state_dict(),teacher=teacher.state_dict() if teacher else None,optimizer=opt.state_dict(),
            step=step,best=best,history=history,pseudo_totals=pseudo_totals,seconds=time.time()-begin+seconds_prior,config=c,code_sha256=fingerprint,
            rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            rng_dataset=ds.rng.getstate(),rng_indices=irng.getstate(),rng_udataset=uds.rng.getstate() if uds else None,
            rng_uindices=uirng.getstate()),out/'last.pt')
    for step in range(start+1,c['supervised_steps']+1):
        if stopped(deadline):break
        m.train();lr=c['lr']*(0.01+0.99*(1+math.cos(math.pi*(step-1)/c['supervised_steps']))/2)
        for group in opt.param_groups:group['lr']=lr
        batch=[ds[irng.randrange(len(ds))] for _ in range(c['batch_size'])]
        a,b,y=[torch.stack([s[i] for s in batch]).cuda() for i in range(3)]
        opt.zero_grad(set_to_none=True);sl=supervised_loss(m(a,b),y,positive_weight=pw)
        ul=sl*0; stats=dict(coverage=0.,positive_coverage=0.)
        ramp=min(1.,max(0.,(step-c['semi_warmup_steps'])/c['semi_ramp_steps'])) if semi else 0.
        if semi and ramp>0:
            ubatch=[uds[uirng.randrange(len(uds))] for _ in range(c['batch_size'])]
            aw,bw,ast,bst=[torch.stack([s[i] for s in ubatch]).cuda() for i in range(4)]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(c['seed']+100000+step)
                with torch.no_grad():q=teacher(aw,bw).sigmoid()
                ul,stats=pseudo_loss(m(ast,bst),q,high=c['pseudo_high'],low=c['pseudo_low'],class_balance=c['pseudo_class_balance'])
            collect_pseudo(q,c,pseudo_totals)
        loss=sl+c['semi_weight']*ramp*ul
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; stop branch, do not change loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),c['grad_clip']);opt.step()
        if teacher:update_teacher(teacher,m,c['ema_decay'])
        last_step=step;interval_losses.append(loss.item())
        if step% c['checkpoint_every']==0:save(step)
        if step==1 or step%c['validate_every']==0 or step==c['supervised_steps']:
            # Save BEFORE validation, so a timeout during validation loses no completed updates.
            save(step)
            result=validate(m,val,out,deadline)
            if result is None:break
            last_val=result
            row=dict(step=step,seconds=time.time()-begin+seconds_prior,lr=lr,loss=float(np.mean(interval_losses)),
                     supervised_loss=sl.item(),unsupervised_loss=ul.item(),ramp=ramp,**stats,pseudo_totals=list(pseudo_totals),val=result)
            history.append(row);interval_losses.clear();write(out/'history.json',history)
            if result['f1'] is not None and result['f1']>best:
                best=result['f1'];atomic_torch(dict(model=m.state_dict(),step=step,val=result,config=c),out/'best.pt')
            save(step); print(kind,step,'val_f1',result['f1'],flush=True)
            write(out/'runtime.json',dict(step=step,seconds=row['seconds'],best_val_f1=best,deadline=deadline))
    save(last_step)
    completed=last_step==c['supervised_steps'] and bool(history) and history[-1]['step']==last_step
    write(out/'result.json',dict(completed=completed,steps=last_step,supervised_samples=last_step*c['batch_size'],
        best_val_f1=best if best>=0 else None,last_val=last_val,seconds=time.time()-begin+seconds_prior,
        unique_labeled_pairs=len(rs),pseudo_totals=list(pseudo_totals),test_accessed=False,formal_result=False,
        note='Fixed-step diagnostic, not proof of convergence; supervised and semi training costs differ.'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['audit','micro','sl10','sl100','semi10'])
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    c=read(ROOT/'config.json')
    import torch
    torch.set_num_threads(4)
    if args.kind=='audit':audit();check_metrics();return
    if args.kind=='micro':
        if not micro(c,args.deadline):raise RuntimeError('Micro sanity gate not passed; stop expansion')
    else:train(c,args.kind,args.deadline)

if __name__=='__main__':main()
