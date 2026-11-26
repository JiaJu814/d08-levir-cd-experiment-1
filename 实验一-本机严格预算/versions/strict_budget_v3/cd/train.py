"""严格标注预算训练协议v3：类别平衡、独立随机流、稀疏验证、原子断点。"""
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from .data import PairDataset
from .evaluate import evaluate
from .losses import supervised_loss, pseudo_loss, update_teacher
from .model import ChangeDetector
from .utils import seed_everything, device_for, read_json, write_json, digest


def atomic_save(value, path):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def train(config):
    c=dict(config);resume=c.pop('resume',False)
    seed_everything(c['seed']);torch.set_num_threads(c.get('cpu_threads',3))
    device=device_for(c.get('device','auto'))
    manifest=Path(c['manifest_dir']);audit=read_json(manifest/'audit.json')
    if not audit.get('standard_levir_counts') and not c.get('allow_fixture_data',False):
        raise ValueError('非完整LEVIR-CD；构造检查须显式allow_fixture_data。')
    folder=manifest/str(c['seed'])/str(c['percent'])
    labeled=read_json(folder/'labeled.json')
    expected_count = audit['counts']['train'] * c['percent'] // 100
    if audit.get('rounding') != 'floor' or len(labeled) != expected_count:
        raise ValueError('严格标注预算不符：必须向下取整且样本数精确匹配。')
    c['actual_labeled_fraction'] = len(labeled) / audit['counts']['train']
    if c.get('positive_weight','auto')=='auto':
        positive=total=0
        for r in labeled:
            with Image.open(r['label']) as im:mask=np.asarray(im.convert('L'))>0
            positive+=int(mask.sum());total+=mask.size
        if not positive:raise ValueError('有标签子集无变化像素。')
        c['positive_weight']=min(20.0,(total-positive)/positive)
        c['labeled_foreground_fraction']=positive/total
    is_semi=c['group'] in ('B','D')
    boundary=c['boundary_weight'] if c['group'] in ('C','D') else 0.0
    if c['percent']==100 and c['group']!='A':raise ValueError('100%仅运行A。')
    out=Path(c['run_root'])/f"{c['percent']}pct_{c['group']}_seed{c['seed']}"
    if resume:
        if not (out/'last.pt').exists() or (out/'training_complete.json').exists():raise ValueError('不能恢复已完成或无断点运行。')
    else:out.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[1]
    sources=sorted((source_root/'cd').glob('*.py'))+[source_root/'main.py']
    c.update(device_actual=str(device),labeled_pairs=len(labeled),manifest_sha256=digest(manifest/'audit.json'),
             code_sha256={str(p.relative_to(source_root)):digest(p) for p in sources})
    if resume:
        if read_json(out/'config.json')!=c:raise ValueError('恢复配置、代码或数据指纹不一致。')
    else:write_json(out/'config.json',c)
    model=ChangeDetector().to(device);teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['epochs'],eta_min=c['lr']*.01)
    ds=PairDataset(labeled,c['crop'],seed=c['seed']+101,cache_pairs=c.get('labeled_cache_pairs',64))
    sampler_rng=torch.Generator().manual_seed(c['seed']+301)
    loader_rng=torch.Generator().manual_seed(c['seed']+401)
    loader=DataLoader(ds,batch_size=c['batch_size'],sampler=RandomSampler(ds,replacement=True,num_samples=c['steps_per_epoch']*c['batch_size'],generator=sampler_rng),generator=loader_rng,num_workers=0)
    uds=uloader=None
    u_sampler_rng=torch.Generator().manual_seed(c['seed']+501)
    u_loader_rng=torch.Generator().manual_seed(c['seed']+601)
    if is_semi:
        uds=PairDataset(read_json(folder/'unlabeled.json'),c['crop'],unlabeled=True,seed=c['seed']+201)
        uloader=DataLoader(uds,batch_size=c['batch_size'],sampler=RandomSampler(uds,replacement=True,num_samples=c['steps_per_epoch']*c['batch_size'],generator=u_sampler_rng),generator=u_loader_rng,num_workers=0)
    val=read_json(manifest/'val.json')
    best=-1.0;start_epoch=0;elapsed=0.;history=[]
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);teacher.load_state_dict(saved['teacher'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
        for generator,key in [(sampler_rng,'rng_sampler'),(loader_rng,'rng_loader'),(u_sampler_rng,'rng_u_sampler'),(u_loader_rng,'rng_u_loader')]:generator.set_state(saved[key])
        ds.rng.setstate(saved['rng_dataset'])
        if uds:uds.rng.setstate(saved['rng_u_dataset'])
        if device.type=='mps':torch.mps.set_rng_state(saved['rng_mps'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['rng_cuda'])
        start_epoch=saved['epoch'];best=saved['best_val_f1'];elapsed=saved['elapsed_seconds'];history=saved['history']
        (out/'history.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in history))
    start=time.time()-elapsed
    print('Training',out,'device',device,'positive_weight',c['positive_weight'],flush=True)
    for epoch in range(start_epoch,c['epochs']):
        model.train();use_u=is_semi and epoch>=c['warmup_epochs'];us=iter(uloader) if use_u else None
        totals={k:0. for k in ('loss','supervised_loss','unsupervised_loss','coverage','positive_coverage')};diagnostic={}
        ramp=min(1.,max(0.,(epoch-c['warmup_epochs']+1)/max(1,c['ramp_epochs'])))
        for index,(a,b,y) in enumerate(loader):
            a,b,y=a.to(device),b.to(device),y.to(device)
            if is_semi and index==0:
                with torch.no_grad():
                    check=teacher(a,b).sigmoid();positive=check>=c['pseudo_high'];negative=check<=c['pseudo_low'];n=positive.sum().item()
                    diagnostic={'teacher_labeled_positive_precision':(positive*y).sum().item()/n if n else None,'teacher_labeled_positive_pixels':int(n),
                                'teacher_labeled_confident_coverage':(positive|negative).float().mean().item(),'diagnostic_scope':'first_labeled_strong_augmented_batch_only'}
            optimizer.zero_grad(set_to_none=True)
            logits=model(a,b);sl=supervised_loss(logits,y,boundary,c['boundary_radius'],c['positive_weight'])
            ul=logits.sum()*0;stats={'coverage':0.,'positive_coverage':0.}
            if use_u:
                aw,bw,ast,bst=[x.to(device) for x in next(us)]
                with torch.no_grad():probability=teacher(aw,bw).sigmoid()
                ul,stats=pseudo_loss(model(ast,bst),probability,c['pseudo_high'],c['pseudo_low'],boundary,c['boundary_radius'],c.get('pseudo_class_balance',False))
            loss=sl+c['unsupervised_weight']*ramp*ul
            if not torch.isfinite(loss):raise FloatingPointError('非有限训练损失。')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip']);optimizer.step()
            update_teacher(teacher,model,c['ema_decay'])
            for key,value in dict(loss=loss.item(),supervised_loss=sl.item(),unsupervised_loss=ul.item(),**stats).items():totals[key]+=value/c['steps_per_epoch']
        scheduler.step()
        validation_due=epoch==0 or (epoch+1)%c.get('validate_every',1)==0 or epoch+1==c['epochs']
        result=evaluate(model,val,device,c['tile'],c['threshold'],c['boundary_tolerance']) if validation_due else None
        if result is not None:
            if result['f1'] is None:raise ValueError('验证F1未定义。')
            if result['f1']>best:
                best=result['f1'];atomic_save({'model':model.state_dict(),'epoch':epoch+1,'config':c,'val':result},out/'best.pt')
        row=dict(epoch=epoch+1,seconds=time.time()-start,lr=optimizer.param_groups[0]['lr'],**totals,pseudo_diagnostic=diagnostic,
                 val={k:v for k,v in result.items() if k!='per_image'} if result else None)
        history.append(row)
        atomic_save({'model':model.state_dict(),'teacher':teacher.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'epoch':epoch+1,'best_val_f1':best,'elapsed_seconds':time.time()-start,'history':history,'config':c,
                     'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
                     'rng_mps':torch.mps.get_rng_state() if device.type=='mps' else None,'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                     'rng_sampler':sampler_rng.get_state(),'rng_loader':loader_rng.get_state(),'rng_u_sampler':u_sampler_rng.get_state(),'rng_u_loader':u_loader_rng.get_state(),
                     'rng_dataset':ds.rng.getstate(),'rng_u_dataset':uds.rng.getstate() if uds else None},out/'last.pt')
        with open(out/'history.jsonl','a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        f1=f"{result['f1']:.4f}" if result else 'not_scheduled'
        print(f"epoch {epoch+1}/{c['epochs']} loss={totals['loss']:.4f} val_f1={f1} coverage={totals['coverage']:.3f} positive_coverage={totals['positive_coverage']:.4f}",flush=True)
    write_json(out/'training_complete.json',{'best_val_f1':best,'seconds':time.time()-start,'epochs':c['epochs'],'test_evaluated':False})
    return out
