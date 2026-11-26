# -*- coding: utf-8 -*-
"""冻结模型的描述性错误分析；按预定规则选案例，不据测试案例改模型。"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
import torch


def read(p):return json.loads(Path(p).read_text())
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--split',choices=('test','val'),default='test')
    args=parser.parse_args();root=Path(args.run_root).resolve()
    runs={g:root/('10pct_'+g+'_seed17') for g in 'ABCD'}
    result_file=args.split+'.json'
    label='测试' if args.split=='test' else '验证'
    results={g:read(p/result_file) for g,p in runs.items() if (p/result_file).exists()}
    if args.split=='test' and any(not (runs[g]/'training_complete.json').exists() for g in results):
        raise RuntimeError('训练未完成，禁止将中途权重作为正式测试案例。')
    if not {'A','B'}<=set(results):raise RuntimeError('案例分析需真实A/B'+label+'记录。')
    configs={g:read(runs[g]/'config.json') for g in results}
    if {c['protocol_version'] for c in configs.values()}!={'strict_budget_v3'}:
        raise RuntimeError('本脚本仅分析严格预算v3，拒绝混用。')
    sys.path.insert(0,str(ROOT/'versions/strict_budget_v3'))
    from cd.model import ChangeDetector
    from cd.evaluate import predict
    from cd.metrics import Metrics
    from cd.utils import device_for,digest
    for g,c in configs.items():
        for filename,expected in c['code_sha256'].items():
            if digest(ROOT/'versions/strict_budget_v3'/filename)!=expected:
                raise RuntimeError('冻结算法指纹不符：'+filename)
    by={g:{r['name']:r for r in v['per_image']} for g,v in results.items()}
    names=sorted(set(by['A'])&set(by['B']))
    candidates=[('B组虚检像素最多',max(names,key=lambda n:(by['B'][n]['fp'],n))),
                ('B组漏检像素最多',max(names,key=lambda n:(by['B'][n]['fn'],n)))]
    f1names=[n for n in names if by['A'][n]['f1'] is not None and by['B'][n]['f1'] is not None]
    if f1names:
        delta=lambda n:by['B'][n]['f1']-by['A'][n]['f1']
        candidates += [('B相对A单图F1改善最大',max(f1names,key=lambda n:(delta(n),n))),
                       ('B相对A单图F1变化最小',min(f1names,key=lambda n:(delta(n),n)))]
    if 'D' in by:
        ns=[n for n in names if by['B'][n]['boundary_f1'] is not None and by['D'][n]['boundary_f1'] is not None]
        if ns:
            d=lambda n:by['D'][n]['boundary_f1']-by['B'][n]['boundary_f1']
            candidates += [('D相对B边界F1改善最大',max(ns,key=lambda n:(d(n),n))),
                           ('D相对B边界F1变化最小',min(ns,key=lambda n:(d(n),n)))]
    selected={}
    for reason,name in candidates:selected.setdefault(name,[]).append(reason)
    manifest=read(Path(configs['A']['manifest_dir'])/result_file)
    if any(set(v)!={r['name'] for r in manifest} for v in by.values()):
        raise RuntimeError('案例评价清单不完整或不同组清单不一致。')
    records={r['name']:r for r in manifest}
    arrays={};predictions={name:{} for name in selected}
    for name in selected:
        r=records[name]
        with Image.open(r['a']) as im:a=np.asarray(im.convert('RGB')).copy()
        with Image.open(r['b']) as im:b=np.asarray(im.convert('RGB')).copy()
        with Image.open(r['label']) as im:y=np.asarray(im.convert('L'))>0
        arrays[name]=(a,b,y)
    device=device_for(args.device);torch.set_num_threads(3)
    for g in results:
        state=torch.load(runs[g]/'best.pt',map_location='cpu',weights_only=True)
        if state['epoch']!=results[g]['selected_epoch']:raise RuntimeError('案例权重与'+label+'记录不符。')
        model=ChangeDetector().to(device).eval();model.load_state_dict(state['model'])
        for name,(a,b,y) in arrays.items():
            pred=predict(model,Image.fromarray(a),Image.fromarray(b),device,configs[g]['tile'])>=configs[g]['threshold']
            m=Metrics(configs[g]['boundary_tolerance']);m.update(pred,y);actual=m.result()
            if any(actual[k]!=by[g][name][k] for k in ('tp','fp','fn','tn')):
                raise RuntimeError('重新生成的案例与原始'+label+'记录不一致：'+g+'/'+name)
            predictions[name][g]=pred
        del model,state
        if device.type=='mps':torch.mps.empty_cache()
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ('PingFang SC','Heiti TC','Noto Sans CJK SC','SimHei') if f in available),None)
    if font is None:raise RuntimeError('缺少中文字体。')
    plt.rcParams['font.sans-serif']=[font];plt.rcParams['axes.unicode_minus']=False
    out=root/('case_analysis' if args.split=='test' else 'validation_cases');out.mkdir(exist_ok=True)
    index=[]
    for name,(a,b,y) in arrays.items():
        fig,axes=plt.subplots(3,4,figsize=(12,9.5))
        for ax in axes.flat:ax.axis('off')
        for ax,arr,title in zip(axes[0],(a,b,y),('第一时相','第二时相','变化真值')):
            ax.imshow(arr,cmap='gray',vmin=0,vmax=1,interpolation='nearest') if arr.ndim==2 else ax.imshow(arr)
            ax.set_title(title)
        axes[0,3].text(0,.75,'白色：正确检出变化\n红色：虚检\n蓝色：漏检\n黑色：正确判为未变化',fontsize=11)
        for j,g in enumerate('ABCD'):
            if g not in predictions[name]:
                axes[1,j].text(.15,.5,g+'组暂无'+label+'结果');continue
            pred=predictions[name][g]
            error=np.zeros((*y.shape,3),dtype=np.uint8)
            error[pred&y]=(255,255,255);error[pred&~y]=(255,0,0);error[~pred&y]=(0,100,255)
            axes[1,j].imshow(pred,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[1,j].set_title(g+'组预测')
            axes[2,j].imshow(error,interpolation='nearest');axes[2,j].set_title(g+'组错误分布')
        epochs='；'.join(g+'第'+str(results[g]['selected_epoch'])+'轮' for g in results)
        fig.suptitle(name+'  种子17 '+label+'集（'+epochs+'）\n'+
                     textwrap.fill('；'.join(selected[name]),width=48),fontsize=12)
        fig.tight_layout(rect=(0,0,1,.91));filename=Path(name).stem+'.png'
        fig.savefig(out/filename,dpi=150);plt.close(fig)
        index.append({'name':name,'selection_reasons':selected[name],'figure':filename,
                      'split':args.split,'selected_epochs':{g:v['selected_epoch'] for g,v in results.items()},
                      'metrics':{g:by[g][name] for g in by}})
    (out/'cases.json').write_text(json.dumps(index,ensure_ascii=False,indent=2,allow_nan=False))
    text=['# 冻结模型案例分析','','案例仅来自种子17的真实'+label+'记录，按预先写明的极值规则选择，不代表典型样本或总体平均。测试案例不用于模型选择；中途验证快照只用于诊断，不能作为主要目标证据。',
          '', '变化掩膜没有季节、阴影或光照成因标签，因此不能从虚检像素直接断言其成因。受控亮度扰动与真实季节因果结论分开报告。','']
    for row in index:
        text += ['## '+row['name'],'','选择规则：'+'；'.join(row['selection_reasons'])+'。','',
                 '![真实'+label+'案例]('+row['figure']+')','','|组别|F1（%）|边界F1（%）|虚检像素|漏检像素|','|---|---|---|---|---|']
        for g,v in row['metrics'].items():
            fmt=lambda x:'未定义' if x is None else f'{100*x:.2f}'
            text.append('|{}|{}|{}|{}|{}|'.format(g,fmt(v['f1']),fmt(v['boundary_f1']),v['fp'],v['fn']))
        text.append('')
    (out/'案例分析.md').write_text('\n'.join(text))
    print('真实案例已生成',len(index),out,flush=True)
if __name__=='__main__':main()
