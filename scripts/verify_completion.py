# -*- coding: utf-8 -*-
"""独立读取证据验收；启动进程、单种子及验证集分数均不算完成。"""
import argparse
import hashlib
import importlib.util
import json
import math
import random
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={(p,g,s) for p in (5,10,20) for g in 'ABCD' for s in (17,42,2026)}
EXPECTED.update((100,'A',s) for s in (17,42,2026))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_source_data(manifest):
    audit=read(manifest/'audit.json');root=Path(audit['root'])
    verified=0
    for item,hashes in audit['fingerprints'].items():
        split,name=item.split('/',1)
        for key,folder in (('a','A'),('b','B'),('label','label')):
            if sha(root/split/folder/name)!=hashes[key]:
                raise ValueError('原始影像或标注的SHA256已变化：'+item+'/'+folder)
            verified+=1
    names=sorted(k.split('/',1)[1] for k in audit['fingerprints'] if k.startswith('train/'))
    for seed in (17,42,2026):
        order=names[:];random.Random(seed).shuffle(order)
        for percent in (5,10,20,100):
            folder=manifest/str(seed)/str(percent);n=len(order)*percent//100
            labeled=read(folder/'labeled.json');unlabeled=read(folder/'unlabeled.json')
            if [v['name'] for v in labeled]!=order[:n] or [v['name'] for v in unlabeled]!=order[n:]:
                raise ValueError('标注清单不符合固定种子的嵌套抽样。')
            if any('label' in v for v in unlabeled):raise ValueError('无标签清单泄漏标签路径。')
    return verified

def check_metrics(r,names):
    if r['images']!=128 or {v['name'] for v in r['per_image']}!=names or len(r['per_image'])!=128:
        raise ValueError('测试样本清单不完整或重复。')
    for k in ('tp','fp','fn','tn'):
        if r[k]!=sum(v[k] for v in r['per_image']):raise ValueError('逐图混淆矩阵与集合汇总不符。')
    tp,fp,fn,tn=(r[k] for k in ('tp','fp','fn','tn'))
    values={'f1':2*tp/(2*tp+fp+fn),'iou':tp/(tp+fp+fn),
            'false_positive_rate':fp/(fp+tn),'false_negative_rate':fn/(fn+tp)}
    for k,v in values.items():
        if not math.isclose(r[k],v,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('混淆矩阵与指标公式不符：'+k)

def check_checkpoint_states(c,history,done,best,last):
    """Read actual saved training state, not merely a completion marker."""
    if (c['epochs'],c['steps_per_epoch'],c['batch_size'])!=(100,100,4):
        raise ValueError('冻结训练预算不是100轮、每轮100步、batch4。')
    if best['config']!=c or last['config']!=c:
        raise ValueError('实际断点配置与运行配置不一致。')
    if last['epoch']!=100 or last['history']!=history:
        raise ValueError('最终断点未完成100轮或与逐轮日志不一致。')
    if [r['epoch'] for r in history]!=list(range(1,101)):
        raise ValueError('训练历史不连续。')
    chosen=max((r for r in history if r['val'] is not None),
               key=lambda r:r['val']['f1'])
    if (best['epoch']!=chosen['epoch'] or
            {k:v for k,v in best['val'].items() if k!='per_image'}!=chosen['val']):
        raise ValueError('最佳断点不是完整验证记录选出的最优权重。')
    if (done['epochs']!=100 or done['best_val_f1']!=chosen['val']['f1'] or
            last['best_val_f1']!=chosen['val']['f1']):
        raise ValueError('完成标记、最终断点与最优验证分数不一致。')
    states=last['optimizer']['state']
    expected_parameters=sum(k.endswith(('.weight','.bias')) for k in last['model'])
    if len(states)!=expected_parameters or not states or any(
            float(s['step'])!=10000 for s in states.values()):
        raise ValueError('实际Adam参数状态未全部完成10000次优化。')
    if last['scheduler']['last_epoch']!=100:
        raise ValueError('学习率调度器没有完成100轮。')
    if set(best['model'])!=set(last['model']) or set(last['teacher'])!=set(last['model']):
        raise ValueError('模型或教师断点缺少参数。')
    for name,value in best['model'].items():
        if value.shape!=last['model'][name].shape:
            raise ValueError('最佳与最终模型参数形状不符。')
    images=best['val']['per_image']
    if best['val']['images']!=64 or len(images)!=64 or len({r['name'] for r in images})!=64:
        raise ValueError('最佳断点不是64幅完整验证。')
    for key in ('tp','fp','fn','tn'):
        if sum(r[key] for r in images)!=best['val'][key]:
            raise ValueError('最佳断点逐图与全局验证计数不一致。')
    return {'optimizer_steps':10000,'optimizer_parameter_states':len(states),
            'epochs':100,'selected_epoch':best['epoch'],
            'checkpoint_history_matches':True,'checkpoint_config_matches':True}

def check_evaluation_settings(r,c,folder,brightness):
    if (r['threshold']!=c['threshold'] or
            r['boundary_tolerance_px']!=c['boundary_tolerance'] or
            r['brightness_factor_t2']!=brightness or
            Path(r['checkpoint']).resolve()!=(folder/'best.pt').resolve()):
        raise ValueError('评价阈值、边界容差、亮度或权重路径不符。')
    for item in [r]+r['per_image']:
        tp,fp,fn,tn=(item[k] for k in ('tp','fp','fn','tn'))
        if any(type(n) is not int or n<0 for n in (tp,fp,fn,tn)):
            raise ValueError('混淆矩阵包含负数或非整数。')
        pixels=128*1024*1024 if item is r else 1024*1024
        if tp+fp+fn+tn!=pixels:raise ValueError('评价像素数不符。')
        fractions={'precision':(tp,tp+fp),'recall':(tp,tp+fn),
                   'f1':(2*tp,2*tp+fp+fn),'iou':(tp,tp+fp+fn),
                   'false_positive_rate':(fp,fp+tn),'false_negative_rate':(fn,fn+tp)}
        for key,(num,den) in fractions.items():
            expected=num/den if den else None
            if expected is None:
                if item[key] is not None:raise ValueError('零分母指标应未定义：'+key)
            elif item[key] is None or not math.isclose(item[key],expected,abs_tol=1e-12,rel_tol=1e-12):
                raise ValueError('逐图或全局指标不符合混淆矩阵：'+key)
        bp,br,bf=(item[k] for k in ('boundary_precision','boundary_recall','boundary_f1'))
        if any(v is not None and (not math.isfinite(v) or not 0<=v<=1) for v in (bp,br,bf)):
            raise ValueError('边界指标超出定义域。')
        if bp is not None and br is not None:
            expected=2*bp*br/(bp+br) if bp+br else 0.
            if bf is None or not math.isclose(bf,expected,abs_tol=1e-12,rel_tol=1e-12):
                raise ValueError('边界F1与精确率、召回率不一致。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',default=str(ROOT/'runs_strict_v3'))
    ap.add_argument('--allow-incomplete',action='store_true');args=ap.parse_args();root=Path(args.run_root).resolve()
    spec=importlib.util.spec_from_file_location('report',ROOT/'scripts/research_report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    missing=[];errors=[];valid=[];by={};artifacts={};source_count=None
    checkpoint_audits={};configs={}
    for p,g,s in sorted(EXPECTED):
        folder=root/f'{p}pct_{g}_seed{s}'
        required=['config.json','history.jsonl','best.pt','last.pt','training_complete.json','test.json']
        if p==10:required += ['test_brightness0.8.json','test_brightness1.2.json']
        absent=[f for f in required if not (folder/f).exists()]
        if absent:missing.append({'run':folder.name,'missing':absent});continue
        try:
            c=read(folder/'config.json');done=read(folder/'training_complete.json');test=read(folder/'test.json')
            if (c['percent'],c['group'],c['seed'])!=(p,g,s):raise ValueError('目录与配置组合不符。')
            if c['protocol_version']!='strict_budget_v3' or not report.verify_annotation_budget(c):
                raise ValueError('严格标注预算不符。')
            for f,h in c['code_sha256'].items():
                if sha(ROOT/'versions/strict_budget_v3'/f)!=h:raise ValueError('冻结算法代码变化：'+f)
            history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
            if [r['epoch'] for r in history]!=list(range(1,101)) or done['epochs']!=100 or c['epochs']!=100:
                raise ValueError('不是100轮完整连续训练。')
            selected=max((r for r in history if r['val'] is not None),key=lambda r:r['val']['f1'])
            if selected['epoch']!=test['selected_epoch'] or selected['val']['f1']!=done['best_val_f1']:
                raise ValueError('测试权重不是验证F1最优轮次。')
            import torch
            best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            checkpoint_audits[folder.name]=check_checkpoint_states(c,history,done,best,last)
            val_names={r['name'] for r in read(Path(c['manifest_dir'])/'val.json')}
            if {r['name'] for r in best['val']['per_image']}!=val_names:
                raise ValueError('最佳断点验证影像清单不符。')
            del best,last
            names={r['name'] for r in read(Path(c['manifest_dir'])/'test.json')}
            check_metrics(test,names)
            check_evaluation_settings(test,c,folder,1.0)
            for f in required:
                if not f.startswith('test_brightness'):continue
                perturb=read(folder/f);check_metrics(perturb,names)
                check_evaluation_settings(perturb,c,folder,float(f[15:-5]))
                if perturb['selected_epoch']!=test['selected_epoch']:
                    raise ValueError('亮度实验更换了模型权重。')
                original={v['name']:v for v in test['per_image']}
                if any((a['tp']+a['fn'],a['fp']+a['tn'])!=(original[a['name']]['tp']+original[a['name']]['fn'],original[a['name']]['fp']+original[a['name']]['tn']) for a in perturb['per_image']):
                    raise ValueError('亮度实验真值变化。')
            by[p,g,s]=test;configs[p,g,s]=c;valid.append(folder.name)
        except (ValueError,KeyError,OSError,ZeroDivisionError,RuntimeError,TypeError) as exc:
            errors.append({'run':folder.name,'error':str(exc)})
    for p in (5,10,20):
        for s in (17,42,2026):
            if (p,'A',s) not in configs:continue
            a=configs[p,'A',s]
            for g in 'BCD':
                if (p,g,s) not in configs:continue
                c=configs[p,g,s]
                if set(a)!=set(c) or any(a[k]!=c[k] for k in a if k!='group'):
                    errors.append({'run':f'{p}pct_{g}_seed{s}',
                                   'error':'与同种子同预算A的配置存在group以外差异。'})
    assessment=report.assess_target(by)
    final_files=['goal_assessment.json','研究结果报告.md','summary.csv','summary.json',
                 'figures/f1.png','figures/iou.png','figures/boundary_f1.png',
                 'figures/f1.pdf','figures/iou.pdf','figures/boundary_f1.pdf',
                 'figures/training_curves.png','figures/training_curves.pdf',
                 'case_analysis/cases.json','case_analysis/案例分析.md']
    missing_artifacts=[f for f in final_files if not (root/f).exists()]
    if not missing and not errors and not missing_artifacts:
        try:source_count=verify_source_data(Path(read(root/valid[0]/'config.json')['manifest_dir']))
        except (ValueError,KeyError,OSError) as exc:errors.append({'run':'source_data','error':str(exc)})
        recorded=read(root/'goal_assessment.json')
        if recorded.get('target_met')!=assessment['target_met'] or not recorded.get('annotation_budget_valid') or not recorded.get('full_matrix_complete'):
            errors.append({'run':'report','error':'报告目标判定或完整性与独立验收不符。'})
        for folder in (root/name for name in valid):
            for f in ('best.pt','config.json','test.json','training_complete.json'):
                file=folder/f;artifacts[str(file.relative_to(root))]=sha(file)
        for f in final_files:artifacts[f]=sha(root/f)
    complete=(len(valid)==39 and not errors and not missing_artifacts and assessment['target_met'] is True)
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'goal_complete':complete,
            'verified_source_files':source_count,'verified_runs':len(valid),'expected_runs':39,'paired_f1_assessment':assessment,
            'pending_runs':missing,'errors':errors,'missing_artifacts':missing_artifacts,
            'checkpoint_audits':checkpoint_audits,
            'metric_audit_scope':'Pixel counts/formulas and BF1 harmonic identity; boundary edge counts are not stored by frozen evaluator and are not independently reconstructed here.',
            'artifact_sha256':artifacts,'note':'仅依据实际完整训练与测试文件；未达3个百分点时目标仍未完成。'}
    root.mkdir(parents=True,exist_ok=True)
    (root/'completion_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('pending_runs','artifact_sha256')},ensure_ascii=False))
    if not complete and not args.allow_incomplete:raise SystemExit(1)
if __name__=='__main__':main()
