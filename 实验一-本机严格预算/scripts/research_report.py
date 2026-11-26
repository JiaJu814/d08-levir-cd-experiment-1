"""基于真实训练与测试记录生成目标判定；缺少3个配对种子时不判定达标。"""
import argparse
import json
import statistics
import math
import hashlib
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t

KEYS=['f1','iou','precision','recall','false_positive_rate','false_negative_rate','boundary_f1']
def read(p):return json.loads(p.read_text())
def stats(v):return {'mean':statistics.mean(v),'std_sample':statistics.stdev(v) if len(v)>1 else None}
def fmt(x):return '未定义' if x is None else f'{100*x:.2f}'
def assess_target(by):
    pairs=[(s,by[10,'B',s]['f1']-by[10,'A',s]['f1']) for s in (17,42,2026) if (10,'B',s) in by and (10,'A',s) in by]
    deltas=[100*d for _,d in pairs];target=None;ci=None
    if len(deltas)==3:
        target=statistics.mean(deltas)>3 or math.isclose(statistics.mean(deltas),3,rel_tol=0,abs_tol=1e-9)
        se=statistics.stdev(deltas)/len(deltas)**.5
        ci=list(map(float,t.interval(.95,len(deltas)-1,loc=statistics.mean(deltas),scale=se))) if se else [statistics.mean(deltas)]*2
    assessment={'target':'10%标注 B-A 测试F1均值提升至少3个百分点，3个配对种子','paired_seeds':[s for s,_ in pairs],'delta_f1_percentage_points':deltas,'mean_delta_percentage_points':statistics.mean(deltas) if deltas else None,'std_delta_percentage_points':statistics.stdev(deltas) if len(deltas)>1 else None,'paired_mean_95pct_t_interval':ci,'target_met':target,'test_used_for_tuning':False,'scope':'Paired F1 criterion; main report also verifies annotation budget.'}
    return assessment

def verify_annotation_budget(config):
    manifest=Path(config['manifest_dir'])
    audit=read(manifest/'audit.json')
    expected=audit['counts']['train']*config['percent']//100
    selected=read(manifest/str(config['seed'])/str(config['percent'])/'labeled.json')
    return (hashlib.sha256((manifest/'audit.json').read_bytes()).hexdigest()==config.get('manifest_sha256')
            and audit['counts']=={'train':445,'val':64,'test':128}
            and audit.get('rounding')=='floor' and config.get('budget_rounding')=='floor'
            and config['labeled_pairs']==len(selected)==expected
            and config.get('actual_labeled_fraction')==len(selected)/audit['counts']['train'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True);args=ap.parse_args();root=Path(args.run_root)
    rows=[];protocols=set();budgets=[];versions=set()
    uniform_fields=('epochs','steps_per_epoch','batch_size','lr','weight_decay','warmup_epochs','ramp_epochs','ema_decay','pseudo_high','pseudo_low','unsupervised_weight','boundary_weight','boundary_radius','boundary_tolerance','threshold','crop','tile','validate_every','protocol_version','budget_rounding')
    for file in sorted(root.glob('*/test.json')):
        c=read(file.parent/'config.json');r=read(file);training=read(file.parent/'training_complete.json')
        budgets.append(verify_annotation_budget(c));versions.add(c.get('protocol_version'))
        signature={'code':c['code_sha256'],'data':c['manifest_sha256'],'shared_config':{k:c.get(k) for k in uniform_fields}}
        protocols.add(json.dumps(signature,sort_keys=True))
        state={'percent':c['percent'],'group':c['group'],'seed':c['seed'],'labeled_pairs':c['labeled_pairs'],'val_f1':training['best_val_f1'],'seconds':training['seconds'],'path':str(file.parent),'selected_epoch':r['selected_epoch']}
        state.update({k:r[k] for k in KEYS});rows.append(state)
    if not rows:raise RuntimeError('没有真实测试结果。')
    if len(protocols)!=1:raise RuntimeError('检测到代码、数据或共享配置混用，拒绝生成达标结论。')
    by={(r['percent'],r['group'],r['seed']):r for r in rows}
    expected={(p,g,s) for p in (5,10,20) for g in 'ABCD' for s in (17,42,2026)}
    expected.update((100,'A',s) for s in (17,42,2026))
    if len(by)!=len(rows) or not set(by)<=expected:
        raise RuntimeError('出现重复或协议以外的实验组合。')
    assessment=assess_target(by)
    assessment['full_matrix_complete']=set(by)==expected
    assessment['completed_test_runs']=len(rows)
    assessment['expected_test_runs']=len(expected)
    assessment['f1_gain_criterion_met']=assessment['target_met']
    assessment['annotation_budget_valid']=all(budgets)
    assessment['scope']='Frozen '+str(next(iter(versions)))+' only; exploratory runs excluded.'
    if not all(budgets):assessment['target_met']=False
    deltas=assessment['delta_f1_percentage_points'];target=assessment['target_met'];ci=assessment['paired_mean_95pct_t_interval']
    (root/'goal_assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2,allow_nan=False))
    report_stage='完整矩阵报告' if set(by)==expected else '阶段性报告 其余实验仍在运行'
    text=['# 遥感变化检测实验结果', '', report_stage,'', '以下指标来自完整真实测试记录。各模型均按验证集F1选取权重，初始未平衡试验及固定图块诊断不纳入比较。','',f'主要目标判定：{("达到" if target else "未达到") if target is not None else "配对种子不足，暂不判定"}。','']
    text += [f'严格标注预算核验：{"通过" if all(budgets) else "不通过，不能作为达标证据"}。','']
    if deltas:text += [f'B−A配对F1提升（百分点）：{deltas}；均值{statistics.mean(deltas):.3f}。该判定依据均值，不等同于统计显著性；95%配对t区间为{ci}。','']
    text += ['|标注比例|组别|种子数|F1均值±样本标准差（%）|IoU（%）|边界F1（%）|虚检率（%）|漏检率（%）|平均日志累计小时|','|---|---|---|---|---|---|---|---|---|']
    grouped={}
    for r in rows:grouped.setdefault((r['percent'],r['group']),[]).append(r)
    for (percent,group),rs in sorted(grouped.items()):
        v=[r['f1'] for r in rs];sd=statistics.stdev(v) if len(v)>1 else None
        def mean(k):
            vals=[r[k] for r in rs if r[k] is not None];return statistics.mean(vals) if vals else None
        text.append(f'|{percent}%|{group}|{len(rs)}|{fmt(statistics.mean(v))}±{fmt(sd)}|{fmt(mean("iou"))}|{fmt(mean("boundary_f1"))}|{fmt(mean("false_positive_rate"))}|{fmt(mean("false_negative_rate"))}|{mean("seconds")/3600:.2f}|')
    text += ['',
        'A为纯监督，B为伪标签一致性，C为边界监督，D为两者组合。'
        '所有组采用相同的类别平衡监督损失、验证频率、优化步数及每步有标签batch。'
        'B/D额外执行无标签教师前向和学生训练分支，这一设计不匹配FLOPs或独占设备耗时。'
        '100%标注结果是相同优化步数下的经验参考，不是数学上界。', '',
        '表中小时数来自断点保存的累计计时，包含验证与I/O。'
        '停机时间及中断前未保存部分的计算不保证计入，且各任务共享本机资源；'
        '它是运行记录，不是完整项目成本或独占设备性能基准。', '',
        '## 有限标注预算下的策略', '']
    for percent in (5,10,20):
        choices=[(statistics.mean(r['val_f1'] for r in rs),g,rs) for (p,g),rs in grouped.items() if p==percent and len(rs)==3]
        if choices:
            _,g,rs=max(choices,key=lambda x:x[0]);m=statistics.mean(r['f1'] for r in rs)
            text.append(f'- {percent}%标注：在当前已完成组别中按三种子验证F1均值选择{g}组，其测试F1均值为{100*m:.2f}%。实际有标签原图对数为{rs[0]["labeled_pairs"]}。')
    text += ['','策略选择依据验证集，不能将测试集上偶然最高的模型直接称为普遍最优。','', '## 边界精化与受控外观扰动','']
    for seed in (17,42,2026):
        if (10,'D',seed) in by and (10,'B',seed) in by:
            d,b=by[10,'D',seed],by[10,'B',seed]
            if d['boundary_f1'] is not None and b['boundary_f1'] is not None:text.append(f'- 种子{seed}，D−B边界F1变化{100*(d["boundary_f1"]-b["boundary_f1"]):.3f}个百分点；区域F1变化{100*(d["f1"]-b["f1"]):.3f}个百分点。')
    text += ['','|组别|第二时相亮度因子|配对种子数|相对原图F1变化（百分点）|虚检率变化（百分点）|','|---|---|---|---|---|']
    for g in 'ABCD':
        for factor in (.8,1.2):
            differences=[];false_positives=[]
            for seed in (17,42,2026):
                if (10,g,seed) not in by:continue
                r=by[10,g,seed];file=Path(r['path'])/('test_brightness'+str(factor)+'.json')
                if file.exists():
                    perturbed=read(file);differences.append(100*(perturbed['f1']-r['f1']));false_positives.append(100*(perturbed['false_positive_rate']-r['false_positive_rate']))
            if differences:text.append(f'|{g}|{factor}|{len(differences)}|{statistics.mean(differences):.3f}|{statistics.mean(false_positives):.3f}|')
    text += ['','亮度扰动实验只检验对指定外观变化的敏感性，不是对真实季节变化的因果证明。真实阴影、季节及边界失败案例仍应结合保存的逐图指标与可视化人工检查。','', '## 数据与复现说明','', '使用原始445/64/128划分；有标签子集按原图对、向下取整、嵌套抽样（5%为22对，10%为44对，20%为89对）。原始标签中156和254的少量像素按非零变化规则读取，源文件不改写。各run保存配置、代码和数据指纹、最佳epoch及训练日志。阈值0.5，边界F1容忍距离2像素。','', '本结论限于本数据集与冻结协议；即使目标未达成，也必须保留并报告真实结果。']
    if all(budgets) and versions=={'strict_budget_v3'} and (10,'A',17) in by and (10,'B',17) in by:
        subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_cases.py')),
                        '--run-root',str(root),'--device','mps'],check=True)
        text += ['', '## 真实测试案例', '',
                 '案例按固定规则展示种子17的虚检、漏检及改善/退化极值，不代表总体平均。',
                 '见[真实案例与逐图指标](case_analysis/案例分析.md)。案例重新推理的混淆矩阵必须与原始测试记录完全一致。']
    (root/'研究结果报告.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(assessment,ensure_ascii=False))
if __name__=='__main__':main()
