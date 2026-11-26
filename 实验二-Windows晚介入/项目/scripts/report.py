import csv
import io
import json
import zipfile
from common import ROOT, RESULTS, read, write

def load(path,default=None):
    return read(path) if path.exists() else default

def build():
    RESULTS.mkdir(exist_ok=True)
    state=load(RESULTS/'task.json',{})
    report=dict(stage='diagnostic_only',formal_goal_achieved=False,test_evaluated=False,
                task=state,preflight=load(RESULTS/'preflight.json'),audit=load(RESULTS/'audit.json'),
                micro=load(RESULTS/'micro/result.json'),gate=load(RESULTS/'baseline_gate.json'),
                branches={k:load(RESULTS/k/'result.json') for k in ['sl10','sl100','semi10']})
    a=report['branches']['sl10'];b=report['branches']['semi10']
    if a and b and a['completed'] and b['completed']:
        report['single_seed_validation_gain_pp']=100*(b['best_val_f1']-a['best_val_f1'])
    else:report['single_seed_validation_gain_pp']=None
    write(RESULTS/'summary.json',report)
    lines=['# D08第一阶段真实执行报告','',
           '**本报告为单种子验证集诊断，不是三种子测试结果，不声明达到正式3点目标。**','',
           '## 执行状态','',f"- 状态：{state.get('status','unknown')}",
           f"- 开始时间戳：{state.get('start_epoch')}",f"- 固定截止时间戳：{state.get('deadline_epoch')}",
           f"- 结束时间戳：{state.get('end_epoch')}",f"- 原因：{state.get('reason','见事件日志')}",'',
           '## 方法与预算','',
           'FC-Siam-diff原网络；种子17；10%为44/445对人工标注；100%为445对监督参照。',
           '随机256裁剪、batch4、Adam、初始lr0.001、weight_decay0.0001、10000更新余弦衰减。',
           '监督损失为按各自有标签子集估计类别权重的BCE+Dice。A10/B10使用相同类别权重。',
           '可选半监督从相同初始化训练10000更新，1000更新预热、2000更新线性增权，EMA0.99，',
           '置信阈值0.95/0.05，类别归一化硬伪标签BCE，权重1，无边界项。',
           '这是项目半监督方案，不是Mean Teacher原论文MSE目标的逐项复现。',
           '10%和100%同更新数不代表同遍历次数或充分收敛；运行结果逐项记录更新数、见样量和耗时。','',
           '## 自动检查证据','',
           '- 环境与CUDA前向/反向：`preflight.json`、`environment/`',
           '- 数据配对、完整解码、掩膜取值、尺寸：`audit.json`',
           '- 指标已知输入检查：`metric_checks.json`',
           '- 小样本固定正样本裁剪学习：`micro/result.json`、`micro/history.json`',
           '- 监督检查门槛与剩余疑问：`baseline_gate.json`','',
           '## 实验结果','',
           '| 分支 | 完成固定更新数 | 已完成更新 | 最佳验证F1 | 见样量 | 耗时秒 |',
           '|---|---|---:|---:|---:|---:|']
    for k,r in report['branches'].items():
        if not r:
            lines.append(f'| {k} | 未完成或未启动，见events.jsonl | — | — | — | — |');continue
        f1=r.get('best_val_f1'); f1='未定义' if f1 is None else f'{100*f1:.4f}%'
        lines.append(f"| {k} | {r['completed']} | {r['steps']} | {f1} | {r['supervised_samples']} | {r['seconds']:.1f} |")
    lines += ['',f"单种子验证增益（百分点）：{report['single_seed_validation_gain_pp']}",'',
              '每个分支的配置、学习曲线、检查点和预测拼图位于同名目录；`history.json`是曲线原始证据。',
              '拼图顺序：时相A、时相B、真实变化、预测变化、误差；误差白色TP、红色FP、蓝色FN。','',
              '## 限制与下一步','',
              '1. 小样本检查通过仅说明能记忆两个固定裁剪，不能证明泛化或所有实现均正确；未通过也不能直接判定实现错误。',
              '2. 验证末段稳定门槛只是预先规定的扩展筛查，不构成收敛证明。100%表现异常时先审查训练充分性、采样和类别权重。',
              '3. 半监督未运行时，不评价其有效或无效；未完成的分支按已有检查点报告。',
              '4. 未对本阶段结果作测试集调参；历史项目曾查看测试集的事实仍需在最终研究中披露。',
              '5. 正式目标仍需至少三个配对种子及冻结协议后的评价；本阶段不开展边界和比例扫描。',
              '6. 不得用此模板文字替代证据。缺少文件或字段即标记未验证，禁止补造数值。','',
              '## 事件与修改证据','',
              '查看`events.jsonl`、`logs/`、`code_sha256.json`和`config_snapshot.json`。任何人工修复须另附差异和重验记录。']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for kind in ['sl10','sl100','semi10']:
            h=load(RESULTS/kind/'history.json',[])
            if h:
                axes[0].plot([r['step'] for r in h],[r['loss'] for r in h],label=kind)
                axes[1].plot([r['step'] for r in h],[100*r['val']['f1'] for r in h],label=kind)
        axes[0].set(xlabel='Updates',ylabel='Training loss');axes[1].set(xlabel='Updates',ylabel='Validation F1 (%)')
        for ax in axes:
            if ax.lines:ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(RESULTS/'learning_curves.png',dpi=160);plt.close(fig)
    except Exception as e:
        write(RESULTS/'plot_error.json',dict(error=repr(e),metrics_unaffected=True))
    target=ROOT/'RETURN_TO_CHATGPT.zip'
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RESULTS.rglob('*')):
            if p.is_file() and p.suffix.lower() in ['.json','.jsonl','.txt','.log','.md','.csv','.png']:
                z.write(p,p.relative_to(ROOT).as_posix())
    target.with_suffix('.tmp').replace(target)
    print('Report:',RESULTS/'REPORT.md',flush=True)

if __name__=='__main__':build()
