"""只读取真实汇总结果绘图；没有结果时退出，不生成示例成绩。"""
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/"work/matplotlib"))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',default=str(ROOT/'runs_strict_v3'));args=parser.parse_args()
    run_root=Path(args.run_root)
    source=run_root/'summary.json'
    if not source.exists():raise FileNotFoundError('先完成真实测试评价和summarize。')
    available={f.name for f in font_manager.fontManager.ttflist}
    for font in ['PingFang SC','Heiti TC','Noto Sans CJK SC','SimHei']:
        if font in available:
            plt.rcParams['font.sans-serif']=[font];break
    else:raise RuntimeError('请安装中文字体后绘图。')
    plt.rcParams['axes.unicode_minus']=False
    rows=json.loads(source.read_text())['groups']
    target=run_root/'figures';target.mkdir(parents=True,exist_ok=True)
    for metric,label in [('f1','变化类F1'),('iou','变化类IoU'),('boundary_f1','边界F1')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for group,name in zip('ABCD',['纯监督','伪标签一致性','边界监督','组合方案']):
            rs=sorted([r for r in rows if r['group']==group and r['percent']<100 and r[metric]['mean'] is not None],key=lambda r:r['percent'])
            if not rs:continue
            x=[r['percent'] for r in rs];y=[100*r[metric]['mean'] for r in rs]
            errors=[100*r[metric]['std_sample'] if r[metric]['std_sample'] is not None else np.nan for r in rs]
            line,=ax.plot(x,y,'o-',label=name)
            if all(np.isfinite(errors)):
                ax.errorbar(x,y,yerr=errors,fmt='none',capsize=3,
                            ecolor=line.get_color(),alpha=.65)
        reference=[r for r in rows if r['percent']==100 and r['group']=='A' and r[metric]['mean'] is not None]
        if reference:
            ax.axhline(100*reference[0][metric]['mean'],ls='--',color='gray',label='全部标注参考')
        ax.set_title('5% / 10% / 20%预算分别使用22 / 44 / 89对影像\n三种子均值；误差棒为样本标准差',fontsize=10)
        ax.set(xlabel='训练影像对标注预算上限（%）',ylabel=label+'（%）',xticks=[5,10,20]);ax.grid(alpha=.25)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=3,fontsize=8)
        fig.tight_layout()
        fig.savefig(target/(metric+'.png'),dpi=220);fig.savefig(target/(metric+'.pdf'));plt.close(fig)

    subprocess.run([sys.executable,str(ROOT/'scripts/plot_training.py'),'--run-root',str(run_root)],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_completion.py'),
                    '--run-root',str(run_root),'--allow-incomplete'],check=True)
    verification=json.loads((run_root/'completion_verification.json').read_text())
    if verification['errors']:raise RuntimeError('独立验收发现错误，请检查completion_verification.json。')

if __name__=='__main__':main()
