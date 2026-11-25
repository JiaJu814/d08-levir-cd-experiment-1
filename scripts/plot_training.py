# -*- coding: utf-8 -*-
"""读取真实验证记录绘图，缺失种子留空；不生成替代数据。"""
import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',default=str(ROOT/'runs_strict_v3'));args=p.parse_args()
    root=Path(args.run_root)
    available={f.name for f in font_manager.fontManager.ttflist}
    for font in ('PingFang SC','Heiti TC','Noto Sans CJK SC','SimHei'):
        if font in available:plt.rcParams['font.sans-serif']=[font];break
    else:raise RuntimeError('缺少中文字体。')
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    found=False
    for ax,seed in zip(axes,(17,42,2026)):
        for g,label in (('A','纯监督A'),('B','伪标签一致性B')):
            file=root/f'10pct_{g}_seed{seed}'/'history.jsonl'
            if not file.exists():continue
            history=[]
            for line in file.read_text().splitlines():
                try:history.append(json.loads(line))
                except json.JSONDecodeError:continue
            rs=[r for r in history if r['val'] is not None]
            if rs:
                ax.plot([r['epoch'] for r in rs],[100*r['val']['f1'] for r in rs],'o-',ms=3,label=label)
                found=True
        ax.set(title='种子'+str(seed),xlabel='训练轮次',xlim=(0,100),ylim=(0,100))
        ax.axvline(10,color='gray',ls=':',alpha=.6)
        ax.grid(alpha=.2)
        if ax.lines and len(ax.lines)>1:ax.legend(fontsize=9)
        else:ax.text(.3,.45,'尚无验证记录',transform=ax.transAxes)
    if not found:plt.close(fig);raise RuntimeError('没有真实验证记录，拒绝生成曲线。')
    axes[0].set_ylabel('验证集变化类F1（%）')
    fig.suptitle('10%预算主对照的验证曲线（不能替代最终测试目标判定）',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    out=root/'figures';out.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(out/('training_curves.'+suffix),dpi=180)
    plt.close(fig)
    print(out/'training_curves.png')
if __name__=='__main__':main()
