"""命令行入口：只负责配置与编排，算法放在cd目录。"""
import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
import torch
from cd.utils import read_json, write_json, device_for

ROOT = Path(__file__).resolve().parent


def load_config(args):
    c = read_json(args.config)
    for key in ('seed', 'percent', 'group', 'device', 'epochs', 'steps_per_epoch', 'batch_size'):
        v = getattr(args, key, None)
        if v is not None:
            c[key] = v
    for key in ('manifest_dir', 'run_root'):
        c[key] = str((ROOT / c[key]).resolve())
    if c['group'] not in ('A', 'B', 'C', 'D') or c['percent'] not in (5, 10, 20, 100):
        raise ValueError('不支持的实验组或标注比例。')
    if not 0 <= c['pseudo_low'] < 0.5 < c['pseudo_high'] <= 1:
        raise ValueError('伪标签阈值需分处0.5两侧。')
    for k in ('epochs', 'steps_per_epoch', 'batch_size', 'crop', 'tile'):
        if c[k] <= 0:
            raise ValueError(k + '必须为正。')
    if c['crop'] % 16 or c['tile'] % 16:
        raise ValueError('crop和tile须为16的倍数。')
    c['resume'] = bool(getattr(args, 'resume', False))
    return c


def summarize(run_root):
    rows = []
    for path in sorted(Path(run_root).glob('*/test.json')):
        r = read_json(path)
        c = read_json(path.parent/'config.json')
        rows.append(dict(percent=c['percent'], group=c['group'], seed=c['seed'], **{k:r[k] for k in ('f1', 'iou', 'precision', 'recall', 'false_positive_rate', 'false_negative_rate', 'boundary_f1')}, run=str(path.parent)))
    if not rows:
        raise FileNotFoundError('没有真实test.json，不能生成实验结果。')
    target = Path(run_root)/'summary.csv'
    with target.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r['percent'],r['group']), []).append(r)
    summary = []
    for (p,g), rs in sorted(groups.items()):
        item = {'percent':p, 'group':g, 'seeds':[r['seed'] for r in rs], 'runs':len(rs), 'at_least_three_seeds':len(rs)>=3}
        for k in ('f1', 'iou', 'boundary_f1', 'false_positive_rate', 'false_negative_rate'):
            values = [r[k] for r in rs if r[k] is not None]
            item[k] = {'mean':statistics.mean(values) if values else None, 'std_sample':statistics.stdev(values) if len(values)>1 else None, 'valid_runs':len(values)}
        summary.append(item)
    by = {(r['percent'],r['group'],r['seed']):r for r in rows}
    comparisons = []
    for p in sorted({r['percent'] for r in rows}):
        for g1,g0 in (('B','A'),('C','A'),('D','A'),('D','B'),('D','C')):
            seeds = sorted(s for pp,g,s in by if pp==p and g==g1 and (p,g0,s) in by)
            values = [100*(by[p,g1,s]['f1']-by[p,g0,s]['f1']) for s in seeds if by[p,g1,s]['f1'] is not None and by[p,g0,s]['f1'] is not None]
            if values:
                mean = statistics.mean(values)
                comparisons.append({'percent':p,'comparison':g1+'-'+g0,'paired_seeds':seeds,'delta_f1_percentage_points':values,'mean':mean,
                                    'target_met':mean>=3 if p==10 and g1=='B' and g0=='A' and len(values)>=3 else None})
    write_json(Path(run_root)/'summary.json', {'groups':summary,'paired_comparisons':comparisons})
    print(target)


def main():
    parser = argparse.ArgumentParser(description='少标注遥感建筑变化检测')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('environment')
    prep = sub.add_parser('prepare'); prep.add_argument('--data-root', required=True); prep.add_argument('--output', default=str(ROOT/'data/manifests'))
    prep.add_argument('--allow-nonstandard-data', action='store_true', help='仅用于显式构造数据的软件检查')
    tr = sub.add_parser('train'); tr.add_argument('--config', default=str(ROOT/'configs/base.json'))
    for name, kind in [('seed',int),('percent',int),('epochs',int),('steps-per-epoch',int),('batch-size',int),('group',str),('device',str)]:
        tr.add_argument('--'+name, type=kind)
    tr.add_argument('--resume', action='store_true')
    ev = sub.add_parser('evaluate'); ev.add_argument('--checkpoint', required=True); ev.add_argument('--split', choices=['val','test'], default='val'); ev.add_argument('--device', default='auto'); ev.add_argument('--brightness', type=float, default=1.0)
    matrix = sub.add_parser('matrix'); matrix.add_argument('--config', default=str(ROOT/'configs/base.json')); matrix.add_argument('--execute', action='store_true'); matrix.add_argument('--scan', action='store_true')
    sm = sub.add_parser('summarize'); sm.add_argument('--run-root', default=str(ROOT/'runs'))
    args = parser.parse_args()
    if args.command == 'environment':
        info = {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'cuda_available':torch.cuda.is_available()}
        write_json(ROOT/'work/environment.json',info); print(json.dumps(info,indent=2))
    elif args.command == 'prepare':
        from cd.prepare import prepare
        prepare(args.data_root,args.output,expected_counts=None if args.allow_nonstandard_data else {'train':445,'val':64,'test':128})
    elif args.command == 'train':
        from cd.train import train
        train(load_config(args))
    elif args.command == 'evaluate':
        from cd.model import ChangeDetector
        from cd.evaluate import evaluate
        ck = Path(args.checkpoint)
        state = torch.load(ck,map_location='cpu',weights_only=True)
        c = state['config']; device = device_for(args.device)
        name = args.split + ('' if args.brightness==1 else '_brightness'+str(args.brightness)) + '.json'
        output = ck.parent/name
        if output.exists():
            raise FileExistsError('评价结果已存在，避免反复测试或覆盖: '+str(output))
        if args.split=='test' and not (ck.parent/'training_complete.json').exists():
            raise ValueError('训练尚未完整结束，禁止测试集评价。')
        if args.brightness <= 0:
            raise ValueError('brightness必须为正。')
        model=ChangeDetector().to(device); model.load_state_dict(state['model'])
        r=evaluate(model,read_json(Path(c['manifest_dir'])/(args.split+'.json')),device,c['tile'],c['threshold'],c['boundary_tolerance'],ck.parent/(name[:-5]+'_visuals'),brightness=args.brightness)
        r.update(checkpoint=str(ck.resolve()),selected_epoch=state['epoch'])
        write_json(output,r); print({k:v for k,v in r.items() if k!='per_image'})
    elif args.command=='matrix':
        runs=[(100,'A')]+[(p,g) for p in ([5,10,20] if args.scan else [10]) for g in 'ABCD']
        commands=[]
        for p,g in runs:
            for seed in [17,42,2026]:
                command=[sys.executable,str(ROOT/'main.py'),'train','--config',args.config,'--percent',str(p),'--group',g,'--seed',str(seed)]
                commands.append(command)
        write_json(ROOT/'work/experiment_commands.json', commands)
        for cmd in commands:
            print(' '.join(cmd),flush=True)
            if args.execute:
                subprocess.run(cmd,cwd=ROOT,check=True)
    elif args.command=='summarize':
        summarize(args.run_root)


if __name__=='__main__':
    main()
