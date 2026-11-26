"""核验原图划分后，按原图对生成嵌套标注清单。"""
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image
from .utils import digest, write_json


def prepare(root, output, seeds=(17, 42, 2026), fractions=(0.05, 0.1, 0.2, 1.0), expected_counts=None):
    root, output = Path(root).resolve(), Path(output)
    splits, fingerprints, seen = {}, {}, {}
    label_value_counts, label_non_binary_files = {}, []
    for split in ('train', 'val', 'test'):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError('缺少数据目录: ' + str(folder))
        sets = [{p.name for p in (folder / key).glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')} for key in ('A', 'B', 'label')]
        if not sets[0] or not sets[0] == sets[1] == sets[2]:
            raise ValueError(split + '的A/B/label文件名不一致或为空。')
        records = []
        for name in sorted(sets[0]):
            paths = [folder / key / name for key in ('A', 'B', 'label')]
            with Image.open(paths[0]) as a, Image.open(paths[1]) as b, Image.open(paths[2]) as m:
                if not a.size == b.size == m.size:
                    raise ValueError('尺寸不一致: ' + str(paths[0]))
                if min(a.size) < 16:
                    raise ValueError('图像太小')
                raw_values, raw_counts = np.unique(np.asarray(m.convert('L')), return_counts=True)
                values = set(raw_values.tolist())
                observed = {int(v):int(n) for v,n in zip(raw_values,raw_counts)}
                for v,n in observed.items():label_value_counts[v]=label_value_counts.get(v,0)+n
                if not values <= {0,1,255}:label_non_binary_files.append({'file':str(paths[2]),'values':observed})
                if not values <= {0, 1, 156, 254, 255}:
                    raise ValueError('非二值标签: ' + str(paths[2]))
            image_hashes = sorted((digest(paths[0]), digest(paths[1])))
            key = ':'.join(image_hashes)
            if key in seen:
                raise ValueError('重复影像对（含时相交换）: ' + seen[key] + ' / ' + split + '/' + name)
            seen[key] = split + '/' + name
            records.append(dict(zip(('a', 'b', 'label'), map(str, paths)), name=name))
            fingerprints[split + '/' + name] = {'a': digest(paths[0]), 'b': digest(paths[1]), 'label': digest(paths[2])}
        splits[split] = records
    actual_counts = {s: len(r) for s, r in splits.items()}
    if expected_counts is not None and actual_counts != expected_counts:
        raise ValueError('原始LEVIR-CD集合数量不符: ' + str(actual_counts))
    write_json(output / 'val.json', splits['val'])
    write_json(output / 'test.json', splits['test'])
    audit = {'root': str(root), 'unit': 'original_image_pair', 'rounding': 'floor',
             'counts': actual_counts, 'standard_levir_counts': actual_counts == {'train':445,'val':64,'test':128}, 'seeds': list(seeds),
             'fractions': list(fractions), 'fingerprints': fingerprints,
             'label_rule':'0 unchanged, every nonzero value changed; source files unchanged',
             'label_value_counts':label_value_counts,'label_non_binary_files':label_non_binary_files,
             'limitation': 'Exact duplicate checks do not establish geographic independence.'}
    for seed in seeds:
        order = list(range(len(splits['train'])))
        random.Random(seed).shuffle(order)
        for fraction in fractions:
            n = math.floor(len(order) * fraction)
            labeled = [splits['train'][i] for i in order[:n]]
            unlabeled = [{k: v for k, v in splits['train'][i].items() if k != 'label'} for i in order[n:]]
            target = output / str(seed) / str(round(fraction * 100))
            write_json(target / 'labeled.json', labeled)
            write_json(target / 'unlabeled.json', unlabeled)
    write_json(output / 'audit.json', audit)
    print({s: len(r) for s, r in splits.items()}, flush=True)
