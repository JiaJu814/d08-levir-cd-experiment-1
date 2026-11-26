import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def now():
    return time.time()

def code_fingerprints():
    files = list((ROOT/'scripts').glob('*.py')) + list((ROOT/'baseline/cd').glob('*.py'))
    files += [ROOT/'config.json', ROOT/'manifests/seed17_labeled_names.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}

def metric(tp, fp, fn, tn):
    def div(a,b): return a/b if b else None
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                f1=div(2*tp,2*tp+fp+fn), iou=div(tp,tp+fp+fn),
                precision=div(tp,tp+fp), recall=div(tp,tp+fn),
                false_positive_rate=div(fp,fp+tn), false_negative_rate=div(fn,fn+tp))

def safe_members(z):
    for info in z.infolist():
        name = getattr(info, 'orig_filename', info.filename)
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:
            raise ValueError('Unsafe ZIP member: '+info.filename)
        yield info
