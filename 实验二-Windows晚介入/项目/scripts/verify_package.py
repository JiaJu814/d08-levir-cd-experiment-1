"""Standard-library transfer integrity gate; manifest is not a digital signature."""
from common import ROOT,RESULTS,read,write,sha

def main():
    manifest=read(ROOT/'file_manifest.json');fail=[]
    for name,expected in manifest['files'].items():
        p=ROOT/name
        if not p.is_file():fail.append(dict(path=name,error='missing'));continue
        if p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:
            fail.append(dict(path=name,error='size_or_hash_mismatch'))
    write(RESULTS/'package_integrity.json',dict(passed=not fail,checked=len(manifest['files']),failures=fail))
    if fail:raise RuntimeError('Package transfer integrity failed; do not run experiments')
    print('PACKAGE INTEGRITY PASSED',len(manifest['files']))

if __name__=='__main__':main()
