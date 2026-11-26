"""Resolve checkout-specific paths before invoking the historical queue."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'versions/strict_budget_v3/configs/base.json'
RUNTIME = ROOT / 'work/reproduction_config.json'


def main():
    config = json.loads(BASE.read_text())
    config.update(manifest_dir=str(ROOT / 'data/manifests_strict'), run_root=str(ROOT / 'runs_strict_v3'))
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME.write_text(json.dumps(config, indent=2) + '\n')
    return subprocess.call([sys.executable, str(ROOT / 'scripts/research_queue_strict.py'),
                            '--config', str(RUNTIME), '--full', *sys.argv[1:]], cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
