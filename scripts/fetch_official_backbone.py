"""Restore the author's unmodified FC-Siam-diff file and verify its recorded digest."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://raw.githubusercontent.com/rcdaudt/fully_convolutional_change_detection/master/siamunet_diff.py'
SHA256 = '7fb2db177141865333d931ff24c00c35b9df60481ce2b978d3db674867f01e07'
DESTINATION = ROOT / 'versions/strict_budget_v3/cd/official_siamunet_diff.py'


def restore(source):
    if DESTINATION.exists():
        if hashlib.sha256(DESTINATION.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f'Existing file differs from the original digest: {DESTINATION}')
    else:
        if source is not None:
            contents = source.read_bytes()
        else:
            with urlopen(URL, timeout=30) as response:
                contents = response.read()
        if hashlib.sha256(contents).hexdigest() != SHA256:
            raise ValueError('The supplied FC-Siam-diff source does not match the recorded SHA-256.')
        DESTINATION.write_bytes(contents)
    print(DESTINATION)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Use an existing local copy of the original source.')
    restore(parser.parse_args().source)
