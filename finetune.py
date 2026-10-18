"""Compatibility entry: execute the unchanged full train.py pipeline."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().with_name('train.py')), run_name='__main__')
