"""前台正式训练入口：执行原 train.py，保留既有主训练链。"""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('train.py')), run_name='__main__')
