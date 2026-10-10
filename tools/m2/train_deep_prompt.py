"""兼容 --cache 入口；正式预算与配置由同一个阶段 A 实现读取。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from server.m2_03.prompt_stage import main

if __name__ == '__main__':
    sys.argv = ['--feature-cache' if value == '--cache' else value for value in sys.argv]
    main()
