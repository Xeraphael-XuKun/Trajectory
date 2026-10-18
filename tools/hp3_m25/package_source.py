"""生成本地源码包与真实提交记录；不上传，不包含凭据、数据或训练产物。"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
CODE = {'config', 'datasets', 'model', 'processor', 'solver', 'loss', 'utils'}
ENTRYPOINTS = {'train.py', 'finetune.py', 'test.py', 'tools/preflight.py'}


def selected(path):
    p = Path(path)
    return (p.parts[0] in CODE and p.suffix in ('.py', '.gz', '.txt') or
            path in ENTRYPOINTS or path.startswith(('tools/m2/', 'tools/hp3_m25/',
                                                   'server/hp3_m25_a800/')) or
            path == 'configs/hp3_m25_a800.yml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit], cwd=ROOT,
                                    text=True).splitlines()
    paths = [p for p in paths if selected(p)]
    if 'configs/hp3_m25_a800.yml' not in paths:
        raise RuntimeError('先提交A800实现，再生成源码包')
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT,
                                     text=True).splitlines()
    if any(selected(p) for p in changed):
        raise RuntimeError('源码包仅导出已提交代码，请先处理当前实现的未提交修改')
    metadata = dict(commit=commit, experiment='HP3_M25_A800_1018',
                    original_hp3='2c71241eda5387030957858ebaaa137c66828c92',
                    integration_parent='b01808d4e6e697170e000008174e8709bfc8a716', files=paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # x模式保留已有包，不允许静默替换。
    with tarfile.open(args.output, 'x:gz') as archive:
        for path in paths:
            data = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            entry.mode = 0o755 if path.endswith('.sh') else 0o644
            archive.addfile(entry, io.BytesIO(data))
        data = json.dumps(metadata, indent=2).encode('utf-8')
        entry = tarfile.TarInfo('DEPLOYMENT.json')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    print(json.dumps(dict(package=str(args.output.resolve()), commit=commit, files=len(paths))))


if __name__ == '__main__':
    main()
