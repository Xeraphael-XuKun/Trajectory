"""可选：检查八组配置只有指定因素不同，不参与正式启动。"""
import argparse
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import load_config as read_config

IDS = ['{}_{}_{}'.format(e, n, m) for e in ['W', 'L']
       for n in ['S', 'C'] for m in ['B', 'T']]


def load_config(name):
    return read_config(ROOT / 'configs' / (name + '.yml'))


def training_settings(data):
    settings = yaml.safe_load(data.dump()) if hasattr(data, 'dump') else json.loads(json.dumps(data))
    settings.pop('OUTPUT_DIR')
    settings.pop('EXPERIMENT')
    return settings


def verify():
    reference = None
    for name in IDS:
        c = load_config(name)
        assert c.EXPERIMENT.ID == name
        assert c.EXPERIMENT.ENVIRONMENT == ('whu_mars' if name[0] == 'W' else 'llmpar')
        assert c.MODEL.TOKEN_TRAJECTORY == name.endswith('T')
        assert not c.MODEL.TEXT_ALIGN and c.SOLVER.TEXT_LOSS_WEIGHT == 0
        assert c.SOLVER.CUDNN_BENCHMARK and c.SOLVER.CUDNN_DETERMINISTIC
        assert c.SOLVER.MAX_EPOCHS == 60 and c.TEST.NECK_FEAT == 'before'
        assert c.DATASETS.PROTOCOL == 'ALL' and c.DATASETS.SUBDIR == 'WHU-MARS'
        expected = ([0.5] * 3, [0.5] * 3) if name[2] == 'S' else (
            [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        assert (list(c.INPUT.PIXEL_MEAN), list(c.INPUT.PIXEL_STD)) == expected
        settings = training_settings(c)
        settings['MODEL'].pop('TOKEN_TRAJECTORY')
        settings['INPUT'].pop('PIXEL_MEAN')
        settings['INPUT'].pop('PIXEL_STD')
        if reference is None:
            reference = settings
        else:
            assert settings == reference, '发现未授权的配置差异：' + name
    print('MATRIX_CONFIG_OK：8 组只改变 method、normalization、environment 和输出目录。')


def main():
    parser = argparse.ArgumentParser(description='可选的八组配置检查')
    parser.parse_args()
    verify()


if __name__ == '__main__':
    main()
