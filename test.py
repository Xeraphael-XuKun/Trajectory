"""加载同一 epoch-60 checkpoint，使用 before 或 after 独立检索。"""
import argparse
from config import load_config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from utils.runtime import prepare_run, runtime_summary


def main():
    parser = argparse.ArgumentParser(description='pre-BN / post-BN 独立测试')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = load_config(args.config_file, args.opts)
    cfg.freeze()
    prepare_run(cfg)
    logger = setup_logger('transreid', cfg.OUTPUT_DIR, if_train=False)
    logger.info(runtime_summary())
    logger.info('Running with config:\n%s', cfg)
    _, val_loaders, num_querys, num_classes = make_dataloader(cfg)
    # 测试直接完整加载训练权重，不先重复读取 CLIP 初始化权重。
    model_cfg = cfg.clone()
    model_cfg.defrost()
    model_cfg.MODEL.PRETRAIN_CHOICE = 'no'
    model_cfg.freeze()
    model = make_model(model_cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    do_inference(cfg, model, val_loaders, num_querys)


if __name__ == '__main__':
    main()
