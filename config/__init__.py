# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .defaults import _C as cfg
from .defaults import _C as cfg_test


def load_config(path, opts=()):
    from pathlib import Path
    current = cfg.clone()
    current.merge_from_other_cfg(current.load_cfg(Path(path).read_text(encoding='utf-8')))
    current.merge_from_list(list(opts))
    return current
