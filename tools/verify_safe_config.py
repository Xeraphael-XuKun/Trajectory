"""Fail fast if the performance-safe config drifts from the verified recipe."""

import argparse
from pathlib import Path

import yaml


def load_result_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping: {}".format(path))
    # The destination directory cannot change the training calculation graph.
    data.pop("OUTPUT_DIR", None)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/vpr_whu.yml")
    parser.add_argument(
        "--reference", default="configs/hihr_whu_text_lam50_clip.yml"
    )
    args = parser.parse_args()

    candidate = load_result_config(args.config)
    reference = load_result_config(args.reference)
    if candidate != reference:
        raise SystemExit(
            "refusing safe-mode run: {} differs from the verified recipe {} "
            "beyond OUTPUT_DIR; use CONFIG=configs/vpr_whu_experimental.yml "
            "for an explicit experiment".format(args.config, args.reference)
        )
    print("safe config verified against {}".format(args.reference))


if __name__ == "__main__":
    main()
