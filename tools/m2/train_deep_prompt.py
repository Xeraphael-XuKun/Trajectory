"""Compatibility wrapper for the canonical M2-3 stage-A trainer."""
import argparse
import sys
from server.m2_03.prompt_stage import main as prompt_main

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=10000)
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], '--feature-cache', args.cache,
                '--clip', args.clip, '--output', args.output,
                '--steps', str(args.steps)] + rest
    prompt_main()
