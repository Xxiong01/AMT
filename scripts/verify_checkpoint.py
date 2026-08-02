#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main() -> None:
    ap = argparse.ArgumentParser('Verify released AMT-L48 checkpoint metadata.')
    ap.add_argument('--checkpoint', type=str, default='checkpoints/amt_l48/reid_best.pt')
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    obj = torch.load(ckpt, map_location='cpu')
    meta = obj.get('meta', {})
    model_cfg = meta.get('model_cfg', {})
    print('model_name:', meta.get('model_name'))
    print('seq_len:', meta.get('seq_len'))
    print('max_seq_len:', model_cfg.get('max_seq_len'))
    print('input_hw:', meta.get('input_hw'))
    print('crop_pad:', meta.get('crop_pad'))
    assert int(meta.get('seq_len')) == 48
    assert int(model_cfg.get('max_seq_len')) == 48
    print('OK: checkpoint metadata is AMT-L48.')


if __name__ == '__main__':
    main()
