#! /bin/bash

python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config-file config/GEMZSL/sun_4w_2s.yaml
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config-file config/GEMZSL/awa_4w_2s.yaml
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --config-file config/GEMZSL/cub_4w_2s.yaml