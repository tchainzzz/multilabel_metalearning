#!/bin/bash
set -x
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr
