#!/bin/bash
set -x
python mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode permutation --multilabel-scheme powerset --log-frequency 100
python mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode greedy --multilabel-scheme binary --log-frequency 100
python mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode permutation --multilabel-scheme binary --log-frequency 100
