#!/bin/bash
set -x
python -u mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode permutation --multilabel-scheme powerset --log-frequency 10 --bs 16
python -u mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode greedy --multilabel-scheme powerset --log-frequency 10 --bs 16
python -u mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode greedy --multilabel-scheme binary --log-frequency 10 --bs 16
python -u mann.py --data-root ../cs330-storage/ --iterations 25000 --sampling-mode permutation --multilabel-scheme binary --log-frequency 10 --bs 16
