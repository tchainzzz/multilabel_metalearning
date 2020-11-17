#!/bin/bash
set -x
python -u maml.py --data-root ../cs330-storage --support-size 16 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment-name support_16
python -u maml.py --data-root ../cs330-storage --support-size 16 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name  support_16_test
python -u maml.py --data-root ../cs330-storage --support-size 32 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment-name support_32
python -u maml.py --data-root ../cs330-storage --support-size 32 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name support_32_test
python -u maml.py --data-root ../cs330-storage --support-size 4 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment-name support_4
python -u maml.py --data-root ../cs330-storage --support-size 4 --label-subset-size 3 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name support_4_test
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 4 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment_name label_subset_size_4
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 4 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name label_subset_size_4_test
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 5 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment-name label_subset_size_5
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 5 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name label_subset_size_5_test
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 6 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --experiment-name label_subset_size_6
python -u maml.py --data-root ../cs330-storage --support-size 8 --label-subset-size 6 --multilabel-scheme binary --sampling-mode permutation --log-frequency 10 --iterations 10000 --learn-inner-lr --test --experiment-name label_subset_size_6_test

