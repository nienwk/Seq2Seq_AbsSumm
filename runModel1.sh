#!/usr/bin/env bash

#mkdir logs/
python train.py -vd --pytorch-seed 15825621376512445961 --trainloader-seed 2975016263124821505 --no-benchmark --save 1 --debug | tee -a logs/train1.txt