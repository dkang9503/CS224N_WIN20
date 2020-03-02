#!/bin/bash
python ../train/bert_train.py --dataset=backtranslation_de
python ../train/bert_train.py --dataset=backtranslation_es
python ../train/bert_train.py --dataset=backtranslation_ja