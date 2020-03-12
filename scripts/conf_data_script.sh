#!/bin/bash
python ../train/train.py --dataset=train_conf80
python ../train/train.py --dataset=train_conf100
#python ../train/elmo_train.py --dataset=train_conf80
#python ../train/elmo_train.py --dataset=train_conf100
#python ../train/bert_train.py --dataset=train_conf80
#python ../train/bert_train.py --dataset=train_conf100