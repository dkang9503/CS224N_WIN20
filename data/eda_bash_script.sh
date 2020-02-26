#!/bin/bash
python eda_script.py --alpha=.5 --num_aug=1 --output=eda_alpha5e-1_num_aug1_train.csv --input=train.csv
python eda_script.py --alpha=.5 --num_aug=2 --output=eda_alpha5e-1_num_aug2_train.csv --input=train.csv
python eda_script.py --alpha=.5 --num_aug=3 --output=eda_alpha5e-1_num_aug3_train.csv --input=train.csv
python eda_script.py --alpha=.5 --num_aug=4 --output=eda_alpha5e-1_num_aug4_train.csv --input=train.csv
python eda_script.py --alpha=.35 --num_aug=5 --output=eda_alpha5e-1_num_aug5_train.csv --input=train.csv
