#!/bin/bash
python eda_script.py --alpha=.1 --num_aug=1 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.1 --num_aug=3 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.1 --num_aug=5 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.3 --num_aug=1 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.3 --num_aug=3 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.3 --num_aug=5 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.5 --num_aug=1 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.5 --num_aug=3 --input=train.csv --da_type=sr_ri_rs_rd
python eda_script.py --alpha=.5 --num_aug=5 --input=train.csv --da_type=sr_ri_rs_rd