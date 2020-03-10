#!/bin/bash
python ../train/train.py --dataset=aug2_data/top4_merged_500
python ../train/train.py --dataset=aug2_data/top4_merged_1000
python ../train/train.py --dataset=aug2_data/top4_merged_5000
python ../train/train.py --dataset=aug2_data/top4_merged_10000
python ../train/train.py --dataset=aug2_data/top4_merged_20000
python ../train/train.py --dataset=aug2_data/top4_merged_30448