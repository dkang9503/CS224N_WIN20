#!/bin/bash
python ../train/train.py --dataset=train

python ../train/train.py --dataset=rd_alpha_0.1_num_aug_1 
python ../train/train.py --dataset=ri_alpha_0.1_num_aug_1
python ../train/train.py --dataset=rs_alpha_0.1_num_aug_1
python ../train/train.py --dataset=sr_alpha_0.1_num_aug_1

python ../train/train.py --dataset=rd_alpha_0.3_num_aug_1
python ../train/train.py --dataset=ri_alpha_0.3_num_aug_1
python ../train/train.py --dataset=rs_alpha_0.3_num_aug_1
python ../train/train.py --dataset=sr_alpha_0.3_num_aug_1

python ../train/train.py --dataset=rd_alpha_0.5_num_aug_1
python ../train/train.py --dataset=ri_alpha_0.5_num_aug_1
python ../train/train.py --dataset=rs_alpha_0.5_num_aug_1
python ../train/train.py --dataset=sr_alpha_0.5_num_aug_1

python ../train/train.py --dataset=rd_alpha_0.1_num_aug_3
python ../train/train.py --dataset=ri_alpha_0.1_num_aug_3
python ../train/train.py --dataset=rs_alpha_0.1_num_aug_3
python ../train/train.py --dataset=sr_alpha_0.1_num_aug_3

python ../train/train.py --dataset=rd_alpha_0.3_num_aug_3
python ../train/train.py --dataset=ri_alpha_0.3_num_aug_3
python ../train/train.py --dataset=rs_alpha_0.3_num_aug_3
python ../train/train.py --dataset=sr_alpha_0.3_num_aug_3

python ../train/train.py --dataset=rd_alpha_0.5_num_aug_3
python ../train/train.py --dataset=ri_alpha_0.5_num_aug_3
python ../train/train.py --dataset=rs_alpha_0.5_num_aug_3
python ../train/train.py --dataset=sr_alpha_0.5_num_aug_3

python ../train/train.py --dataset=rd_alpha_0.1_num_aug_5
python ../train/train.py --dataset=ri_alpha_0.1_num_aug_5
python ../train/train.py --dataset=rs_alpha_0.1_num_aug_5
python ../train/train.py --dataset=sr_alpha_0.1_num_aug_5

python ../train/train.py --dataset=rd_alpha_0.3_num_aug_5
python ../train/train.py --dataset=ri_alpha_0.3_num_aug_5
python ../train/train.py --dataset=rs_alpha_0.3_num_aug_5
python ../train/train.py --dataset=sr_alpha_0.3_num_aug_5

python ../train/train.py --dataset=rd_alpha_0.5_num_aug_5
python ../train/train.py --dataset=ri_alpha_0.5_num_aug_5
python ../train/train.py --dataset=rs_alpha_0.5_num_aug_5
python ../train/train.py --dataset=sr_alpha_0.5_num_aug_5

python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_1
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_1
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_1
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_3
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_3
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_3
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_5
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_5
python ../train/train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_5

python ../train/train.py --dataset=backtranslation_de
python ../train/train.py --dataset=backtranslation_es
python ../train/train.py --dataset=backtranslation_ja