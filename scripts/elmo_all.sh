#!/bin/bash
python ../train/elmo_train.py --dataset=train

python ../train/elmo_train.py --dataset=rd_alpha_0.1_num_aug_1 
python ../train/elmo_train.py --dataset=ri_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=rs_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=sr_alpha_0.1_num_aug_1

python ../train/elmo_train.py --dataset=rd_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=ri_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=rs_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=sr_alpha_0.3_num_aug_1

python ../train/elmo_train.py --dataset=rd_alpha_0.5_num_aug_1
python ../train/elmo_train.py --dataset=ri_alpha_0.5_num_aug_1
python ../train/elmo_train.py --dataset=rs_alpha_0.5_num_aug_1
python ../train/elmo_train.py --dataset=sr_alpha_0.5_num_aug_1

python ../train/elmo_train.py --dataset=rd_alpha_0.1_num_aug_3
python ../train/elmo_train.py --dataset=ri_alpha_0.1_num_aug_3
python ../train/elmo_train.py --dataset=rs_alpha_0.1_num_aug_3
python ../train/elmo_train.py --dataset=sr_alpha_0.1_num_aug_3

python ../train/elmo_train.py --dataset=rd_alpha_0.3_num_aug_3
python ../train/elmo_train.py --dataset=ri_alpha_0.3_num_aug_3
python ../train/elmo_train.py --dataset=rs_alpha_0.3_num_aug_3
python ../train/elmo_train.py --dataset=sr_alpha_0.3_num_aug_3

python ../train/elmo_train.py --dataset=rd_alpha_0.5_num_aug_3
python ../train/elmo_train.py --dataset=ri_alpha_0.5_num_aug_3
python ../train/elmo_train.py --dataset=rs_alpha_0.5_num_aug_3
python ../train/elmo_train.py --dataset=sr_alpha_0.5_num_aug_3

python ../train/elmo_train.py --dataset=rd_alpha_0.1_num_aug_5
python ../train/elmo_train.py --dataset=ri_alpha_0.1_num_aug_5
python ../train/elmo_train.py --dataset=rs_alpha_0.1_num_aug_5
python ../train/elmo_train.py --dataset=sr_alpha_0.1_num_aug_5

python ../train/elmo_train.py --dataset=rd_alpha_0.3_num_aug_5
python ../train/elmo_train.py --dataset=ri_alpha_0.3_num_aug_5
python ../train/elmo_train.py --dataset=rs_alpha_0.3_num_aug_5
python ../train/elmo_train.py --dataset=sr_alpha_0.3_num_aug_5

python ../train/elmo_train.py --dataset=rd_alpha_0.5_num_aug_5
python ../train/elmo_train.py --dataset=ri_alpha_0.5_num_aug_5
python ../train/elmo_train.py --dataset=rs_alpha_0.5_num_aug_5
python ../train/elmo_train.py --dataset=sr_alpha_0.5_num_aug_5

python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_1
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_3
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_3
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_3
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.3_num_aug_5
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.5_num_aug_5
python ../train/elmo_train.py --dataset=sr_ri_rs_rd_alpha_0.1_num_aug_5

python ../train/elmo_train.py --dataset=backtranslation_de
python ../train/elmo_train.py --dataset=backtranslation_es
python ../train/elmo_train.py --dataset=backtranslation_ja