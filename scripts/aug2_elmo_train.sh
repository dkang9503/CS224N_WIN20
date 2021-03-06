#!/bin/bash
python ../train/elmo_train.py --dataset=aug2_data/contextin_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/contextin_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/contextsub_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/contextsub_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/glovetwit_alpha_0.1_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/glovetwit_alpha_0.1_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/glovetwit_alpha_0.3_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/glovetwit_alpha_0.3_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/key_alpha_0.25_alpha_char_0.1_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/key_alpha_0.25_alpha_char_0.1_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/key_alpha_0.25_alpha_char_0.3_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/key_alpha_0.25_alpha_char_0.3_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/ocr_alpha_0.25_alpha_char_0.1_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/ocr_alpha_0.25_alpha_char_0.1_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/ocr_alpha_0.25_alpha_char_0.3_num_aug_1
python ../train/elmo_train.py --dataset=aug2_data/ocr_alpha_0.25_alpha_char_0.3_num_aug_3
python ../train/elmo_train.py --dataset=aug2_data/sentence_gen_num_aug_1_model_gpt2
python ../train/elmo_train.py --dataset=aug2_data/sentence_gen_num_aug_3_model_gpt2
