#!/bin/bash
python word_aug_script.py --input=../train.csv --da_type=glovetwit --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=glovetwit --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3
python word_aug_script.py --input=../train.csv --da_type=glovetwit --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=glovetwit --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3
python word_aug_script.py --input=../train.csv --da_type=contextin --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=contextin --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3
python word_aug_script.py --input=../train.csv --da_type=contextin --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=contextin --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3
python word_aug_script.py --input=../train.csv --da_type=contextsub --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=contextsub --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3
python word_aug_script.py --input=../train.csv --da_type=contextsub --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.1
python word_aug_script.py --input=../train.csv --da_type=contextsub --num_aug=3 --model_dir=../../models/saved_models/ --output_dir=../aug2_data/ --alpha=0.3


# python word_aug_script.py --input=../../data/aug2_data/contextsub_alpha_0.1_num_aug_1.csv --da_type=glovetwit --num_aug=1 --model_dir=../../models/saved_models/ --output_dir=../../data/aug2_data/multiple --alpha=0.3