#!/bin/bash
python char_aug_script.py --input=../train.csv --da_type=ocr --num_aug=1 --alpha_char=0.1 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=ocr --num_aug=1 --alpha_char=0.3 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=ocr --num_aug=3 --alpha_char=0.1 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=ocr --num_aug=3 --alpha_char=0.3 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=key --num_aug=1 --alpha_char=0.1 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=key --num_aug=1 --alpha_char=0.3 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=key --num_aug=3 --alpha_char=0.1 --output_dir=../aug2_data/
python char_aug_script.py --input=../train.csv --da_type=key --num_aug=3 --alpha_char=0.3 --output_dir=../aug2_data/
