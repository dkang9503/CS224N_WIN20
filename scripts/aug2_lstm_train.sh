#!/bin/bash
python ../train/train.py --dataset=aug2_data/multiple/key_consub_combined
python ../train/train.py --dataset=aug2_data/multiple/ocr_consub_combined
python ../train/train.py --dataset=aug2_data/multiple/key_twit_combined
python ../train/train.py --dataset=aug2_data/multiple/ocr_twit_combined
python ../train/train.py --dataset=aug2_data/multiple/key_twit_consub_combined
python ../train/train.py --dataset=aug2_data/multiple/ocr_twit_consub_combined
python ../train/train.py --dataset=aug2_data/multiple/twit_consub_combined
